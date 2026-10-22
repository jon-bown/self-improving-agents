"""SI_AGENT=openclaw: OpenClaw, with the instructions under test rendered as its AGENTS.md.

Each call runs `openclaw agent --local --json` in a fresh session against an isolated profile
in .selfimprove/openclaw/ (OPENCLAW_STATE_DIR), so ~/.openclaw and any gateway you already run
are never touched. The workspace (AGENTS.md + SI_WORKSPACE_DIR files) is selected per call with
OPENCLAW_WORKSPACE_DIR. OpenClaw reads that variable only when the config names no workspace for
the agent (agents.defaults.workspace or agents.list[].workspace win over it), so the isolated
profile deliberately leaves both unset.

`selfimprove doctor` writes the isolated config the first time. With SI_OPENCLAW_MLFLOW=1 it
also builds and installs the MLflow tracing plugin from connectors/openclaw-mlflow into that
profile. OpenClaw then logs its own trace for every run (AGENT -> LLM / TOOL spans, token usage)
to the same experiment. Those traces carry no si_source tag, so the loop never reads them as
production traffic; the harness span records `openclaw.session_key` to find the matching one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from selfimprove import config, models

PLUGIN_ID = "mlflow-openclaw"
# harness/selfimprove/agents/openclaw.py -> <repo>/connectors/openclaw-mlflow
PLUGIN_DIR = Path(__file__).resolve().parents[3] / "connectors" / "openclaw-mlflow"


def state_dir():
    return config.settings.state_dir / "openclaw"


def env(workspace=None) -> dict[str, str]:
    e = dict(os.environ)
    e["OPENCLAW_STATE_DIR"] = str(state_dir())
    e["OPENCLAW_CONFIG_PATH"] = str(state_dir() / "openclaw.json")
    if workspace is not None:
        e["OPENCLAW_WORKSPACE_DIR"] = str(workspace)
    return e


def write_config() -> str:
    """The isolated profile: AGENTS.md is the only instruction file, read-only tools.

    No `workspace` key: a configured one overrides OPENCLAW_WORKSPACE_DIR, and every call would
    then run against the same directory instead of the instructions under test. Without it,
    runs that set no OPENCLAW_WORKSPACE_DIR (`config validate`) fall back to <state>/workspace.
    """
    state_dir().mkdir(parents=True, exist_ok=True)
    cfg = {
        "agents": {"defaults": {
            "model": {"primary": models.openclaw_model(config.settings.agent_model)},
            "skipBootstrap": True,  # no SOUL.md/BOOTSTRAP.md: the Prompt Registry defines behavior
        }},
        "tools": {"allow": ["read"]},
        # Not your gateway's port (18789): a CLI --local run never needs one, but if a gateway is
        # ever started from this profile it stays on loopback, out of the way.
        "gateway": {"mode": "local", "port": 19555, "bind": "loopback"},
    }
    path = state_dir() / "openclaw.json"
    # Keep what OpenClaw itself added (plugins, install records); only the keys above are ours.
    existing = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**existing, **cfg}, indent=2))
    return str(path)


def _config_current() -> bool:
    """False for a missing profile, or one written before the workspace key was dropped."""
    path = state_dir() / "openclaw.json"
    if not path.is_file():
        return False
    agents = json.loads(path.read_text()).get("agents", {})
    return "workspace" not in agents.get("defaults", {}) and "list" not in agents


def plugin_installed() -> bool:
    return (state_dir() / "extensions" / PLUGIN_ID / "dist" / "index.js").is_file()


def install_plugin(experiment_id: str) -> None:
    """Build connectors/openclaw-mlflow and install it into the isolated profile."""
    if not (PLUGIN_DIR / "install.sh").is_file():
        raise FileNotFoundError(f"{PLUGIN_DIR}/install.sh not found: run from a checkout of this repo")
    subprocess.run([str(PLUGIN_DIR / "install.sh"), "--tracking-uri", config.settings.tracking_uri,
                    "--experiment-id", experiment_id],
                   env={**env(), "OPENCLAW_BIN": config.settings.openclaw_bin}, check=True)


def plugin_check(experiment_id: str) -> str:
    if not config.settings.openclaw_mlflow:
        return "off (SI_OPENCLAW_MLFLOW=1 adds OpenClaw's own LLM/tool spans)"
    if not config.settings.tracking_uri.startswith(("http://", "https://", "databricks")):
        return (f"skipped: the plugin exports over HTTP, but MLFLOW_TRACKING_URI is "
                f"{config.settings.tracking_uri} (run `mlflow server` and point .env at it)")
    if not plugin_installed():
        install_plugin(experiment_id)
    return f"installed in the isolated profile, exporting to experiment {experiment_id}"


def check() -> str:
    binary = config.settings.openclaw_bin
    if not shutil.which(binary):
        return "openclaw not found on PATH (npm i -g openclaw)"
    if not (state_dir() / "openclaw.json").is_file():
        write_config()
    out = subprocess.run([binary, "config", "validate"], env=env(), capture_output=True, text=True)
    return f"isolated profile {state_dir()} is " + ("valid" if out.returncode == 0 else f"INVALID: {out.stderr[-300:]}")


def _reply(payload: dict) -> str:
    """`openclaw agent --json` has shipped a few envelope shapes; accept all of them."""
    if isinstance(payload.get("final"), str):
        return payload["final"]
    for key in ("payloads", "messages"):
        items = payload.get(key) or (payload.get("result") or {}).get(key) or []
        texts = [i.get("text", "") for i in items if isinstance(i, dict)]
        if any(texts):
            return "\n".join(t for t in texts if t)
    for key in ("text", "reply", "output"):
        if isinstance(payload.get(key), str):
            return payload[key]
    raise ValueError(f"no reply in OpenClaw output: {json.dumps(payload)[:400]}")


def call_openclaw(question, instructions, workspace):
    if not _config_current():
        write_config()
    session_id = f"si-{uuid.uuid4().hex[:12]}"
    cmd = [config.settings.openclaw_bin, "agent", "--local", "--agent", "main",
           "--session-id", session_id,
           "--model", models.openclaw_model(config.settings.agent_model),
           "--thinking", "off", "--message", question, "--json"]
    with mlflow.start_span("openclaw.agent", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"message": question, "workspace": str(workspace)})
        # The MLflow plugin's trace for this run has the same mlflow.trace.session.
        span.set_attribute("openclaw.session_key", f"agent:main:explicit:{session_id}")
        proc = subprocess.run(cmd, env=env(workspace), capture_output=True, text=True,
                              timeout=config.settings.agent_timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"openclaw exited {proc.returncode}: {proc.stderr[-1500:]}")
        payload = json.loads(proc.stdout)
        meta = payload.get("meta", {})
        tools = payload.get("toolSummary") or meta.get("toolSummary") or {}
        span.set_attributes({"openclaw.tool_calls": tools.get("calls", 0),
                             "openclaw.tools": ",".join(tools.get("tools", []))})
        text = _reply(payload)
        span.set_outputs({"text": text})
        return text
