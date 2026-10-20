"""Create an isolated OpenClaw profile for the demo (never touches ~/.openclaw)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from sia import config, llm
from sia.agent import openclaw_env


def base_config(workspace: Path, model: str | None = None) -> dict:
    """The demo's OpenClaw settings: one workspace, one model, read-only tools, no default port."""
    model = model or config.AGENT_MODEL
    cfg: dict = {
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "model": {"primary": llm.openclaw_model(model)},
                # Don't create SOUL.md / BOOTSTRAP.md etc.: AGENTS.md is the only instruction file,
                # so the Prompt Registry fully determines the agent's behavior.
                "skipBootstrap": True,
            }
        },
        # A support agent only needs to read its workspace. Allowing just `read` also keeps
        # OpenClaw's tool catalog out of the prompt, which matters a lot for small local models.
        "tools": {"allow": ["read"]},
        # Never talk to a gateway on OpenClaw's default port (18789): that would be *your* OpenClaw.
        # Some CLI commands fall back to a gateway when the state dir is locked by a running agent.
        "gateway": {"mode": "local", "port": config.OPENCLAW_GATEWAY_PORT, "bind": "loopback"},
    }
    if llm.split(model)[0] == "ollama":
        # Local model: use Ollama's native API (not /v1, which breaks tool calls) and give it
        # enough context for OpenClaw's system prompt + AGENTS.md + a knowledge file.
        oc_model = llm.openclaw_model(model)
        cfg["agents"]["defaults"]["models"] = {oc_model: {"params": {"num_ctx": llm.OLLAMA_NUM_CTX}}}
        cfg["models"] = {"providers": {"ollama": {"baseUrl": llm.OLLAMA_BASE, "api": "ollama",
                                                  "apiKey": "OLLAMA_API_KEY"}}}
    return cfg


def write_config(mlflow_otel: bool = False, experiment_id: str | None = None) -> dict:
    state = config.OPENCLAW_STATE_DIR
    state.mkdir(parents=True, exist_ok=True)
    # The workspace is replaced per run by a per-workspace config; see sia.agent.openclaw_env.
    cfg = base_config(config.WORKSPACES_DIR / "default")
    if mlflow_otel and experiment_id:
        # Optional: stream OpenClaw's own spans (model calls, tool calls) to MLflow's OTLP endpoint.
        # Requires: openclaw plugins install clawhub:@openclaw/diagnostics-otel
        cfg["plugins"] = {"allow": ["diagnostics-otel"], "entries": {"diagnostics-otel": {"enabled": True}}}
        cfg["diagnostics"] = {
            "enabled": True,
            "otel": {
                "enabled": True,
                "endpoint": config.TRACKING_URI,
                "protocol": "http/protobuf",
                "headers": {"x-mlflow-experiment-id": experiment_id},
                "traces": True, "metrics": False, "logs": False,
                "captureContent": True,
            },
        }
    (state / "openclaw.json").write_text(json.dumps(cfg, indent=2))
    (config.WORKSPACES_DIR / "default").mkdir(parents=True, exist_ok=True)
    return cfg


def check() -> str:
    if not shutil.which(config.OPENCLAW_BIN):
        return "openclaw not found on PATH (npm i -g openclaw). You can still use SIA_BACKEND=direct."
    out = subprocess.run([config.OPENCLAW_BIN, "config", "validate"], env=openclaw_env(),
                         capture_output=True, text=True)
    version = subprocess.run([config.OPENCLAW_BIN, "--version"], capture_output=True, text=True).stdout.strip()
    status = "valid" if out.returncode == 0 else f"INVALID: {out.stderr.strip()[-400:]}"
    return f"{version}; isolated config at {config.OPENCLAW_STATE_DIR / 'openclaw.json'} is {status}"
