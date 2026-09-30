"""Create an isolated OpenClaw profile for the demo (never touches ~/.openclaw)."""

from __future__ import annotations

import json
import shutil
import subprocess

from sia import config, llm
from sia.agent import openclaw_env


def write_config(mlflow_otel: bool = False, experiment_id: str | None = None) -> dict:
    state = config.OPENCLAW_STATE_DIR
    state.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "agents": {
            "defaults": {
                # Overridden per run with OPENCLAW_WORKSPACE_DIR; see sia.agent.run_openclaw.
                "workspace": str(config.WORKSPACES_DIR / "default"),
                "model": {"primary": llm.openclaw_model(config.AGENT_MODEL)},
                # Don't create SOUL.md / BOOTSTRAP.md etc.: AGENTS.md is the only instruction file,
                # so the Prompt Registry fully determines the agent's behavior.
                "skipBootstrap": True,
            }
        },
    }
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
