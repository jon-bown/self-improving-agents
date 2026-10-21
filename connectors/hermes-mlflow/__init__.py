"""Hermes Agent -> MLflow tracing plugin.

Every agent turn becomes one MLflow trace with a span per model request and per tool call, exported
over OTLP to a running MLflow server. Settings come from the environment (Hermes loads
~/.hermes/.env) or from `plugins.entries.mlflow-otel` in config.yaml:

    MLFLOW_TRACKING_URI           default http://127.0.0.1:5000
    MLFLOW_EXPERIMENT_ID          or MLFLOW_EXPERIMENT_NAME (default "hermes-agent", created if missing)
    MLFLOW_OTEL_CAPTURE_CONTENT   default true; false records structure, timing and tokens only
"""

from __future__ import annotations

import atexit
import logging

from .tracer import Config, HermesTracer

logger = logging.getLogger(__name__)
PLUGIN_KEY = "mlflow-otel"

_HOOKS = (
    "pre_llm_call", "post_llm_call", "on_session_end",
    "pre_api_request", "post_api_request", "api_request_error",
    "pre_tool_call", "post_tool_call",
    "subagent_start", "subagent_stop",
)


def _config_overrides() -> dict:
    try:
        from hermes_cli.config import load_config

        entry = ((load_config() or {}).get("plugins") or {}).get("entries", {}).get(PLUGIN_KEY) or {}
        return entry.get("config", entry) if isinstance(entry, dict) else {}
    except Exception:
        return {}


def _safe(fn):
    # Hermes swallows hook exceptions too, but a tracer must also never *return* anything:
    # a value from pre_llm_call is injected into the prompt, and one from pre_tool_call can block the tool.
    def hook(**kwargs):
        try:
            fn(**kwargs)
        except Exception:
            logger.debug("mlflow-otel: %s failed", fn.__name__, exc_info=True)

    hook.__name__ = fn.__name__
    return hook


def register(ctx) -> None:
    cfg = Config.from_env(_config_overrides())
    if not cfg.tracking_uri.startswith(("http://", "https://")):
        logger.warning("mlflow-otel: MLFLOW_TRACKING_URI must be an http(s) server URL, got %r; tracing is off",
                       cfg.tracking_uri)
        return
    tracer = HermesTracer(cfg)
    for name in _HOOKS:
        ctx.register_hook(name, _safe(getattr(tracer, name)))
    ctx.register_hook("on_session_finalize", _safe(tracer.flush_open_turns))
    atexit.register(tracer.flush_open_turns)
