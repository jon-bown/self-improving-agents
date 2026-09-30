"""Central settings. Everything can be overridden with environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

from sia import llm
from sia.scenario import REPO_ROOT

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.getenv("SIA_EXPERIMENT", "acme-support-agent")

# Which runtime answers questions: "openclaw" (the real agent), "direct" (a lightweight tool-use
# loop over the same workspace files, any provider; much faster for optimizer runs), or "offline".
BACKEND = os.getenv("SIA_BACKEND", "openclaw")

# Models are MLflow URIs, "<provider>:/<model>" (see sia/llm.py). SIA_PROVIDER picks a preset
# pair; SIA_AGENT_MODEL / MLFLOW_GENAI_JUDGE_DEFAULT_MODEL override either one.
PROVIDER = os.getenv("SIA_PROVIDER", "anthropic").lower()
if PROVIDER not in llm.PRESETS:
    raise ValueError(f"SIA_PROVIDER={PROVIDER!r}; choose one of {sorted(llm.PRESETS)} "
                     "or set SIA_AGENT_MODEL and MLFLOW_GENAI_JUDGE_DEFAULT_MODEL explicitly")
_default_agent, _default_judge = llm.PRESETS[PROVIDER]


def _model(env: str, default: str) -> str:
    value = os.getenv(env, default)
    # A bare model name ("gpt-5.5") is read as belonging to SIA_PROVIDER.
    return llm.normalize(value if ("/" in value) else f"{PROVIDER}:/{value}")


# The support agent. A small model makes the seeded failures easy to see.
AGENT_MODEL = _model("SIA_AGENT_MODEL", _default_agent)
# LLM judges, the trace analyst, the prompt optimizer's reflection step, and OpenClaw self-edits.
JUDGE_MODEL = _model("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", _default_judge)
os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = JUDGE_MODEL
# Optional embedding model; enables MemAlign judge alignment (SIMBA otherwise).
EMBEDDING_MODEL = os.getenv("SIA_EMBEDDING_MODEL") or llm.EMBEDDINGS.get(llm.split(JUDGE_MODEL)[0])

# Without credentials for the judge model only the code scorers run and the agent must use the
# offline backend. (Ollama needs no key.)
HAS_LLM = llm.has_credentials(JUDGE_MODEL) and os.getenv("SIA_NO_LLM") != "1"

# The agent's instructions (its AGENTS.md) live in the MLflow Prompt Registry under this name.
PROMPT_NAME = os.getenv("SIA_PROMPT_NAME", "acme-support-agents-md")
DATASET_NAME = os.getenv("SIA_DATASET_NAME", "acme_support_regressions")

# Isolated OpenClaw profile so the demo never touches ~/.openclaw.
OPENCLAW_BIN = os.getenv("OPENCLAW_BIN", "openclaw")
OPENCLAW_STATE_DIR = Path(os.getenv("SIA_OPENCLAW_STATE_DIR", REPO_ROOT / ".openclaw-state")).resolve()
WORKSPACES_DIR = Path(os.getenv("SIA_WORKSPACES_DIR", REPO_ROOT / ".workspaces")).resolve()
