"""Level 0 · Setup: register the agent's instructions and create an isolated OpenClaw profile.

Before anything can improve, two things must be true:
  1. the thing we will change (AGENTS.md) is versioned, so every change has a number and a reason;
  2. every answer the agent gives is traced, so there is something to learn from.

This script handles (1) and prepares OpenClaw. 02_first_traces.py handles (2).

    python levels/level0_instrument/01_setup.py [--mlflow-otel]
"""

import argparse

import mlflow

from sia import config, llm, openclaw_setup, tracking
from sia.workspace import seed_agents_md

parser = argparse.ArgumentParser()
parser.add_argument("--mlflow-otel", action="store_true",
                    help="also stream OpenClaw's internal spans to MLflow via OTLP (needs the diagnostics-otel plugin)")
args = parser.parse_args()

experiment_id = tracking.setup()
print(f"MLflow: {config.TRACKING_URI}  experiment '{config.EXPERIMENT}' (id {experiment_id})")

# ---- 1. AGENTS.md -> Prompt Registry, with a production alias ------------------------------
try:
    current = mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}@production")
    print(f"Prompt '{config.PROMPT_NAME}' already registered; production -> v{current.version}")
except Exception:
    v1 = mlflow.genai.register_prompt(
        name=config.PROMPT_NAME,
        template=seed_agents_md(),
        commit_message="v1: initial AGENTS.md for the Acme Cloud support agent",
        tags={"author": "human", "level": "0"},
    )
    mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=v1.version)
    print(f"Registered '{config.PROMPT_NAME}' v{v1.version} and pointed 'production' at it")

# ---- 2. Isolated OpenClaw profile ----------------------------------------------------------
openclaw_setup.write_config(mlflow_otel=args.mlflow_otel, experiment_id=experiment_id)
print("OpenClaw:", openclaw_setup.check())

# ---- 3. Hermes Agent (only when it's the runtime) -------------------------------------------
if config.BACKEND == "hermes":
    try:
        import run_agent  # noqa: F401  (the hermes-agent package)

        status = "installed"
        if llm.split(config.AGENT_MODEL)[0] == "ollama":
            variant = llm.ensure_ollama_ctx_variant(llm.split(config.AGENT_MODEL)[1], llm.HERMES_NUM_CTX)
            status += f"; Ollama model {variant} ({llm.HERMES_NUM_CTX // 1024}K context, required by Hermes)"
    except ImportError:
        status = 'not installed: pip install -e ".[hermes]"'
    except RuntimeError as e:
        status = f"model not ready: {e}"
    print(f"Hermes: {status}; isolated state in {config.HERMES_STATE_DIR}")
print(f"Provider: {config.PROVIDER}   backend: {config.BACKEND}")
print(f"Agent model: {config.AGENT_MODEL}   judge model: {config.JUDGE_MODEL}   "
      f"embeddings: {config.EMBEDDING_MODEL or 'none (SIMBA alignment)'}")
if "ollama" in (llm.split(config.AGENT_MODEL)[0], llm.split(config.JUDGE_MODEL)[0]):
    models = llm.ollama_models()
    status = "not reachable" if models is None else f"up, models: {', '.join(sorted(models)) or 'none'}"
    print(f"Ollama at {llm.OLLAMA_BASE}: {status}")
if not config.HAS_LLM:
    print(f"\nNo credentials for {config.JUDGE_MODEL} ({llm.missing_key_hint(config.JUDGE_MODEL)}): "
          "LLM judges are disabled. Use SIA_BACKEND=offline to rehearse.")
elif config.BACKEND != "offline" and not llm.has_credentials(config.AGENT_MODEL):
    print(f"\nWarning: no credentials for the agent model {config.AGENT_MODEL} "
          f"({llm.missing_key_hint(config.AGENT_MODEL)}).")
