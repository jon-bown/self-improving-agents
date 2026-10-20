"""Small helpers for the walkthrough notebooks (notebooks/*.ipynb).

Call setup() in the first cell, *before* importing anything else from sia: sia.config reads
the environment once, at import time.

    from sia import notebook as nb
    nb.setup()                       # local store notebooks/.mlflow, offline runtime
    nb.setup(backend="openclaw")     # or "hermes" / "direct"
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = REPO_ROOT / "notebooks"
NB_STORE = NB_DIR / ".mlflow"


def setup(backend: str = "offline", provider: str | None = None, store: str = "local",
          fresh: bool = False, experiment: str = "acme-support-notebooks", db: str = "mlflow.db",
          llm: bool = True) -> dict:
    """Configure the environment for a notebook. Returns a summary dict.

    store="local"  -> sqlite at notebooks/.mlflow/mlflow.db (shared by all notebooks, so later
                      notebooks build on earlier ones). Browse it with `make notebooks-ui`.
    store="server" -> whatever MLFLOW_TRACKING_URI / .env points at (e.g. `make server`).
    fresh=True     -> wipe this notebook database first (notebook 00 offers this).
    llm=False      -> code-only scorers and the rule-based analyst, even if a model is available
                      (deterministic outputs; notebooks 03-04 use this).
    db             -> sqlite file name under notebooks/.mlflow/. The Prompt Registry is global to a
                      store, so a notebook that must start from v1 (the autonomous loop) uses its own.
    """
    if "sia.config" in sys.modules:
        print("sia.config was already imported in this kernel; restart the kernel to change settings.")
    NB_STORE.mkdir(parents=True, exist_ok=True)
    if fresh:
        # Only this database and its agents' state: notebooks/.mlflow/mlruns holds the artifacts of
        # every notebook database, so it stays (orphaned artifacts are harmless).
        (NB_STORE / db).unlink(missing_ok=True)
        for state in (f"openclaw-{Path(db).stem}", f"hermes-{Path(db).stem}"):
            shutil.rmtree(NB_STORE / state, ignore_errors=True)
    if store == "local":
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{NB_STORE / db}"
        os.environ["SIA_WORKSPACES_DIR"] = str(NB_STORE / "workspaces")
        stem = Path(db).stem   # each notebook database gets its own agent state
        os.environ["SIA_OPENCLAW_STATE_DIR"] = str(NB_STORE / f"openclaw-{stem}")
        os.environ["SIA_HERMES_STATE_DIR"] = str(NB_STORE / f"hermes-{stem}")
    os.environ["SIA_BACKEND"] = backend
    os.environ["SIA_EXPERIMENT"] = experiment
    if provider:
        os.environ["SIA_PROVIDER"] = provider
    if not llm:
        os.environ["SIA_NO_LLM"] = "1"            # code scorers only: fast and deterministic
    os.chdir(NB_STORE)  # MLflow writes run artifacts to ./mlruns; keep them out of the repo
    _quiet()

    from sia import config, tracking

    experiment_id = tracking.setup()
    if shutil.which(config.OPENCLAW_BIN):
        # Also when another backend is selected: skill proposals go through OpenClaw's Skill
        # Workshop (security scan + rollback metadata) whenever OpenClaw is installed.
        from sia import openclaw_setup

        openclaw_setup.write_config()
    summary = {"tracking_uri": config.TRACKING_URI, "experiment": f"{config.EXPERIMENT} (id {experiment_id})",
               "backend": config.BACKEND, "agent_model": config.AGENT_MODEL, "judge_model": config.JUDGE_MODEL,
               "llm_judges": config.HAS_LLM}
    for k, v in summary.items():
        print(f"{k:<13} {v}")
    return summary


def _quiet() -> None:
    """Keep notebook output about the agent, not MLflow's progress bars and info logs."""
    import logging
    import warnings

    os.environ["TQDM_DISABLE"] = "1"
    for name in ("mlflow", "LiteLLM", "litellm", "httpx", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*IProgress not found.*")


def available_backends() -> dict[str, str]:
    """Which runtimes can run right now, and why not if they can't."""
    from sia import config, llm

    out = {"offline": "ok", "direct": "ok" if llm.has_credentials(config.AGENT_MODEL) else
           f"needs a model: {llm.missing_key_hint(config.AGENT_MODEL)}"}
    model_ready = llm.has_credentials(config.AGENT_MODEL)
    out["openclaw"] = ("ok" if shutil.which(config.OPENCLAW_BIN) and model_ready else
                       "openclaw not on PATH" if not shutil.which(config.OPENCLAW_BIN) else
                       f"needs a model: {llm.missing_key_hint(config.AGENT_MODEL)}")
    try:
        import importlib.util

        hermes_installed = importlib.util.find_spec("run_agent") is not None
    except Exception:
        hermes_installed = False
    out["hermes"] = ("ok" if hermes_installed and model_ready else
                     'not installed: make hermes' if not hermes_installed else
                     f"needs a model: {llm.missing_key_hint(config.AGENT_MODEL)}")
    return out


def ensure_production(agents_md: str | None = None, version_note: str = "v1") -> int:
    """Register AGENTS.md (the flawed v1 by default) and point @production at it, once."""
    import mlflow

    from sia import config
    from sia.workspace import seed_agents_md

    try:
        return int(mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}@production").version)
    except Exception:
        pv = mlflow.genai.register_prompt(name=config.PROMPT_NAME, template=agents_md or seed_agents_md(),
                                          commit_message=version_note)
        mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=pv.version)
        return int(pv.version)


def sdk_workspace(runtime: str, agents_md: str | None = None) -> Path:
    """The live workspace one agent works in for the notebooks/sdk series (notebooks/.mlflow/sdk/<runtime>).

    Unlike sia.agent's per-version workspaces, this one stays put: a new AGENTS.md version is
    pushed into it (OpenClaw: agents.files.set through the gateway; Hermes: the file it reads).
    """
    from sia.scenario import KNOWLEDGE_DIR
    from sia.workspace import seed_agents_md

    ws = NB_STORE / "sdk" / runtime
    (ws / "knowledge").mkdir(parents=True, exist_ok=True)
    for f in KNOWLEDGE_DIR.glob("*.md"):
        shutil.copy(f, ws / "knowledge" / f.name)
    if agents_md is not None or not (ws / "AGENTS.md").exists():
        (ws / "AGENTS.md").write_text(agents_md or seed_agents_md())
    return ws


def span_tree(trace, width: int = 70) -> str:
    """A trace as an indented tree: span type, name, duration, and the most useful input/output."""
    spans = trace.data.spans
    children: dict[str | None, list] = {}
    for s in spans:
        children.setdefault(s.parent_id, []).append(s)
    lines = []

    def brief(s) -> str:
        inp = s.inputs if isinstance(s.inputs, dict) else {}
        if s.span_type == "TOOL":
            return str(inp.get("path") or inp.get("name") or inp)[:width]
        if s.span_type in ("LLM", "CHAT_MODEL"):
            out = s.outputs if isinstance(s.outputs, dict) else {}
            calls = out.get("tool_calls")
            return (f"-> tools {calls}" if calls else f"-> {str(out.get('content') or out.get('text') or '')[:width]!r}")
        if s.parent_id is None:
            return repr(str(s.outputs)[:width])
        return ""

    def walk(span, depth):
        dur = (span.end_time_ns - span.start_time_ns) / 1e9 if span.end_time_ns else 0
        lines.append(f"{'  ' * depth}{'└─ ' if depth else ''}{span.span_type:<10} {span.name:<20} "
                     f"{dur:6.1f}s  {brief(span)}")
        for c in sorted(children.get(span.span_id, []), key=lambda x: x.start_time_ns):
            walk(c, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(lines)


def show(trace) -> None:
    print(span_tree(trace))
    tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
    if tags:
        print("tags:", tags)


def source(module, name: str) -> str:
    """Source of a top-level function in a module, including its decorators. Works for MLflow
    @scorer functions too, which inspect.getsource() can't see through."""
    import inspect

    lines = inspect.getsource(module).splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"def {name}(")), None)
    if start is None:
        return f"# {name} not found in {module.__name__}\n"
    end = start + 1
    while start > 0 and lines[start - 1].startswith("@"):        # include decorators
        start -= 1
    while end < len(lines) and (not lines[end] or lines[end][0] in " \t" or lines[end].startswith(")")):
        end += 1                                                    # body ends at the next top-level line
    return "\n".join(lines[start:end]).rstrip() + "\n"


def last_trace():
    import mlflow

    mlflow.flush_trace_async_logging()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def verdicts(trace) -> dict:
    """name -> (value, source) for every feedback on a trace."""
    import mlflow

    t = mlflow.get_trace(trace.info.trace_id)
    return {a.name: (a.value, str(a.source.source_type).split(".")[-1]) for a in t.info.assessments
            if a.feedback is not None}
