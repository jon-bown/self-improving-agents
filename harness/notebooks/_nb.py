"""Notebook helpers: one isolated store for the whole series, and a few ways to look at traces.

Every notebook starts with `nb.setup()`. It points the harness at .selfimprove/notebooks/ (its
own SQLite store, artifacts and workspaces), so the walkthrough never touches the store the CLI
uses or any MLflow server you run. Notebook 00 calls `nb.setup(fresh=True)` to start over.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
STORE = HARNESS / ".selfimprove" / "notebooks"
MIN_MLFLOW = "3.17.2"  # keep in step with harness/pyproject.toml


def setup(fresh: bool = False, offline: bool | None = None):
    """Configure the harness for the notebooks and return the experiment id.

    offline=True runs the example agent's rule-based stand-in with code checks only (no API key
    needed). The default is live if ANTHROPIC_API_KEY is in .env, offline otherwise (or when
    SI_NOTEBOOK_OFFLINE=1).
    """
    os.chdir(HARNESS)
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    _check_mlflow()
    _drop_cached_stores()
    if fresh:
        _reset_store()
    STORE.mkdir(parents=True, exist_ok=True)
    env_file = HARNESS / ".env" if (HARNESS / ".env").is_file() else HARNESS / ".env.example"
    os.environ.update({
        "SI_ENV_FILE": str(env_file),
        "MLFLOW_TRACKING_URI": f"sqlite:///{STORE}/mlflow.db",
        "SI_STATE_DIR": str(STORE),
        "SI_EXPERIMENT": "walkthrough",
        "SI_AGENT": "python",
        "SI_AGENT_TARGET": "examples.support_agent:answer",
        "SI_PROMPT_NAME": "walkthrough-instructions",
        "SI_INSTRUCTIONS_FILE": "examples/instructions.md",
        "SI_WORKSPACE_DIR": "examples/knowledge",
        "SI_CRITERIA_FILE": "examples/criteria.yaml",
        "SI_GOLDEN_FILE": "examples/golden.yaml",
        # Other notebooks' kernels share this store; never serve a prompt from before a reset.
        "MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS": "0",
        "MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS": "0",
    })
    os.environ.pop("ANTHROPIC_BASE_URL", None)  # talk to the Anthropic API directly
    from dotenv import load_dotenv

    load_dotenv(env_file, override=False)
    if offline is None:
        offline = os.getenv("SI_NOTEBOOK_OFFLINE") == "1" or not os.getenv("ANTHROPIC_API_KEY")
    for k in ("SI_NO_LLM", "SI_EXAMPLE_OFFLINE"):
        if offline:
            os.environ[k] = "1"
        else:
            os.environ.pop(k, None)

    from selfimprove import config, signals, traces

    config.reload()
    signals.reload()
    exp = traces.setup()
    print(config.settings.describe())
    print(f"mode          {'offline (rule-based agent, code checks only)' if offline else 'live'}")
    return exp


def _drop_cached_stores() -> None:
    """Start this kernel on fresh MLflow connections, with no cached stores or prompts.

    MLflow caches one engine per URI on the store classes (and the tracking store itself). If
    mlflow.db was ever deleted under those connections, they stay on the unlinked file and every
    write fails with "attempt to write a readonly database". Re-running any notebook's setup cell
    clears that.
    """
    if "mlflow" not in sys.modules:
        return
    import mlflow
    from mlflow.prompt.registry_utils import PromptCache
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._model_registry import utils as registry_utils
    from mlflow.tracking._tracking_service import utils as tracking_utils

    try:
        mlflow.flush_trace_async_logging()  # don't let a previous run's traces land after a reset
    except Exception:
        pass
    for cls in (TrackingStore, RegistryStore):
        with cls._engine_map_lock:
            for engine in cls._engine_map.values():
                engine.dispose()
            cls._engine_map.clear()
    tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()
    registry_utils._get_store_registry()._get_store_with_resolved_uri.cache_clear()
    PromptCache._reset_instance()


def _check_mlflow() -> None:
    """Refuse to run on an MLflow older than the harness's.

    Every notebook shares one mlflow.db. A newer MLflow migrates its schema forward when it
    connects, and an older one can't open it after that ("Detected out-of-date database schema"),
    or builds an old schema that the newer kernels then migrate. Either way the cause is usually a
    notebook on the wrong kernel, so name the interpreter.
    """
    from importlib.metadata import version

    from packaging.version import Version

    found = version("mlflow")
    if Version(found) < Version(MIN_MLFLOW):
        raise RuntimeError(
            f"This kernel runs MLflow {found} ({sys.executable}); the notebooks need >= {MIN_MLFLOW}. "
            f"Switch to the 'selfimprove (.venv)' kernel from `make install`, or upgrade this "
            f"environment: uv pip install -p {sys.prefix} 'mlflow[genai]>={MIN_MLFLOW}'"
        )


def _reset_store() -> None:
    """Empty the notebook store in place, keeping the same mlflow.db file.

    The file is never deleted: a process that has it open (another notebook's kernel, the MLflow
    UI) would stay on the unlinked copy, where every write fails with "attempt to write a readonly
    database". Instead every table is dropped, and this kernel's MLflow recreates the schema (at its
    own version) the next time it connects. Everything else under STORE (artifacts, workspaces,
    the human queue) is removed.
    """
    import sqlite3

    db = STORE / "mlflow.db"
    if STORE.is_dir():
        for p in STORE.iterdir():
            if not p.name.startswith("mlflow.db"):  # keep mlflow.db and its -journal/-wal/-shm
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not db.exists():
        return

    con = sqlite3.connect(db, timeout=30, isolation_level=None)
    try:
        con.execute("PRAGMA foreign_keys = OFF")
        con.execute("BEGIN IMMEDIATE")
        for kind, name in con.execute(
                "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY type DESC").fetchall():
            con.execute(f'DROP {kind.upper()} IF EXISTS "{name}"')
        con.execute("COMMIT")
    finally:
        con.close()


def tree(trace, width: int = 90) -> None:
    """Print a trace as an indented span tree: name, type, and a peek at inputs/outputs."""
    spans = trace.data.spans
    children: dict[str | None, list] = {}
    for s in spans:
        children.setdefault(s.parent_id, []).append(s)

    def peek(v) -> str:
        text = str(v).replace("\n", " ")
        return text if len(text) <= width else text[: width - 3] + "..."

    def walk(span, depth):
        ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        print(f"{'  ' * depth}{span.name}  [{span.span_type}]  {ms:,.0f} ms")
        if span.inputs is not None:
            print(f"{'  ' * depth}  in : {peek(span.inputs)}")
        if span.outputs is not None:
            print(f"{'  ' * depth}  out: {peek(span.outputs)}")
        for c in children.get(span.span_id, []):
            walk(c, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)


def verdicts(trs, names=None):
    """One row per trace, one column per assessment (latest value), as a DataFrame."""
    import pandas as pd

    from selfimprove import traces

    rows = []
    for t in trs:
        row = {"trace": t.info.trace_id[:10], "v": t.info.tags.get("si_prompt_version"),
               "question": traces.question_of(t)[:45]}
        for a in t.info.assessments:
            if a.feedback is not None and (names is None or a.name in names):
                row[a.name] = "PASS" if not traces.is_fail(a.value) else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")
