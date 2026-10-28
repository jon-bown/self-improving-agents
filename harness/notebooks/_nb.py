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


def setup(fresh: bool = False, offline: bool | None = None):
    """Configure the harness for the notebooks and return the experiment id.

    offline=True runs the example agent's rule-based stand-in with code checks only (no API key
    needed). The default is live if ANTHROPIC_API_KEY is in .env, offline otherwise (or when
    SI_NOTEBOOK_OFFLINE=1).
    """
    os.chdir(HARNESS)
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    if fresh:
        _drop_cached_stores()
        shutil.rmtree(STORE, ignore_errors=True)
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
    """Close MLflow's pooled SQLite connections so a re-run of setup(fresh=True) can delete the store.

    MLflow caches one engine per URI on the store classes (and the tracking store itself). Deleting
    mlflow.db under those open connections leaves them on the unlinked file, and the next write fails
    with "attempt to write a readonly database".
    """
    if "mlflow" not in sys.modules:
        return
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._tracking_service import utils as tracking_utils

    for cls in (TrackingStore, RegistryStore):
        with cls._engine_map_lock:
            for engine in cls._engine_map.values():
                engine.dispose()
            cls._engine_map.clear()
    tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()


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
