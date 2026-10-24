"""Notebook helpers for the session: the precomputed store, and a few ways to look at it.

Every notebook starts with `s.setup(loop)`. It points the harness at session/session.env, with
that loop's verifier, and at the store the session server serves (`make session-server`). The
notebooks only read: every trace, label, test, version and gate in them was produced by
`python -m session.precompute`, the same steps RUNBOOK.md runs from the command line.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

SESSION = Path(__file__).resolve().parents[1]
HARNESS = SESSION.parent
RESULTS = SESSION / "store" / "results"
SERVER = "http://127.0.0.1:19560"

pd.set_option("display.max_colwidth", 120)
# Before anything imports selfimprove: its settings are read from this file at import time.
os.environ["SI_ENV_FILE"] = str(SESSION / "session.env")
warnings.filterwarnings("ignore", message="IProgress not found")


def load_keys() -> None:
    """Provider API keys from harness/.env. Only the keys: its other settings are the CLI's."""
    from dotenv import dotenv_values

    for k, v in dotenv_values(HARNESS / ".env").items():
        if k.endswith("_API_KEY") and v and not os.environ.get(k):
            os.environ[k] = v


def setup(loop: int) -> str:
    """Configure the harness for this loop and return the experiment id."""
    os.chdir(HARNESS)
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    os.environ["SI_ENV_FILE"] = str(SESSION / "session.env")
    os.environ["SI_CRITERIA_FILE"] = str(SESSION / "criteria" / f"loop{loop}.yaml")
    os.environ.pop("ANTHROPIC_BASE_URL", None)
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=3)
    except OSError:
        raise RuntimeError(f"No MLflow server at {SERVER}. In harness/, run `make session-server` "
                           "in another terminal, then re-run this cell.") from None
    load_keys()
    from selfimprove import config, signals, traces

    config.reload()
    signals.reload()
    exp = traces.setup()
    print(f"store         {SERVER} (session/store), experiment {config.settings.experiment}")
    print(f"agent         OpenClaw, {config.settings.agent_model}, traced by the MLflow connector")
    print(f"verifier      session/criteria/loop{loop}.yaml")
    print(f"MLflow UI     {SERVER}/#/experiments/{exp}")
    return exp


def results(loop: int) -> dict:
    """The summary precompute.py saved for this loop."""
    return json.loads((RESULTS / f"loop{loop}.json").read_text())


def loop_traces(loop: int, phase: str = "before"):
    """Production traces from this loop's traffic: phase "before" (the failure) or "after" (the fix)."""
    import mlflow

    return mlflow.search_traces(filter_string=f"tags.session_loop = '{loop}' AND tags.session_phase = '{phase}'",
                                return_type="list", order_by=["timestamp_ms ASC"])


def answers(trs, width: int = 160) -> pd.DataFrame:
    """Question, answer and the reviewer's verdict, one row per trace."""
    from selfimprove import traces

    rows = []
    for t in trs:
        fb = traces.human_feedback(t)
        rows.append({"v": t.info.tags.get("si_prompt_version"),
                     "question": traces.question_of(t),
                     "answer": " ".join(traces.response_of(t).split())[:width],
                     "reviewer": "" if fb is None else ("FAIL" if traces.is_fail(fb.value) else "PASS")})
    return pd.DataFrame(rows)


def verdicts(trs, names=None) -> pd.DataFrame:
    """One row per trace, one column per assessment (latest value)."""
    from selfimprove import traces

    rows = []
    for t in trs:
        row = {"v": t.info.tags.get("si_prompt_version"), "question": traces.question_of(t)[:55]}
        for a in t.info.assessments:
            if a.feedback is not None and a.valid is not False and (names is None or a.name in names):
                label = a.name if a.source.source_type != "HUMAN" else "reviewer"
                row[label] = "FAIL" if traces.is_fail(a.value) else "PASS"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def failure_rate(trs, names=None) -> pd.Series:
    """Share of traces each verifier failed (reviewer = the human labels)."""
    df = verdicts(trs, names).drop(columns=["v", "question"])
    return (df == "FAIL").sum() / (df != "").sum()


def rationale(trace, name: str) -> str:
    for a in trace.info.assessments:
        if a.name == name and a.feedback is not None:
            return f"{name}: {'FAIL' if str(a.value).lower() in ('false', 'no') else 'PASS'}. {a.rationale or ''}"
    return f"{name}: (none)"


def tree(trace, width: int = 110) -> None:
    """The harness trace, then the OpenClaw plugin's trace for the same run (found by session key)."""
    from selfimprove import traces

    def walk(t):
        spans = t.data.spans
        children: dict = {}
        for sp in spans:
            children.setdefault(sp.parent_id, []).append(sp)

        def peek(v) -> str:
            text = str(v).replace("\n", " ")
            return text if len(text) <= width else text[: width - 3] + "..."

        def rec(sp, depth):
            print(f"{'  ' * depth}{sp.name}  [{sp.span_type}]")
            if sp.span_type == "TOOL":
                print(f"{'  ' * depth}  in : {peek(sp.inputs)}")
                print(f"{'  ' * depth}  out: {peek(sp.outputs)}")
            for c in children.get(sp.span_id, []):
                rec(c, depth + 1)

        for root in children.get(None, []):
            rec(root, 0)

    print(f"harness trace {trace.info.trace_id}")
    walk(trace)
    for linked in traces.linked(trace, wait_s=0):
        print(f"\nOpenClaw plugin trace {linked.info.trace_id}  (session {linked.info.trace_metadata.get('mlflow.trace.session')})")
        walk(linked)


def gate_table(loop: int) -> pd.DataFrame:
    """The gate's scorecards: pass rate per scorer, production vs candidate, on both splits."""
    g = results(loop)["gate"]
    order = ["train/baseline", "train/candidate", "holdout/baseline", "holdout/candidate"]
    df = pd.DataFrame({k: g["cards"][k] for k in order if k in g["cards"]})
    return df.sort_index()


def before_after(loop: int) -> pd.DataFrame:
    """The loop's production traffic before and after the promotion, as the reviewers saw it."""
    r = results(loop)
    rows = {f"before (v{r['production_before']})": r["before"]}
    if "after" in r:
        rows[f"after (v{r['production_after']})"] = r["after"]
    return pd.DataFrame(rows).T[["labeled", "failed", "failure_rate"]]


def show(df: pd.DataFrame, pct: bool = False):
    """Left-aligned display; percentages for pass/failure rates."""
    styler = df.style.set_properties(**{"text-align": "left"})
    return styler.format("{:.0%}", na_rep="-") if pct else styler
