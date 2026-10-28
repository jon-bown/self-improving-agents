"""`selfimprove self status | note`, offline: the commands the agent runs about itself.

What matters most is what a note is *not*: it never counts as a failure, never marks a trace as
scored, and only becomes a label when a person accepts it from the queue.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]

INSPECT = """
import json, sys
from selfimprove import traces, signals
traces.setup()
t = traces.search(f"trace.request_id = '{sys.argv[1]}'")[0]
print(json.dumps({"failed": traces.failed(t, signals.must_pass()), "verdicts": traces.failing_verdicts(t),
                  "unscored": traces.unscored(t),
                  "human": None if traces.human_feedback(t) is None else str(traces.human_feedback(t).value),
                  "self_reported": traces.feedback(t, traces.SELF_REPORTED) is not None,
                  "expected": traces.expectation(t)}))
"""

TRACES = """
import json, sys
from selfimprove import traces
traces.setup()
print(json.dumps([{"id": t.info.trace_id, "source": t.info.tags.get("si_source")}
                  for t in traces.search(None)]))
"""


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    state = tmp_path_factory.mktemp("selfimprove-agentside")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SI_", "MLFLOW_"))}
    env.update({
        "SI_ENV_FILE": str(HARNESS / ".env.example"),
        "SI_STATE_DIR": str(state),
        "MLFLOW_TRACKING_URI": f"sqlite:///{state}/mlflow.db",
        "SI_EXPERIMENT": "agentside-test",
        "SI_NO_LLM": "1",
        "SI_EXAMPLE_OFFLINE": "1",
        "SI_REVIEWER": "owner",
        "ANTHROPIC_API_KEY": "",
    })

    def run(*args: str, **overrides: str) -> str:
        proc = subprocess.run([sys.executable, "-m", "selfimprove.cli", *args], cwd=HARNESS,
                              env={**env, **overrides}, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"selfimprove {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr[-3000:]}"
        return proc.stdout

    def self_(*args: str, **overrides: str) -> dict:
        return json.loads(run("self", *args, **overrides).strip().splitlines()[-1])

    def python(code: str, *args: str) -> object:
        proc = subprocess.run([sys.executable, "-c", code, *args], cwd=HARNESS, env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
        return json.loads(proc.stdout.strip().splitlines()[-1])

    run.self_, run.python = self_, python
    return run


NOTE = ("note", "--user-said", "No, refunds are 60 days now.",
        "--interpretation", "I quoted the old 30-day refund window.")


def test_status_before_init(cli):
    out = cli.self_("status")
    assert out["ok"] is False and "init" in out["refused"]


def test_status(cli):
    cli("init")
    cli("ask", "How long can I wait to request a refund?")
    out = cli.self_("status")
    assert out["ok"] is True
    assert out["prompt"]["production"] == 1 and out["prompt"]["replaced"] is None
    assert out["last_gate"] is None and out["open_candidates"] == []
    assert out["pending_self_reports"] == 0
    assert set(out["live"]) == {"failure_rate", "judged_traces"}


def test_note_is_not_a_verdict_until_a_person_accepts(cli):
    newest = next(t for t in cli.python(TRACES) if t["source"] == "production")
    out = cli.self_(*NOTE, "--expected", "Refunds can be requested within 60 days.")
    assert out["ok"] is True and out["trace_id"] == newest["id"], out
    trace_id, item = out["trace_id"], out["queue_item"]

    seen = cli.python(INSPECT, trace_id)
    assert seen["self_reported"] is True
    assert seen["failed"] is False and seen["verdicts"] == [] and seen["unscored"] is True
    assert seen["human"] is None and seen["expected"] is None     # --expected waits in the queue

    assert "self_report" in cli("queue", "list")
    assert cli.self_("status")["pending_self_reports"] == 1

    again = cli.self_(*NOTE)
    assert again["ok"] is False and "already reported" in again["refused"]

    cli("queue", "accept", item)
    seen = cli.python(INSPECT, trace_id)
    assert seen["human"] is not None and seen["failed"] is True
    assert seen["expected"] == "Refunds can be requested within 60 days."
    assert cli.self_("status")["pending_self_reports"] == 0

    labeled = cli.self_(*NOTE[:1], "--trace", trace_id, *NOTE[1:])
    assert labeled["ok"] is False and "already labeled" in labeled["refused"]


def test_rejected_report_leaves_no_label(cli):
    cli("ask", "Do you ship to Canada?")
    out = cli.self_(*NOTE)
    assert out["ok"] is True
    cli("queue", "reject", out["queue_item"], "--note", "the agent was right")
    seen = cli.python(INSPECT, out["trace_id"])
    assert seen["human"] is None and seen["failed"] is False


def test_only_production_traces(cli):
    cli("ask", "--source", "eval", "What is your phone number?")
    evals = [t for t in cli.python(TRACES) if t["source"] == "eval"]
    out = cli.self_(*NOTE[:1], "--trace", evals[0]["id"], *NOTE[1:])
    assert out["ok"] is False and "no production answer" in out["refused"]


def test_nothing_recent(cli):
    out = cli.self_(*NOTE, SI_SELF_NOTE_WINDOW_MIN="0")
    assert out["ok"] is False and "no production answer" in out["refused"]


def test_needs_the_users_words(cli):
    out = cli.self_("note", "--user-said", " ", "--interpretation", "I was wrong.")
    assert out["ok"] is False and "--user-said" in out["refused"]


def test_session_limit(cli):
    cli("ask", "Can I change my billing date?")
    cli("ask", "How do I reset my password?")
    ids = [t["id"] for t in cli.python(TRACES) if t["source"] == "production"][:2]
    first = cli.self_(*NOTE[:1], "--trace", ids[0], "--session", "s1", *NOTE[1:],
                      SI_SELF_NOTE_MAX_PER_SESSION="1")
    assert first["ok"] is True, first
    second = cli.self_(*NOTE[:1], "--trace", ids[1], "--session", "s1", *NOTE[1:],
                       SI_SELF_NOTE_MAX_PER_SESSION="1")
    assert second["ok"] is False and "note limit" in second["refused"]
