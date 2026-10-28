"""The first-pass decision model (Jev) cascade, with stand-in judges: no keys, no network.

  * confident first-pass verdicts are kept; verdicts in the uncertain band go to the text judge
  * the grounded check fails missing or empty tool calls before any model is asked
  * server-written first-pass verdicts become the loop's `quality` verdicts in `score`
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

HARNESS = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    # One store for the module: MLflow cannot migrate a second fresh SQLite file in one process.
    tmp = tmp_path_factory.mktemp("first-pass")
    mp = pytest.MonkeyPatch()
    for k in list(os.environ):
        if k.startswith("SI_") or k.startswith("MLFLOW_TRACKING"):
            mp.delenv(k)
    mp.chdir(HARNESS)
    for k, v in {"SI_ENV_FILE": str(HARNESS / ".env.example"), "SI_STATE_DIR": str(tmp),
                 "MLFLOW_TRACKING_URI": f"sqlite:///{tmp}/mlflow.db", "SI_EXPERIMENT": "first-pass-test",
                 "SI_EXAMPLE_OFFLINE": "1", "SI_FAST_JUDGE_MODEL": "typesafe:/jev-latest",
                 "SI_FAST_JUDGE_BAND": "0.5,0.8", "TYPESAFE_API_KEY": "test-not-used",
                 "ANTHROPIC_API_KEY": "test-not-used"}.items():
        mp.setenv(k, v)
    from selfimprove import config, instructions, signals, traces

    config.reload()
    signals.reload()
    traces.setup()
    instructions.seed(open("examples/instructions.md").read())          # v1: never looks anything up
    yield signals
    mp.undo()


def last_trace():
    import mlflow

    mlflow.flush_trace_async_logging()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def fake_fast(p: float):
    def judge(inputs=None, outputs=None, **_):
        return Feedback(name="quality", value=p >= 0.5, metadata={"typesafe.probability": json.dumps(p)},
                        source=AssessmentSource(AssessmentSourceType.LLM_JUDGE, "typesafe:/jev-latest"))
    return judge


def fake_text(value: bool):
    calls = []

    def judge(inputs=None, outputs=None, **_):
        calls.append(outputs)
        return Feedback(name="quality", value=value, rationale="text judge says so")
    judge.calls = calls
    return judge


def test_confident_first_pass_is_kept(harness):
    text = fake_text(False)
    out = harness.cascade(fake_fast(0.95), text)(inputs={"question": "q"}, outputs="a")
    assert out.value is True and not text.calls
    assert out.metadata["si.escalated"] == "false"


@pytest.mark.parametrize("p", [0.5, 0.62, 0.8])
def test_uncertain_band_is_rejudged(harness, p):
    text = fake_text(False)
    out = harness.cascade(fake_fast(p), text)(inputs={"question": "q"}, outputs="a")
    assert out.value is False and len(text.calls) == 1
    assert out.metadata["si.escalated"] == "true" and "text judge says so" in out.rationale


def test_confident_fail_is_kept(harness):
    text = fake_text(True)
    out = harness.cascade(fake_fast(0.1), text)(inputs={"question": "q"}, outputs="a")
    assert out.value is False and not text.calls


def test_grounded_span_checks_run_before_any_model(harness, monkeypatch):
    from selfimprove import agents, instructions

    v2 = instructions.register_candidate(open("examples/fixes/v2_manual.md").read(), "v2", "test")
    agents.run("How long can I wait to request a refund?", ref=v2)       # v2 always calls lookup_policy
    looked = last_trace()
    agents.run("How long can I wait to request a refund?", ref=1)
    skipped = last_trace()

    asked = []
    monkeypatch.setattr(harness, "fast_judge", lambda name, instr: lambda **kw: asked.append(kw) or fake_fast(0.9)())
    monkeypatch.setattr(harness, "make_judge", lambda **kw: fake_text(True))
    check = harness.make_grounded({"name": "grounded_in_policy", "tool": "lookup_policy",
                                   "requires_input": "topic", "instructions": "Does {{ outputs }} ..."})
    assert check(trace=skipped).value is False and not asked              # caught for free
    assert check(trace=looked).value is True and len(asked) == 1          # only this one reached Jev
    assert "lookup_policy" in asked[0]["inputs"]


def test_server_first_pass_verdicts_become_quality(harness, monkeypatch):
    import mlflow

    from selfimprove import agents, traces
    from selfimprove.workflows import score

    ids = []
    for p in (0.97, 0.65):
        agents.run("How long can I wait to request a refund?", source="production")
        tid = last_trace().info.trace_id
        mlflow.log_feedback(trace_id=tid, name=harness.FIRST_PASS, value=p >= 0.5,
                            metadata={"typesafe.probability": json.dumps(p)},
                            source=AssessmentSource(AssessmentSourceType.LLM_JUDGE, "typesafe:/jev-latest"))
        ids.append(tid)
    text = fake_text(False)
    monkeypatch.setattr(harness, "quality_judge", lambda registered=True: text)
    mlflow.flush_trace_async_logging()
    kept, rejudged = score.resolve_first_pass(traces.search())
    assert (kept, rejudged) == (1, 1)
    confident, uncertain = (traces.feedback(mlflow.get_trace(t), traces.QUALITY) for t in ids)
    assert confident.value is True and confident.source.source_id == "typesafe:/jev-latest"
    assert uncertain.value is False and "re-judged" in uncertain.rationale
    assert score.resolve_first_pass(traces.search()) == (0, 0)            # idempotent
