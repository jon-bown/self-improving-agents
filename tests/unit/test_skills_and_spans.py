"""Skills as an update target, OpenClaw trajectory spans, and tool-level scorers (no LLM, no OpenClaw)."""

import json
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from sia import config, gate, improvers, signals, skills
from sia.agent import AgentSpec, pin, production_skills, register_skill, resolve
from sia.openclaw_trace import replay
from sia.scenario import SCENARIO_DIR

FIXTURE = Path(__file__).parent / "fixtures" / "openclaw_trajectory_events.jsonl"


def _traced(events, topic):
    with mlflow.start_span("acme_support_agent", span_type=SpanType.AGENT):
        mlflow.update_current_trace(tags={"topic": topic})
        with mlflow.start_span("openclaw.agent", span_type=SpanType.CHAIN) as span:
            n = replay(events, span)
    mlflow.flush_trace_async_logging()
    return n, mlflow.get_trace(mlflow.get_last_active_trace_id())


def test_trajectory_replay_builds_llm_and_tool_spans():
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    n, trace = _traced(events, "account_access")
    kinds = [s.span_type for s in trace.data.spans]
    assert n == 5 and kinds.count(SpanType.LLM) == 2 and kinds.count(SpanType.TOOL) == 3
    # the agent issued all three reads in one parallel tool-call turn (same timestamp)
    assert set(signals.knowledge_reads(trace)) == {
        "knowledge/billing.md", "knowledge/account_access.md", "knowledge/product_howto.md"}


def test_tool_scorers_judge_how_the_answer_was_found():
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    _, trace = _traced(events, "account_access")      # read the right file... and every other one
    assert signals.read_right_policy(trace=trace).value is True
    assert signals.focused_reading(trace=trace).value is False
    _, trace = _traced([], "billing")                  # answered without reading anything
    fb = signals.read_right_policy(trace=trace)
    assert fb.value is False and "without reading" in fb.rationale


def _setup_registry():
    mlflow.set_experiment("unit")
    p = mlflow.genai.register_prompt(name=config.PROMPT_NAME, template="Use knowledge/ for answers.")
    mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=p.version)


def test_skill_versions_pin_promote_and_roll_back():
    _setup_registry()
    assert pin("production") == AgentSpec(1, ())
    v = register_skill("handle-x", "---\nname: handle-x\n---\nbody", "new", author="test", level=3)
    candidate = AgentSpec("production", (("handle-x", v),))
    assert pin(candidate).label == "v1+handle-x@1"
    agents, loaded = resolve(candidate)
    assert set(loaded) == {"handle-x"} and production_skills() == {}   # registered, not serving

    receipt = gate.promote(candidate, reason="unit test")
    assert production_skills() == {"handle-x": 1}
    assert receipt["skills"] == {"handle-x": {"from": None, "to": 1}}
    assert "removed" in gate.rollback(receipt) and production_skills() == {}


def test_offline_analyst_proposes_a_skill_for_an_uncovered_topic():
    agents_md = (SCENARIO_DIR / "fixes" / "level2_manual_AGENTS.md").read_text()
    report = [{"trace_id": "t1", "topic": "support_process", "failures": [{"rationale": "Missing a key fact"}]}]
    p = improvers.offline_analyze(report, agents_md)
    assert p["update_target"] == "skill" and p["skill"]["name"] == "handle-support-process"
    assert "knowledge/support_process.md" in p["skill"]["body"]
    assert improvers.offline_analyze(report, agents_md, target="agents_md")["update_target"] == "agents_md"


def test_skill_proposal_without_openclaw_registers_in_mlflow_only(monkeypatch):
    _setup_registry()
    monkeypatch.setattr(skills, "workshop_available", lambda: False)
    p = skills.propose("handle-y", "When to use it", "Do the thing.", goal="g", evidence="e",
                       author="test", level=3)
    assert p.proposal_id is None and p.version == 1
    assert skills.apply(p)[0] is True
