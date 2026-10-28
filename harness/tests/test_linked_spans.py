"""Tool checks read the agent's own trace when the harness trace has no tool spans.

The OpenClaw adapter's harness trace is AGENT -> CHAIN; the `read` calls are TOOL spans in the
trace the OpenClaw MLflow plugin logs for the same run. No OpenClaw, no network: traces.linked is
replaced with a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlflow.entities import SpanStatusCode

from selfimprove import signals, traces


def span(name, span_type, inputs=None, outputs=None, error=False):
    return SimpleNamespace(name=name, span_type=span_type, inputs=inputs, outputs=outputs,
                           status=SimpleNamespace(status_code=SpanStatusCode.ERROR if error else SpanStatusCode.OK))


def trace(spans):
    def search_spans(span_type=None, name=None):
        return [s for s in spans if (span_type is None or s.span_type == span_type)
                and (name is None or s.name == name)]
    return SimpleNamespace(search_spans=search_spans, data=SimpleNamespace(spans=spans, request="q", response="a"),
                           info=SimpleNamespace(trace_id="tr-harness"))


HARNESS = trace([span("acme_support", "AGENT"), span("openclaw.agent", "CHAIN")])
CHECK = {"name": "read_knowledge", "requires_tool": "read", "tool_input_match": "knowledge/"}


def run_check(monkeypatch, plugin_spans):
    monkeypatch.setattr(traces, "linked", lambda t, wait_s=20.0: [trace(plugin_spans)])
    return signals.make_check(CHECK)(trace=HARNESS)


def test_read_found_in_the_plugin_trace(monkeypatch):
    fb = run_check(monkeypatch, [span("read", "TOOL", {"path": "/ws/knowledge/billing.md"},
                                      {"result": {"content": [{"text": "14 days"}]}})])
    assert fb.value is True


def test_no_read_fails(monkeypatch):
    fb = run_check(monkeypatch, [span("llm_call", "LLM")])
    assert fb.value is False and "knowledge/" in fb.rationale


def test_reads_outside_knowledge_do_not_count(monkeypatch):
    fb = run_check(monkeypatch, [span("read", "TOOL", {"path": "/ws/AGENTS.md"}, {"result": {"content": []}})])
    assert fb.value is False


def test_failed_reads_do_not_count(monkeypatch):
    # OpenClaw reports these with an OK span status: a directory, and a guessed file name.
    fb = run_check(monkeypatch, [
        span("read", "TOOL", {"path": "knowledge/"}, {"error": "Read requires a file path"}),
        span("read", "TOOL", {"path": "/ws/knowledge/limits.md"},
             {"result": {"content": [], "details": {"kind": "not_found"}}}),
    ])
    assert fb.value is False and "failed" in fb.rationale


def test_linked_finds_the_plugin_trace_by_session_key(monkeypatch):
    harness = trace([span("openclaw.agent", "CHAIN")])
    harness.data.spans[0].attributes = {traces.SESSION_KEY: "agent:main:explicit:si-1"}
    plugin = SimpleNamespace(info=SimpleNamespace(trace_id="tr-plugin"))
    seen = []

    def search_traces(filter_string, return_type, max_results):
        seen.append(filter_string)
        return [plugin]

    monkeypatch.setattr(traces.mlflow, "search_traces", search_traces)
    assert traces.linked(harness, wait_s=0) == [plugin]
    assert seen == ["metadata.`mlflow.trace.session` = 'agent:main:explicit:si-1'"]
