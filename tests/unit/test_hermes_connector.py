"""The Hermes -> MLflow connector turns plugin hook calls into one well-formed trace per turn."""

import importlib.util
import json
import sys
from types import SimpleNamespace

from sia.scenario import REPO_ROOT

PLUGIN_DIR = REPO_ROOT / "connectors" / "hermes-mlflow"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.mlflow_otel_test", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plugin = _load_plugin()
tracer_mod = sys.modules["hermes_plugins.mlflow_otel_test.tracer"]


class FakeExporter:
    def __init__(self):
        self.batches = []

    def export(self, spans):
        self.batches.append([s.to_otlp() for s in spans])
        return True


def _tracer(**cfg):
    exporter = FakeExporter()
    return tracer_mod.HermesTracer(tracer_mod.Config(**cfg), exporter), exporter


def _attrs(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


IDS = dict(session_id="s1", task_id="t1", turn_id="s1:t1:abc")


def _one_turn(tr):
    tr.pre_llm_call(**IDS, user_message="How long do refunds take?", model="claude-haiku-5-5", platform="cli",
                    conversation_history=[{"role": "user", "content": "How long do refunds take?"}], sender_id="")
    tr.pre_api_request(**IDS, api_request_id="s1:t1:abc:api:1", model="claude-haiku-5-5", provider="anthropic",
                       api_mode="anthropic_messages", api_call_count=1,
                       request_messages=[{"role": "user", "content": [{"type": "text", "text": "How long do refunds take?"}]}])
    call = SimpleNamespace(id="tc1", function=SimpleNamespace(name="read_file", arguments='{"path": "billing.md"}'))
    tr.post_api_request(api_request_id="s1:t1:abc:api:1", usage={"input_tokens": 50, "output_tokens": 10, "cache_read_tokens": 4},
                        assistant_message=SimpleNamespace(content="", tool_calls=[call]), finish_reason="tool_calls")
    tr.pre_tool_call(**IDS, tool_name="read_file", args={"path": "billing.md"}, tool_call_id="tc1")
    tr.post_tool_call(**IDS, tool_name="read_file", tool_call_id="tc1", result='{"content": "14 days"}', status="ok", duration_ms=2)
    tr.pre_api_request(**IDS, api_request_id="s1:t1:abc:api:2", model="claude-haiku-5-5", provider="anthropic",
                       api_call_count=2, request_messages=[
                           {"role": "assistant", "content": [{"type": "tool_use", "id": "tc1", "name": "read_file", "input": {"path": "billing.md"}}]},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "14 days"}]}])
    tr.post_api_request(api_request_id="s1:t1:abc:api:2", usage={"input_tokens": 70, "output_tokens": 8},
                        assistant_message=SimpleNamespace(content="Refunds take 14 days.", tool_calls=None), finish_reason="stop")
    tr.post_llm_call(**IDS, assistant_response="Refunds take 14 days.")
    tr.on_session_end(**IDS, completed=True, interrupted=False)


def test_one_turn_is_one_trace():
    tr, ex = _tracer()
    _one_turn(tr)
    assert len(ex.batches) == 1
    root, llm1, tool, llm2 = ex.batches[0]
    assert {s["traceId"] for s in ex.batches[0]} == {root["traceId"]}
    assert "parentSpanId" not in root and all(s["parentSpanId"] == root["spanId"] for s in (llm1, tool, llm2))
    r = _attrs(root)
    assert r["mlflow.spanType"] == "AGENT" and r["session.id"] == "s1"
    assert r["mlflow.traceTag.mlflow.traceName"] == "hermes_agent"
    assert json.loads(r["mlflow.spanInputs"]) == {"messages": [{"role": "user", "content": "How long do refunds take?"}]}
    assert json.loads(r["mlflow.spanOutputs"])["choices"][0]["message"]["content"] == "Refunds take 14 days."
    assert root["status"]["code"] == 1

    l1, l2 = _attrs(llm1), _attrs(llm2)
    assert l1["mlflow.spanType"] == "CHAT_MODEL" and l1["gen_ai.provider.name"] == "anthropic"
    assert json.loads(l1["mlflow.spanOutputs"])["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(l1["mlflow.chat.tokenUsage"]) == {
        "input_tokens": 50, "output_tokens": 10, "total_tokens": 60, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 0}
    # Anthropic tool_use / tool_result blocks are shown as OpenAI tool calls / tool messages
    msgs = json.loads(l2["mlflow.spanInputs"])["messages"]
    assert msgs[0]["tool_calls"][0]["id"] == "tc1" and msgs[1] == {"role": "tool", "tool_call_id": "tc1", "content": "14 days"}

    t = _attrs(tool)
    assert t["mlflow.spanType"] == "TOOL" and json.loads(t["mlflow.spanInputs"]) == {"path": "billing.md"}
    assert json.loads(t["mlflow.spanOutputs"]) == {"content": "14 days"}


def test_capture_content_off():
    tr, ex = _tracer(capture_content=False)
    _one_turn(tr)
    for span in ex.batches[0]:
        a = _attrs(span)
        assert "mlflow.spanInputs" not in a and "mlflow.spanOutputs" not in a
    assert "mlflow.chat.tokenUsage" in _attrs(ex.batches[0][1])


def test_errors_and_interrupts():
    tr, ex = _tracer()
    tr.pre_llm_call(**IDS, user_message="hi", conversation_history=[])
    tr.pre_api_request(**IDS, api_request_id="r1", model="m")
    tr.api_request_error(api_request_id="r1", error={"type": "RateLimit", "message": "429 slow down"}, status_code=429)
    tr.pre_tool_call(**IDS, tool_name="terminal", args={"cmd": "rm -rf /"}, tool_call_id="tc9")
    tr.post_tool_call(**IDS, tool_name="terminal", tool_call_id="tc9", result="blocked", status="blocked", error_message="denied")
    tr.pre_tool_call(**IDS, tool_name="web", args={}, tool_call_id="tc10")   # never finishes
    tr.on_session_end(**IDS, completed=False, interrupted=True)
    root, llm, blocked, unfinished = ex.batches[0]
    assert root["status"] == {"code": 2, "message": "interrupted"}
    assert llm["status"] == {"code": 2, "message": "429 slow down"}
    assert blocked["status"]["code"] == 2
    assert _attrs(unfinished)["hermes.unfinished"] is True


def test_subagent_turn_nests_inside_parent_trace():
    tr, ex = _tracer()
    tr.pre_llm_call(**IDS, user_message="research X", conversation_history=[])
    tr.pre_tool_call(**IDS, tool_name="delegate_task", args={"goal": "X"}, tool_call_id="d1")
    tr.subagent_start(parent_session_id="s1", parent_turn_id=IDS["turn_id"], child_session_id="child")
    child = dict(session_id="child", task_id="ct", turn_id="child:ct:1")
    tr.pre_llm_call(**child, user_message="X", conversation_history=[])
    tr.pre_api_request(**child, api_request_id="child:ct:1:api:1", model="m")
    tr.post_api_request(api_request_id="child:ct:1:api:1", assistant_message=SimpleNamespace(content="found X", tool_calls=None))
    tr.post_llm_call(**child, assistant_response="found X")
    tr.on_session_end(**child, completed=True)
    assert ex.batches == []   # the subagent goes out with its parent
    tr.subagent_stop(child_session_id="child")
    tr.post_tool_call(**IDS, tool_name="delegate_task", tool_call_id="d1", result="found X", status="ok")
    tr.on_session_end(**IDS, completed=True)
    spans = ex.batches[0]
    by_name = {s["name"]: s for s in spans}
    assert len({s["traceId"] for s in spans}) == 1
    assert by_name["hermes_subagent"]["parentSpanId"] == by_name["delegate_task"]["spanId"]


def test_open_turns_flush_on_exit():
    tr, ex = _tracer()
    tr.pre_llm_call(**IDS, user_message="hi", conversation_history=[])
    tr.flush_open_turns()
    assert ex.batches[0][0]["status"]["code"] == 2


def test_register_wires_hooks_and_never_returns_values(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")   # nothing listening: export must not raise
    hooks = {}
    plugin.register(SimpleNamespace(register_hook=lambda name, fn: hooks.setdefault(name, fn)))
    assert {"pre_llm_call", "pre_api_request", "post_tool_call", "on_session_end", "on_session_finalize"} <= set(hooks)
    assert hooks["pre_llm_call"](**IDS, user_message="hi", conversation_history=[]) is None
    assert hooks["pre_tool_call"](**IDS, tool_name="x", args={}, tool_call_id="1") is None
    assert hooks["on_session_end"](**IDS, completed=True) is None
