"""openclaw-sdk + MLflow: the v4 handshake, live tool spans, and LLM spans from chat.history.

Uses recorded shapes from OpenClaw 2026.9.1's gateway; no gateway or model is needed.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import mlflow
from mlflow.entities import SpanType

from sia import openclaw_client as oc
from sia import signals

RUN = "run-1"


def _event(stream, data, ts):
    return SimpleNamespace(data={"event": "agent", "payload": {"runId": RUN, "stream": stream, "data": data, "ts": ts}})


TOOL_EVENTS = [
    _event("thinking", {"text": "check billing"}, 1000),
    _event("tool", {"phase": "start", "name": "read", "toolCallId": "t1", "args": {"path": "knowledge/billing.md"}}, 1100),
    _event("tool", {"phase": "result", "name": "read", "toolCallId": "t1", "isError": False,
                    "result": {"content": [{"type": "text", "text": "Refunds within 14 days."}]}}, 1150),
]

HISTORY = [
    {"role": "user", "content": "Refund window?", "timestamp": 990},
    {"role": "assistant", "model": "claude-haiku-5-5", "provider": "anthropic", "timestamp": 1000,
     "content": [{"type": "toolCall", "id": "t1", "name": "read", "arguments": {"path": "knowledge/billing.md"}}],
     "usage": {"input": 4, "output": 50, "cacheRead": 100, "cacheWrite": 10, "totalTokens": 164,
               "cost": {"input": 0.001, "output": 0.002, "cacheRead": 0.0001, "cacheWrite": 0.0002, "total": 0.0033}},
     "__openclaw": {"runId": RUN, "recordTimestampMs": 1090}},
    {"role": "toolResult", "toolCallId": "t1", "content": [{"type": "text", "text": "Refunds within 14 days."}]},
    {"role": "assistant", "model": "claude-haiku-5-5", "provider": "anthropic", "timestamp": 1160,
     "content": [{"type": "text", "text": "Within 14 days."}], "usage": {"input": 5, "output": 8, "totalTokens": 13},
     "__openclaw": {"runId": RUN, "recordTimestampMs": 1300}},
    {"role": "assistant", "content": "an earlier run", "__openclaw": {"runId": "other"}},
]


def test_tool_events_and_history_become_one_trace():
    with mlflow.start_span("acme_support_agent", span_type=SpanType.AGENT) as root:
        tracer = oc.MLflowTracer(root)
        for e in TOOL_EVENTS:
            asyncio.run(tracer.on_stream_event("main", e))
        tracer.close()
        turns = oc.add_llm_spans(root, HISTORY, RUN)
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    tools = [s for s in trace.data.spans if s.span_type == SpanType.TOOL]
    llms = [s for s in trace.data.spans if s.span_type == SpanType.CHAT_MODEL]
    assert [t.inputs["path"] for t in tools] == ["knowledge/billing.md"]
    assert tools[0].outputs["result"] == "Refunds within 14 days."
    assert (tools[0].end_time_ns - tools[0].start_time_ns) == 50 * 1_000_000   # gateway timestamps, not ours
    assert len(llms) == 2 and len(turns) == 2                                  # the other run is ignored
    assert llms[0].outputs["tool_calls"][0]["name"] == "read"
    usage = llms[0].attributes["mlflow.chat.tokenUsage"]
    assert usage["input_tokens"] == 114 and usage["cache_read_input_tokens"] == 100
    assert llms[0].attributes["mlflow.llm.cost"]["total_cost"] == 0.0033
    assert trace.info.token_usage["output_tokens"] == 58                        # MLflow rolls usage up
    assert signals.read_right_policy(trace=_with_topic(trace, "billing")).value is True


def _with_topic(trace, topic):
    trace.info.tags["topic"] = topic
    return trace


def test_handshake_speaks_protocol_4_with_tool_events(tmp_path):
    ident = oc.device_identity(tmp_path / "device.json")
    assert oc.device_identity(tmp_path / "device.json") == ident                 # created once
    assert oct(os.stat(tmp_path / "device.json").st_mode)[-3:] == "600"
    gw = oc._sdk_gateway_class()("ws://127.0.0.1:1", "tok", ident)
    sent = []

    async def capture(payload):
        sent.append(payload)

    gw._send_json = capture

    async def go():
        await gw._handle_challenge({"nonce": "n1", "ts": 1234})

    asyncio.run(go())
    params = sent[0]["params"]
    assert sent[0]["method"] == "connect"
    assert params["minProtocol"] == params["maxProtocol"] == 4
    assert params["caps"] == ["tool-events"] and params["auth"] == {"token": "tok"}
    assert params["device"]["signedAt"] == 1234 and params["device"]["nonce"] == "n1"
    assert params["device"]["id"] == ident["deviceId"]


def test_gateway_config_is_isolated(tmp_path):
    gw = oc.Gateway(tmp_path / "gw", tmp_path / "ws", model="anthropic:/claude-haiku-5-5", port=19999)
    cfg = gw.write_config()
    saved = json.loads(gw.config_path.read_text())
    assert saved == cfg and cfg["gateway"]["port"] == 19999 and cfg["gateway"]["auth"]["mode"] == "token"
    assert cfg["agents"]["defaults"]["workspace"] == str((tmp_path / "ws").resolve())
    assert cfg["tools"]["allow"] == ["read"]
    env = gw.env()
    assert env["OPENCLAW_STATE_DIR"] == str(gw.state_dir) and "ANTHROPIC_BASE_URL" not in env
    assert env["NO_COLOR"] == "1"                                      # so "[gateway] ready" is findable in the log
    assert gw.token and gw.token == oc.Gateway(tmp_path / "gw", tmp_path / "ws").token   # stable per home
