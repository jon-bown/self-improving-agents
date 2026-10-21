"""Turn Hermes plugin hooks into one MLflow trace per agent turn, sent over OTLP/HTTP (JSON).

    hermes_agent (AGENT)          pre_llm_call ... on_session_end     conversation in/out, session
    ├─ chat <model> (CHAT_MODEL)  pre_api_request / post_api_request  messages, reply, tool calls, tokens
    ├─ <tool> (TOOL)              pre_tool_call / post_tool_call      arguments, result, error
    └─ hermes_agent (AGENT)       a delegated subagent's turn, nested in the parent's trace

Hooks are synchronous and can run on tool worker threads, so state lives in lock-protected dicts
keyed by Hermes' ids (turn_id, api_request_id, tool_call_id), and parents are set explicitly.
Standard library only: no OpenTelemetry SDK or exporter package is needed.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hermes_plugins.mlflow_otel")

VERSION = "0.1.0"
OK, ERROR = 1, 2


def _hex(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


def _now_ns() -> int:
    return time.time_ns()


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value))


@dataclass
class Span:
    trace_id: str
    name: str
    span_type: str
    parent_id: str | None = None
    span_id: str = field(default_factory=lambda: _hex(8))
    start_ns: int = field(default_factory=_now_ns)
    end_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: tuple[int, str] | None = None

    def end(self, status: int = OK, message: str = "") -> None:
        if self.end_ns is None:
            self.end_ns = _now_ns()
            self.status = (status, message)

    def to_otlp(self) -> dict:
        attrs = {"mlflow.spanType": self.span_type, **self.attributes}
        out = {
            "traceId": self.trace_id, "spanId": self.span_id, "name": self.name, "kind": 1,
            "startTimeUnixNano": str(self.start_ns), "endTimeUnixNano": str(self.end_ns or _now_ns()),
            "attributes": [{"key": k, "value": _otlp_value(v)} for k, v in attrs.items() if v is not None],
            "status": {"code": self.status[0], "message": self.status[1]} if self.status else {"code": 0},
        }
        if self.parent_id:
            out["parentSpanId"] = self.parent_id
        return out


def _otlp_value(v: Any) -> dict:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


@dataclass
class Config:
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_id: str | None = None
    experiment_name: str = "hermes-agent"
    capture_content: bool = True
    max_content_chars: int = 20_000
    service_name: str = "hermes-agent"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 5.0

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> Config:
        o = overrides or {}

        def flag(name: str, default: bool) -> bool:
            v = os.getenv(name)
            return default if v is None else v.strip().lower() not in ("0", "false", "no", "off")

        headers = dict(o.get("headers") or {})
        if os.getenv("MLFLOW_TRACKING_TOKEN"):
            headers.setdefault("Authorization", f"Bearer {os.environ['MLFLOW_TRACKING_TOKEN']}")
        return cls(
            tracking_uri=str(o.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI") or cls.tracking_uri).rstrip("/"),
            experiment_id=str(o.get("experiment_id") or os.getenv("MLFLOW_EXPERIMENT_ID") or "") or None,
            experiment_name=str(o.get("experiment_name") or os.getenv("MLFLOW_EXPERIMENT_NAME") or cls.experiment_name),
            capture_content=bool(o.get("capture_content", flag("MLFLOW_OTEL_CAPTURE_CONTENT", True))),
            max_content_chars=int(o.get("max_content_chars") or cls.max_content_chars),
            service_name=str(o.get("service_name") or cls.service_name),
            headers=headers,
        )


class Exporter:
    """POSTs finished traces to `{tracking_uri}/v1/traces`. Resolves (or creates) the experiment once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._experiment_id = cfg.experiment_id
        self._warned = False

    def _request(self, url: str, body: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json", **self.cfg.headers, **(headers or {})})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as res:
            return res.read()

    def experiment_id(self) -> str:
        if not self._experiment_id:
            base = f"{self.cfg.tracking_uri}/api/2.0/mlflow/experiments"
            name = self.cfg.experiment_name
            try:
                got = self._request(f"{base}/get-by-name?experiment_name={urllib.parse.quote(name)}")
                self._experiment_id = json.loads(got)["experiment"]["experiment_id"]
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                created = self._request(f"{base}/create", _json({"name": name}).encode())
                self._experiment_id = json.loads(created)["experiment_id"]
        return self._experiment_id

    def export(self, spans: list[Span]) -> bool:
        if not spans:
            return True
        body = _json({"resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": self.cfg.service_name}},
                {"key": "service.version", "value": {"stringValue": VERSION}},
            ]},
            "scopeSpans": [{"scope": {"name": "mlflow-otel", "version": VERSION},
                            "spans": [s.to_otlp() for s in spans]}],
        }]}).encode()
        try:
            self._request(f"{self.cfg.tracking_uri}/v1/traces", body, {"x-mlflow-experiment-id": self.experiment_id()})
            self._warned = False
            return True
        except Exception as e:  # never break the agent because MLflow is down
            if not self._warned:
                logger.warning("mlflow-otel: could not export to %s: %s", self.cfg.tracking_uri, e)
                self._warned = True
            return False


# ------------------------------------------------------------------------------------------ helpers
def _text(content: Any) -> str:
    """Text of an OpenAI string/parts content or Anthropic content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text") and p.get("text"))
    return "" if content is None else str(content)


def chat_messages(messages: Any) -> list[dict]:
    """Hermes / provider messages -> OpenAI chat messages (what MLflow's chat view and previews read)."""
    out: list[dict] = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            continue
        role, content = m["role"], m.get("content")
        blocks = content if isinstance(content, list) else []
        # Anthropic: tool results arrive as user messages made of tool_result blocks.
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        if results:
            out.extend({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": _text(b.get("content"))} for b in results)
            continue
        item: dict[str, Any] = {"role": role, "content": _text(content)}
        calls = m.get("tool_calls") or [
            {"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": _json(b.get("input") or {})}}
            for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if calls:
            item["tool_calls"] = calls
        if role == "tool" and m.get("tool_call_id"):
            item["tool_call_id"] = m["tool_call_id"]
        out.append(item)
    return out


def _tool_calls(assistant_message: Any) -> list[dict]:
    calls = []
    for c in getattr(assistant_message, "tool_calls", None) or []:
        fn = getattr(c, "function", None) or (c.get("function") if isinstance(c, dict) else None)
        name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
        args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
        calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else _json(args)}})
    return calls


@dataclass
class Turn:
    root: Span
    spans: list[Span]                 # shared with the parent turn for a subagent
    owner: Turn | None = None        # parent turn whose trace this one is nested in
    reply: str | None = None


# ------------------------------------------------------------------------------------------- tracer
class HermesTracer:
    def __init__(self, cfg: Config, exporter: Exporter | None = None):
        self.cfg = cfg
        self.exporter = exporter or Exporter(cfg)
        self._lock = threading.RLock()
        self._turns: dict[str, Turn] = {}
        self._turn_by_session: dict[str, str] = {}      # session_id -> its open turn_id
        self._llm: dict[str, tuple[Turn, Span]] = {}
        self._tools: dict[str, tuple[Turn, Span]] = {}
        self._child_parent: dict[str, str] = {}         # child session_id -> parent turn_id

    def _content(self, value: Any) -> str | None:
        if not self.cfg.capture_content:
            return None
        s = value if isinstance(value, str) and value.startswith(("{", "[", '"')) and _is_json(value) else _json(value)
        n = self.cfg.max_content_chars
        return s if len(s) <= n else f"{s[:n]}… [truncated {len(s) - n} chars]"

    # ----- turns
    def _turn(self, turn_id: str, session_id: str = "", create: bool = True, **attrs) -> Turn | None:
        with self._lock:
            if not turn_id:
                turn_id = self._turn_by_session.get(session_id, "")
            turn = self._turns.get(turn_id)
            if turn or not create or not turn_id:
                return turn
            parent = self._turns.get(self._child_parent.get(session_id, ""))
            if parent:   # a delegated subagent: nest under the parent's open delegate tool (or its root)
                owner = parent.owner or parent
                tool = next((s for t, s in reversed(list(self._tools.values())) if t is parent), None)
                root = Span(trace_id=owner.root.trace_id, parent_id=(tool or parent.root).span_id,
                            name="hermes_subagent", span_type="AGENT")
                turn = Turn(root=root, spans=owner.spans, owner=owner)
            else:
                root = Span(trace_id=_hex(16), name="hermes_agent", span_type="AGENT")
                root.attributes.update({
                    "mlflow.traceTag.mlflow.traceName": "hermes_agent",
                    "user.id": attrs.get("sender_id") or getpass.getuser(),
                    **{f"mlflow.traceTag.{k}": v for k, v in _env_trace_tags().items()},
                })
                turn = Turn(root=root, spans=[])
            root.attributes.update({
                "mlflow.message.format": "openai",
                "gen_ai.operation.name": "invoke_agent",
                "session.id": session_id or None,
                "hermes.turn_id": turn_id,
                "hermes.task_id": attrs.get("task_id") or None,
                "gen_ai.request.model": attrs.get("model") or None,
            })
            if attrs.get("platform") and not parent:
                root.attributes["mlflow.traceTag.hermes.platform"] = attrs["platform"]
            turn.spans.append(root)
            self._turns[turn_id] = turn
            if session_id:
                self._turn_by_session[session_id] = turn_id
            return turn

    def pre_llm_call(self, turn_id="", session_id="", task_id="", user_message=None, conversation_history=None,
                     model="", platform="", sender_id="", **_):
        turn = self._turn(turn_id, session_id, task_id=task_id, model=model, platform=platform, sender_id=sender_id)
        if not turn:
            return
        messages = chat_messages(conversation_history)
        text = _text(user_message)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != text:
            messages.append({"role": "user", "content": text})
        turn.root.attributes["mlflow.spanInputs"] = self._content({"messages": messages})

    def post_llm_call(self, turn_id="", session_id="", assistant_response=None, **_):
        turn = self._turn(turn_id, session_id, create=False)
        if turn and assistant_response is not None:
            turn.reply = _text(assistant_response)

    def on_session_end(self, turn_id="", session_id="", completed=None, interrupted=None, **_):
        with self._lock:
            turn = self._turn(turn_id, session_id, create=False)
            if not turn:
                return
            tid = turn.root.attributes.get("hermes.turn_id")
            self._turns.pop(tid, None)
            if self._turn_by_session.get(session_id) == tid:
                self._turn_by_session.pop(session_id, None)
            # close anything this turn left open
            for registry in (self._llm, self._tools):
                for key, (t, span) in list(registry.items()):
                    if t is turn:
                        span.attributes["hermes.unfinished"] = True
                        span.end(ERROR, "turn ended before this span finished")
                        registry.pop(key)
        if turn.reply is not None:
            turn.root.attributes["mlflow.spanOutputs"] = self._content(
                {"choices": [{"message": {"role": "assistant", "content": turn.reply}}]})
        turn.root.attributes["hermes.completed"] = bool(completed)
        turn.root.attributes["hermes.interrupted"] = bool(interrupted)
        if interrupted:
            turn.root.end(ERROR, "interrupted")
        elif completed is False:
            turn.root.end(ERROR, "turn did not complete")
        else:
            turn.root.end(OK)
        if turn.owner is None:   # subagent spans go out with the parent's trace
            self.exporter.export(turn.spans)

    # ----- model calls
    def pre_api_request(self, turn_id="", session_id="", api_request_id="", model="", provider="", api_mode="",
                        request_messages=None, api_call_count=0, base_url="", **_):
        turn = self._turn(turn_id, session_id, model=model)
        if not turn or not api_request_id:
            return
        span = Span(trace_id=turn.root.trace_id, parent_id=turn.root.span_id, name=f"chat {model or 'model'}",
                    span_type="CHAT_MODEL", attributes={
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": model or None,
                        "gen_ai.provider.name": provider or None,
                        "hermes.api_mode": api_mode or None,
                        "hermes.api_call_count": api_call_count,
                        "mlflow.spanInputs": self._content({"messages": chat_messages(request_messages)}),
                    })
        with self._lock:
            turn.spans.append(span)
            self._llm[api_request_id] = (turn, span)

    def post_api_request(self, api_request_id="", usage=None, assistant_message=None, finish_reason="",
                         response_model=None, api_duration=None, **_):
        with self._lock:
            entry = self._llm.pop(api_request_id, None)
        if not entry:
            return
        _, span = entry
        message: dict[str, Any] = {"role": "assistant", "content": _text(getattr(assistant_message, "content", None))}
        calls = _tool_calls(assistant_message)
        if calls:
            message["tool_calls"] = calls
        span.attributes["mlflow.spanOutputs"] = self._content({"choices": [{"message": message, "finish_reason": finish_reason}]})
        if response_model:
            span.attributes["gen_ai.response.model"] = str(response_model)
        u = usage if isinstance(usage, dict) else {}
        if u:
            inp, out = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
            span.attributes["mlflow.chat.tokenUsage"] = _json({
                "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                "cache_read_input_tokens": int(u.get("cache_read_tokens") or 0),
                "cache_creation_input_tokens": int(u.get("cache_write_tokens") or 0),
            })
        span.end(OK)

    def api_request_error(self, api_request_id="", error=None, status_code=None, retryable=None, **_):
        with self._lock:
            entry = self._llm.pop(api_request_id, None)
        if not entry:
            return
        _, span = entry
        msg = (error or {}).get("message") if isinstance(error, dict) else str(error or "")
        span.attributes.update({"hermes.status_code": status_code, "hermes.retryable": retryable})
        span.end(ERROR, str(msg or "provider request failed")[:500])

    # ----- tools
    def pre_tool_call(self, tool_name="", args=None, tool_call_id="", turn_id="", session_id="", **_):
        turn = self._turn(turn_id, session_id)
        if not turn:
            return
        span = Span(trace_id=turn.root.trace_id, parent_id=turn.root.span_id, name=tool_name or "tool",
                    span_type="TOOL", attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tool_name or None,
                        "gen_ai.tool.call.id": tool_call_id or None,
                        "mlflow.spanInputs": self._content(args if args is not None else {}),
                    })
        with self._lock:
            turn.spans.append(span)
            self._tools[tool_call_id or f"{turn_id}:{tool_name}"] = (turn, span)

    def post_tool_call(self, tool_name="", tool_call_id="", turn_id="", result=None, status=None,
                       error_message=None, duration_ms=None, **_):
        with self._lock:
            entry = self._tools.pop(tool_call_id or f"{turn_id}:{tool_name}", None)
        if not entry:
            return
        _, span = entry
        span.attributes["mlflow.spanOutputs"] = self._content(result)
        span.attributes["hermes.tool_status"] = str(status) if status else None
        if isinstance(duration_ms, (int, float)):
            span.attributes["hermes.duration_ms"] = duration_ms
        if status in ("error", "blocked", "failed"):
            span.end(ERROR, str(error_message or status)[:500])
        else:
            span.end(OK)

    # ----- subagents
    def subagent_start(self, parent_turn_id="", child_session_id="", **_):
        if parent_turn_id and child_session_id:
            with self._lock:
                self._child_parent[child_session_id] = parent_turn_id

    def subagent_stop(self, child_session_id="", **_):
        with self._lock:
            self._child_parent.pop(child_session_id, None)

    def flush_open_turns(self, **_):
        """Session finalize / process exit: export turns that never saw on_session_end."""
        with self._lock:
            open_turns = [(t, t.root.attributes.get("session.id") or "") for t in self._turns.values() if t.owner is None]
        for turn, session in open_turns:
            self.on_session_end(turn_id=turn.root.attributes["hermes.turn_id"], session_id=session, completed=False)


def _env_trace_tags() -> dict[str, str]:
    """Extra trace tags from MLFLOW_OTEL_TRACE_TAGS (a JSON object), e.g. to link a harness's own trace."""
    try:
        tags = json.loads(os.getenv("MLFLOW_OTEL_TRACE_TAGS") or "{}")
        return {str(k): str(v) for k, v in tags.items()} if isinstance(tags, dict) else {}
    except ValueError:
        return {}


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False
