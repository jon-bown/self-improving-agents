"""OpenClaw-native spans: rebuild what OpenClaw did inside each MLflow trace.

`openclaw agent --json` only reports a tool *summary*. To see which files the agent read, with
what arguments, and what came back, we export the session's trajectory (OpenClaw's own
record of the run, `openclaw sessions export-trajectory`) and replay it as spans under the
harness's `openclaw.agent` span, with OpenClaw's original timestamps:

    acme_support_agent (AGENT)
      └─ openclaw.agent (CHAIN)
           ├─ llm_call (LLM)        model, token usage, the assistant turn
           ├─ read (TOOL)           {"path": "knowledge/billing.md"} -> file contents
           └─ llm_call (LLM)        final answer

That turns tool use into signal: scorers can check *how* the agent got its answer
(sia.signals.read_right_policy, MLflow's ToolCallCorrectness), and issue discovery sees the
execution path of every failing trace.

Why not the @mlflow/mlflow-openclaw plugin? It produces this same span tree, but as of 0.2.0-rc.0
it ships TypeScript only and imports a plugin-SDK path that OpenClaw 2026.9.x removed, and it
flushes asynchronously after the run -- which `agent --local` (one process per answer) exits
before. The trajectory is written by OpenClaw itself, so this works with any provider and offline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from sia import config

MAX_TEXT = 4000


def _ns(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)


def _text(content) -> str:
    if isinstance(content, str):
        return content[:MAX_TEXT]
    parts = [c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]
    return "\n".join(parts)[:MAX_TEXT]


def export_events(session_id: str, env: dict[str, str]) -> list[dict]:
    """Export one session's trajectory and return its events (oldest first)."""
    tmp = Path(tempfile.mkdtemp(prefix="sia-traj-"))
    try:
        proc = subprocess.run(
            [config.OPENCLAW_BIN, "sessions", "export-trajectory", "--agent", "main",
             "--session-key", f"agent:main:explicit:{session_id}", "--workspace", str(tmp),
             "--output", session_id, "--json"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        events_file = tmp / ".openclaw" / "trajectory-exports" / session_id / "events.jsonl"
        if proc.returncode != 0 or not events_file.exists():
            return []
        return [json.loads(line, strict=False) for line in events_file.read_text().splitlines() if line.strip()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def replay(events: list[dict], parent) -> int:
    """Create LLM and TOOL child spans of `parent` from trajectory events. Returns spans created."""
    created = 0
    turn_start = None
    open_tools: dict[str, object] = {}
    for e in events:
        kind, data, ts = e.get("type"), e.get("data") or {}, e.get("ts")
        if not ts:
            continue
        if kind in ("prompt.submitted", "user.message") and turn_start is None:
            turn_start = ts
        elif kind == "assistant.message":
            msg = data.get("message", {})
            content = msg.get("content") or []
            calls = [c for c in content if isinstance(c, dict) and c.get("type") == "toolCall"]
            span = mlflow.start_span_no_context(
                "llm_call", span_type=SpanType.LLM, parent_span=parent,
                inputs={"turn": "after tool results" if created else "user prompt"},
                attributes={"model": f"{msg.get('provider')}/{msg.get('model')}",
                            "usage": json.dumps(msg.get("usage", {}).get("totalTokens"))},
                start_time_ns=_ns(turn_start or ts),
            )
            span.end(outputs={"text": _text(content),
                              "tool_calls": [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls]},
                     end_time_ns=_ns(ts))
            created += 1
            turn_start = ts
        elif kind == "tool.call":
            open_tools[data.get("toolCallId")] = mlflow.start_span_no_context(
                data.get("name", "tool"), span_type=SpanType.TOOL, parent_span=parent,
                inputs=data.get("arguments") or {}, attributes={"tool_call_id": data.get("toolCallId")},
                start_time_ns=_ns(ts),
            )
        elif kind == "tool.result":
            msg = data.get("message", {})
            span = open_tools.pop(msg.get("toolCallId"), None)
            if span is not None:
                span.end(outputs={"result": _text(msg.get("content"))},
                         status="ERROR" if msg.get("isError") else "OK", end_time_ns=_ns(ts))
                created += 1
            turn_start = ts
    for span in open_tools.values():  # tool calls that never returned
        span.end(status="ERROR")
    return created


def replay_session(session_id: str, env: dict[str, str], parent) -> int:
    try:
        n = replay(export_events(session_id, env), parent)
    except Exception as e:  # never fail an answer because the trace couldn't be enriched
        parent.set_attribute("openclaw.replay_error", str(e)[:500])
        return 0
    parent.set_attribute("openclaw.replayed_spans", n)
    return n
