"""OpenClaw from Python: the `openclaw-sdk` client, an isolated gateway, and MLflow on top.

    from sia.openclaw_client import Gateway, OpenClawAgent

    gw = Gateway.for_workspace(workspace).start()   # `openclaw gateway run` on port 19556
    agent = OpenClawAgent(gw)                       # openclaw-sdk client over the gateway WebSocket
    agent.set_agents_md(text)                       # agents.files.set: swap in a prompt version
    answer = agent.ask("Can we use Okta SSO on Pro?")   # one MLflow trace per answer

Three pieces:

Gateway        An OpenClaw gateway of its own (state dir, config, token, workspace, port), so the
               notebooks never touch ~/.openclaw or the gateway you use day to day (port 18789).

SDKGateway     openclaw-sdk 2.1.0's WebSocket client speaks gateway protocol v3 and reads its token
               and device key from ~/.openclaw. OpenClaw 2026.9 gateways speak v4. This subclass
               changes only the `connect` handshake: protocol 4, the `tool-events` capability (so
               tool calls are streamed to us), and an explicit token + device key.

MLflowTracer   An openclaw-sdk CallbackHandler that turns the gateway's live tool events into
               MLflow TOOL spans. After the run, the session's `chat.history` adds one LLM span
               per model turn, with the model name, token usage, and cost OpenClaw recorded.

    acme_support_agent (AGENT)
      └─ openclaw.run (CHAIN)            run id, session key
           ├─ claude-haiku-5-5 (CHAT_MODEL)   tokens, cost, tool calls requested
           ├─ read (TOOL)                {"path": "knowledge/billing.md"} -> contents
           └─ claude-haiku-5-5 (CHAT_MODEL)

The SDK is async; OpenClawAgent runs it on a background event loop so `ask()` is a plain function
(what mlflow.genai.evaluate's predict_fn needs) and the spans attach to the caller's trace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import CostKey, SpanAttributeKey, TokenUsageKey

from sia import config

SDK_PORT = int(os.getenv("SIA_OPENCLAW_SDK_PORT", "19556"))  # not 18789 (yours), not 19555 (CLI profile)
PROTOCOL = 4
SCOPES = ["operator.read", "operator.write", "operator.admin"]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------------------------
# The gateway process
# ---------------------------------------------------------------------------------------------

class Gateway:
    """An isolated `openclaw gateway run`. Everything it owns lives under `home`."""

    def __init__(self, home: Path, workspace: Path, model: str | None = None, port: int = SDK_PORT):
        self.home, self.workspace, self.port = Path(home).resolve(), Path(workspace).resolve(), port
        self.model = model or config.AGENT_MODEL
        self.state_dir = self.home / "state"
        self.config_path = self.state_dir / "openclaw.json"
        self.log_path = self.home / "gateway.log"
        self._pid_path = self.home / "gateway.pid"
        token_path = self.home / "token"
        if not token_path.exists():
            self.home.mkdir(parents=True, exist_ok=True)
            token_path.write_text(secrets.token_urlsafe(24))
            token_path.chmod(0o600)
        self.token = token_path.read_text().strip()

    @classmethod
    def for_workspace(cls, workspace: Path, **kw) -> Gateway:
        return cls(config.OPENCLAW_STATE_DIR.parent / "openclaw-gateway", workspace, **kw)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def identity_path(self) -> Path:
        return self.home / "sdk-device.json"

    def env(self) -> dict[str, str]:
        env = dict(os.environ, OPENCLAW_STATE_DIR=str(self.state_dir), OPENCLAW_CONFIG_PATH=str(self.config_path))
        env.pop("ANTHROPIC_BASE_URL", None)  # talk to the real API with your ANTHROPIC_API_KEY
        env.update(NO_COLOR="1", FORCE_COLOR="0")  # plain log lines (Jupyter kernels force color on)
        return env

    def write_config(self) -> dict:
        from sia import openclaw_setup

        cfg = openclaw_setup.base_config(self.workspace, self.model)
        cfg["gateway"] = {"mode": "local", "port": self.port, "bind": "loopback",
                          "auth": {"mode": "token", "token": self.token}}
        # Don't advertise this throwaway gateway on the local network.
        cfg["plugins"] = {"entries": {"bonjour": {"enabled": False}}}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(cfg, indent=2))
        return cfg

    def pid(self) -> int | None:
        try:
            pid = int(self._pid_path.read_text())
            os.kill(pid, 0)
            return pid
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            return None

    def listening(self) -> bool:
        with socket.socket() as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self, timeout: float = 90) -> Gateway:
        """Start the gateway (or reuse the one this home already started) and wait until ready."""
        if not shutil.which(config.OPENCLAW_BIN):
            raise RuntimeError("openclaw is not on PATH: npm i -g openclaw")
        if self.pid() and self.listening():
            return self
        if self.listening():
            raise RuntimeError(f"port {self.port} is in use by another process; set SIA_OPENCLAW_SDK_PORT")
        self.write_config()
        log = self.log_path.open("w")
        proc = subprocess.Popen([config.OPENCLAW_BIN, "gateway", "run", "--port", str(self.port)],
                                env=self.env(), stdout=log, stderr=subprocess.STDOUT,
                                cwd=self.home, start_new_session=True)
        self._pid_path.write_text(str(proc.pid))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"gateway exited ({proc.returncode}):\n{self.log_path.read_text()[-1500:]}")
            if "[gateway] ready" in _ANSI.sub("", self.log_path.read_text(errors="replace")):
                return self
            time.sleep(0.5)
        raise TimeoutError(f"gateway not ready after {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        pid = self.pid()
        if pid:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(40):
                if not self.pid():
                    break
                time.sleep(0.25)
        self._pid_path.unlink(missing_ok=True)

    def __enter__(self) -> Gateway:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------------------------
# openclaw-sdk, adapted to protocol v4
# ---------------------------------------------------------------------------------------------

def device_identity(path: Path) -> dict:
    """An Ed25519 device key for the SDK, created once. The device id is the key's fingerprint."""
    if path.exists():
        return json.loads(path.read_text())
    from cryptography.hazmat.primitives import serialization as ser
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    raw = key.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
    ident = {
        "deviceId": hashlib.sha256(raw).hexdigest(),
        "privateKeyPem": key.private_bytes(ser.Encoding.PEM, ser.PrivateFormat.PKCS8, ser.NoEncryption()).decode(),
        "publicKeyPem": key.public_key().public_bytes(ser.Encoding.PEM, ser.PublicFormat.SubjectPublicKeyInfo).decode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ident))
    path.chmod(0o600)
    return ident


def _sdk_gateway_class():
    from openclaw_sdk.gateway import protocol

    class SDKGateway(protocol.ProtocolGateway):
        """openclaw-sdk's ProtocolGateway with a protocol-v4 `connect` handshake."""

        def __init__(self, url: str, token: str, identity: dict, **kw):
            super().__init__(url, token=token, **kw)
            self._identity = identity
            self.connect_error: Exception | None = None

        async def _handle_challenge(self, payload: dict[str, Any]) -> None:
            nonce = payload.get("nonce", "")
            signed_at = int(payload.get("ts") or time.time() * 1000)  # v4: sign the server's timestamp
            d = self._identity
            signature = protocol._sign_device_payload(d["privateKeyPem"], "|".join(
                ["v2", d["deviceId"], "cli", "cli", "operator", ",".join(SCOPES), str(signed_at), self._token, nonce]))
            params = {
                "minProtocol": PROTOCOL, "maxProtocol": PROTOCOL,
                "client": {"id": "cli", "version": protocol._SDK_VERSION, "platform": platform.system().lower(),
                           "mode": "cli"},
                "role": "operator", "scopes": SCOPES,
                "caps": ["tool-events"],  # without it the gateway doesn't stream tool calls to us
                "commands": [], "permissions": {}, "auth": {"token": self._token},
                "locale": "en-US", "userAgent": f"openclaw-sdk/{protocol._SDK_VERSION} (sia)",
                "device": {"id": d["deviceId"], "publicKey": protocol._extract_raw_public_key(d["publicKeyPem"]),
                           "signature": signature, "signedAt": signed_at, "nonce": nonce},
            }
            req_id = self._next_id()
            self._connect_req_id = req_id
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(lambda f: setattr(self, "connect_error", f.exception()))
            self._pending[req_id] = future
            await self._send_json({"type": "req", "id": req_id, "method": "connect", "params": params})

        async def _do_connect(self) -> None:
            await super()._do_connect()
            if self.connect_error is not None:  # the SDK would otherwise report a failed handshake as connected
                await self._cleanup_ws()
                self._connected = False
                raise self.connect_error

    return SDKGateway


async def connect(gateway: Gateway, callbacks: list | None = None, timeout: int = 300):
    """An openclaw-sdk OpenClawClient connected to `gateway`."""
    from openclaw_sdk import OpenClawClient
    from openclaw_sdk.core.config import ClientConfig

    ws = _sdk_gateway_class()(gateway.url, gateway.token, device_identity(gateway.identity_path),
                              default_timeout=60)
    await ws.connect()
    cfg = ClientConfig(gateway_ws_url=gateway.url, api_key=gateway.token, timeout=timeout)
    return OpenClawClient(config=cfg, gateway=ws, callbacks=callbacks or [])


# ---------------------------------------------------------------------------------------------
# MLflow on top
# ---------------------------------------------------------------------------------------------

def _ns(ms: float | None) -> int | None:
    return int(ms * 1_000_000) if ms else None


def _text(content: Any, limit: int = 4000) -> str:
    """Text out of an OpenClaw content value (a string or a list of content blocks)."""
    if isinstance(content, dict):
        content = content.get("content", content)
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content or "")[:limit]


def _callback_base():
    from openclaw_sdk.callbacks.handler import CallbackHandler

    return CallbackHandler


class MLflowTracer(_callback_base()):
    """openclaw-sdk callbacks -> MLflow TOOL spans, live, under `parent`.

    The gateway streams `agent` events; those with stream == "tool" carry
    {phase: "start" | "result", toolCallId, name, args | result, isError} plus a timestamp.
    """

    def __init__(self, parent):
        self.parent = parent
        self.open: dict[str, Any] = {}
        self.events: list[dict] = []  # every raw event, for inspection in the notebooks

    async def on_stream_event(self, agent_id: str, event) -> None:
        payload = event.data.get("payload") or {}
        self.events.append({"event": event.data.get("event"), **payload})
        if payload.get("stream") != "tool":
            return
        data = payload.get("data") or {}
        call_id, ts = data.get("toolCallId"), _ns(payload.get("ts"))
        if data.get("phase") == "start":
            self.open[call_id] = mlflow.start_span_no_context(
                data.get("name") or "tool", span_type=SpanType.TOOL, parent_span=self.parent,
                inputs=data.get("args"), start_time_ns=ts, attributes={"openclaw.tool_call_id": call_id})
        elif data.get("phase") == "result" and call_id in self.open:
            self.open.pop(call_id).end(outputs={"result": _text(data.get("result"))},
                                       status="ERROR" if data.get("isError") else "OK", end_time_ns=ts)

    def close(self) -> None:
        for span in self.open.values():  # tools that never reported a result
            span.end(status="ERROR")
        self.open.clear()


def add_llm_spans(parent, history: list[dict], run_id: str) -> list[dict]:
    """One CHAT_MODEL span per assistant turn of this run, from the session's chat.history."""
    turns = []
    for i, msg in enumerate(history):
        meta = msg.get("__openclaw") or {}
        if msg.get("role") != "assistant" or meta.get("runId") != run_id:
            continue
        usage = msg.get("usage") or {}
        cost = usage.get("cost") or {}
        blocks = msg.get("content") or []
        calls = [{"name": b.get("name"), "arguments": b.get("arguments")} for b in blocks
                 if isinstance(b, dict) and b.get("type") == "toolCall"]
        cached = (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)
        attrs = {
            SpanAttributeKey.MODEL: msg.get("model"), SpanAttributeKey.MODEL_PROVIDER: msg.get("provider"),
            SpanAttributeKey.CHAT_USAGE: {
                TokenUsageKey.INPUT_TOKENS: (usage.get("input") or 0) + cached,
                TokenUsageKey.OUTPUT_TOKENS: usage.get("output") or 0,
                TokenUsageKey.TOTAL_TOKENS: usage.get("totalTokens") or 0,
                TokenUsageKey.CACHE_READ_INPUT_TOKENS: usage.get("cacheRead") or 0,
                TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: usage.get("cacheWrite") or 0,
            },
            "openclaw.stop_reason": msg.get("stopReason"), "openclaw.response_id": msg.get("responseId"),
        }
        if cost:
            input_cost = sum(cost.get(k) or 0 for k in ("input", "cacheRead", "cacheWrite"))
            attrs[SpanAttributeKey.LLM_COST] = {CostKey.INPUT_COST: input_cost,
                                                CostKey.OUTPUT_COST: cost.get("output") or 0,
                                                CostKey.TOTAL_COST: cost.get("total") or 0}
        span = mlflow.start_span_no_context(
            msg.get("model") or "llm", span_type=SpanType.CHAT_MODEL, parent_span=parent,
            inputs={"messages": [{"role": m.get("role"), "content": _text(m.get("content"), 1500)}
                                 for m in history[max(0, i - 3):i]]},
            start_time_ns=_ns(msg.get("timestamp")), attributes=attrs)
        span.end(outputs={"content": _text(blocks), "tool_calls": calls},
                 end_time_ns=_ns(meta.get("recordTimestampMs") or msg.get("timestamp")))
        turns.append({"model": msg.get("model"), "usage": usage, "tool_calls": [c["name"] for c in calls]})
    return turns


class OpenClawAgent:
    """A synchronous, MLflow-traced handle on one OpenClaw agent, through openclaw-sdk."""

    def __init__(self, gateway: Gateway, agent_id: str = "main", timeout: int = 300):
        self.gateway, self.agent_id, self.timeout = gateway, agent_id, timeout
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="openclaw-sdk", daemon=True).start()
        self.client = self.run(connect(gateway, timeout=timeout))
        self.last: dict = {}

    def run(self, coro, timeout: float | None = None):
        """Run a coroutine on the SDK's event loop from ordinary (sync) code."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def agent(self, session: str | None = None):
        return self.client.get_agent(self.agent_id, session_name=session or f"sia-{uuid.uuid4().hex[:12]}")

    # -- the agent's instructions: AGENTS.md through the gateway's agents.files.* methods ---------
    def get_agents_md(self) -> str:
        return self.run(self.agent().get_file("AGENTS.md"))["file"].get("content", "")

    def set_agents_md(self, text: str) -> None:
        self.run(self.agent().set_file("AGENTS.md", text))

    # -- answering ---------------------------------------------------------------------------------
    def ask(self, question: str, session: str | None = None, tags: dict | None = None) -> str:
        """Answer one question as one trace (or as a child of the caller's active span)."""
        with mlflow.start_span("acme_support_agent", span_type=SpanType.AGENT) as root:
            root.set_inputs({"question": question})
            if tags:
                mlflow.update_current_trace(tags=tags)
            with mlflow.start_span("openclaw.run", span_type=SpanType.CHAIN) as run_span:
                run_span.set_inputs({"message": question, "agent": self.agent_id})
                out = self.run(self._execute(question, session, run_span), timeout=self.timeout + 30)
                run_span.set_attributes({"openclaw.run_id": out["run_id"], "openclaw.session_key": out["session_key"],
                                         "openclaw.latency_ms": out["latency_ms"]})
                run_span.set_outputs({"text": out["text"], "stop_reason": out["stop_reason"]})
            root.set_outputs(out["text"])
        self.last = out
        return out["text"]

    async def _execute(self, question: str, session: str | None, parent) -> dict:
        agent = self.agent(session)
        tracer = MLflowTracer(parent)
        try:
            result = await agent.execute(question, callbacks=[tracer])
        finally:
            tracer.close()
        run_id = next((e.get("runId") for e in tracer.events if e.get("runId")), "")
        history = (await self.client.gateway.call("chat.history", {"sessionKey": agent.session_key})).get("messages", [])
        turns = add_llm_spans(parent, history, run_id)
        if not result.success:
            raise RuntimeError(f"openclaw run failed: {result.error_message or result.stop_reason}")
        return {"text": result.content, "run_id": run_id, "session_key": agent.session_key,
                "latency_ms": result.latency_ms, "stop_reason": result.stop_reason,
                "turns": turns, "events": tracer.events}

    def close(self) -> None:
        try:
            self.run(self.client.close(), timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
