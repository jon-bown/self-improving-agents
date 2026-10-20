"""The Hermes runtime's tracing and isolation, against a stand-in for Hermes' AIAgent.

The stand-in mimics the parts of hermes-agent 0.19.0 the runtime relies on: a model call through
_interruptible_api_call, tool_start/tool_complete callbacks, and run_conversation's result dict.
"""

import os
from types import SimpleNamespace

import mlflow
from mlflow.entities import SpanType

from sia import config, hermes_runtime, llm, signals


class FakeAIAgent:
    """Answers by 'reading' knowledge/billing.md through the same hooks Hermes uses."""

    seen_cwd = None

    def __init__(self, **kw):
        self.kw = kw

    def _interruptible_api_call(self, api_kwargs):
        msg = SimpleNamespace(content=api_kwargs["reply"], tool_calls=None)
        usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 42})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)

    def run_conversation(self, user_message):
        FakeAIAgent.seen_cwd = os.getcwd()
        self.kw["tool_start_callback"]("c1", "read_file", {"path": "knowledge/billing.md"})
        self.kw["tool_complete_callback"]("c1", "read_file", {"path": "knowledge/billing.md"}, '{"content": "14 days"}')
        resp = self._interruptible_api_call({"model": self.kw["model"], "messages": [], "reply": "Within 14 days."})
        return {"final_response": resp.choices[0].message.content, "completed": True, "api_calls": 1}


def _install_fake(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HERMES_STATE_DIR", tmp_path / "hermes")
    monkeypatch.setattr(config, "AGENT_MODEL", "openai:/gpt-5.4-mini")  # no Ollama variant needed
    homes = []
    fake = (hermes_runtime.traced_agent_class(FakeAIAgent),
            lambda home: homes.append(home) or "token", lambda token: homes.append(None))
    monkeypatch.setattr(hermes_runtime, "_hermes", lambda: fake)
    hermes_runtime._endpoint.cache_clear()
    return homes


def test_hermes_answer_is_one_trace_with_llm_and_tool_spans(monkeypatch, tmp_path):
    homes = _install_fake(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    (ws / "skills" / "handle-x").mkdir(parents=True)
    (ws / "skills" / "handle-x" / "SKILL.md").write_text("---\nname: handle-x\n---\nbody")
    cwd = os.getcwd()

    with mlflow.start_span("acme_support_agent", span_type=SpanType.AGENT):
        mlflow.update_current_trace(tags={"topic": "billing"})
        answer = hermes_runtime.run_hermes("Refund window?", ws)
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    assert answer == "Within 14 days."
    kinds = {s.name: s.span_type for s in trace.data.spans}
    assert kinds["hermes.agent"] == SpanType.CHAIN and kinds["llm_call"] == SpanType.CHAT_MODEL
    assert kinds["read_file"] == SpanType.TOOL
    assert signals.read_right_policy(trace=trace).value is True
    # ran inside the workspace with this version's home, then restored both
    assert FakeAIAgent.seen_cwd == str(ws.resolve()) or FakeAIAgent.seen_cwd == str(ws)
    assert os.getcwd() == cwd and homes[-1] is None
    home = homes[0]
    assert (ws / ".git").exists()            # its own git root: Hermes never sees this repo's answer key
    assert (home / "skills" / "handle-x" / "SKILL.md").exists()
    assert "write_approval: true" in (home / "config.yaml").read_text()


def test_hermes_endpoints():
    ep = llm.hermes_endpoint("ollama:/gemma4:e4b")
    assert ep["model"] == "gemma4-e4b-64k" and ep["base_url"].endswith("/v1")
    assert llm.hermes_endpoint("openai:/gpt-5.5")["base_url"] == "https://api.openai.com/v1"
    claude = llm.hermes_endpoint("anthropic:/claude-haiku-5-5")
    assert claude["provider"] == "anthropic" and claude["model"] == "claude-haiku-5-5"
    assert llm.ollama_ctx_variant("gemma4:e4b", 65536) == "gemma4-e4b-64k"


def test_anthropic_message_usage_is_recorded():
    usage = SimpleNamespace(input_tokens=5, output_tokens=40, cache_read_input_tokens=100, cache_creation_input_tokens=7)
    blocks = [SimpleNamespace(type="text", text="Reading the policy."),
              SimpleNamespace(type="tool_use", name="read_file")]
    outputs, attrs = hermes_runtime._response_summary(SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage))
    assert outputs == {"content": "Reading the policy.", "finish_reason": "tool_use", "tool_calls": ["read_file"]}
    assert attrs["mlflow.chat.tokenUsage"] == {"input_tokens": 112, "output_tokens": 40, "total_tokens": 152,
                                               "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7}


def test_hermes_uses_the_api_key_as_an_api_key(monkeypatch):
    import pytest

    aa = pytest.importorskip("agent.anthropic_adapter")
    monkeypatch.setattr(aa, "_is_oauth_token", aa._is_oauth_token)          # undo the patch after the test
    monkeypatch.setattr(aa, "resolve_anthropic_token", aa.resolve_anthropic_token)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-usr-newer-console-key")
    hermes_runtime._api_key_only()
    assert aa._is_oauth_token("sk-ant-usr-newer-console-key") is False        # Hermes 0.19.0 says True
    assert aa._is_oauth_token("sk-ant-oat01-a-real-setup-token") is True
    assert aa.resolve_anthropic_token() == "sk-ant-usr-newer-console-key"     # never Claude Code's login


def test_hermes_leaves_notebook_stdio_alone(monkeypatch):
    import sys

    import pytest

    pytest.importorskip("agent.agent_init")
    from agent import agent_init, conversation_loop, process_bootstrap

    calls = []
    for module in (process_bootstrap, agent_init, conversation_loop):   # restored after the test
        monkeypatch.setattr(module, "_install_safe_stdio", lambda: calls.append("wrapped"))
    hermes_runtime._keep_notebook_stdio()
    assert agent_init._install_safe_stdio is conversation_loop._install_safe_stdio

    monkeypatch.setattr(sys, "stdout", type("OutStream", (), {"__module__": "ipykernel.iostream"})())
    conversation_loop._install_safe_stdio()
    assert calls == []                       # inside Jupyter: streams untouched
    monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    agent_init._install_safe_stdio()
    assert calls == ["wrapped"]              # anywhere else: Hermes' wrapper as usual
