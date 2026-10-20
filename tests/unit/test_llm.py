"""Provider plumbing, tested with LiteLLM's mock responses (no keys, no network)."""

import json

import litellm
import pytest

from sia import llm


@pytest.mark.parametrize("uri, litellm_id, openclaw_id", [
    ("anthropic:/claude-haiku-4-5", "anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
    ("openai:/gpt-5.4-mini", "openai/gpt-5.4-mini", "openai/gpt-5.4-mini"),
    ("gemini:/gemini-3.5-flash", "gemini/gemini-3.5-flash", "google/gemini-3.5-flash"),
    ("ollama:/gemma4:e4b", "ollama_chat/gemma4:e4b", "ollama/gemma4:e4b"),
    ("groq/llama-3.3-70b-versatile", "groq/llama-3.3-70b-versatile", "groq/llama-3.3-70b-versatile"),
])
def test_model_id_translation(uri, litellm_id, openclaw_id):
    assert llm.litellm_model(uri) == litellm_id
    assert llm.openclaw_model(uri) == openclaw_id


def test_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not llm.has_credentials("openai:/gpt-5.5")
    assert llm.missing_key_hint("openai:/gpt-5.5") == "set OPENAI_API_KEY"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.has_credentials("openai:/gpt-5.5")


def test_ollama_readiness(monkeypatch):
    """Ollama needs no key, but the server must be up and the model pulled."""
    monkeypatch.setattr(llm, "ollama_models", lambda: None)
    assert not llm.has_credentials("ollama:/gemma4:e4b")
    assert "ollama serve" in llm.missing_key_hint("ollama:/gemma4:e4b")
    monkeypatch.setattr(llm, "ollama_models", lambda: {"gemma4:e4b"})
    assert llm.has_credentials("ollama:/gemma4:e4b")
    assert not llm.has_credentials("ollama:/gpt-oss:20b")
    assert llm.missing_key_hint("ollama:/gpt-oss:20b") == "ollama pull gpt-oss:20b"


class _Captured(Exception):
    pass


def test_ollama_calls_request_a_larger_context(monkeypatch):
    seen = {}

    def capture(**kw):
        seen.update(kw)
        raise _Captured

    monkeypatch.setattr(litellm, "completion", capture)
    with pytest.raises(_Captured):
        llm.chat("ollama:/gemma4:e4b", [{"role": "user", "content": "hi"}])
    assert seen["model"] == "ollama_chat/gemma4:e4b" and seen["num_ctx"] == llm.OLLAMA_NUM_CTX


@pytest.fixture
def scripted(monkeypatch):
    """Replace litellm.completion with a script of mock replies, recording each request."""
    real, requests = litellm.completion, []

    def install(*replies):
        it = iter(replies)

        def fake(**kw):
            requests.append(kw)
            reply = next(it)
            if isinstance(reply, Exception):
                raise reply
            return real(**kw, **reply)
        monkeypatch.setattr(litellm, "completion", fake)
        return requests
    return install


def tool_call(name, **args):
    return {"mock_tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_direct_runtime_tool_loop(scripted, tmp_path, monkeypatch):
    from sia import config
    from sia.agent import run_direct
    from sia.workspace import build_workspace

    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path)
    monkeypatch.setattr(config, "AGENT_MODEL", "openai:/gpt-5.4-mini")   # the LiteLLM path (mocked)
    requests = scripted(tool_call("read_file", path="knowledge/billing.md"),
                        {"mock_response": "Refunds are available within 14 days."})
    ws = build_workspace("Read knowledge/ before answering.", "t")
    assert run_direct("Refund window?", ws) == "Refunds are available within 14 days."
    tool_msg = requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "14 days of purchase" in tool_msg["content"]


def test_structured_output_forced_tool(scripted):
    from sia.improvers import PROPOSAL_TOOL

    scripted(tool_call("submit_proposal", themes=[], agents_md="# new", change_summary="ok"))
    out = llm.structured("openai:/gpt-5.5", "sys", "user", PROPOSAL_TOOL)
    assert out["agents_md"] == "# new"


def test_structured_output_falls_back_to_json_text(scripted):
    from sia.improvers import PROPOSAL_TOOL

    err = litellm.exceptions.BadRequestError("tool_choice unsupported", model="m", llm_provider="x")
    scripted(err, {"mock_response": 'Sure: {"themes": [], "agents_md": "# y", "change_summary": "s"}'})
    assert llm.structured("ollama:/gpt-oss:20b", "sys", "user", PROPOSAL_TOOL)["agents_md"] == "# y"


def test_claude_structured_output_schema_is_closed_and_fully_required():
    from sia.improvers import PROPOSAL_TOOL

    schema = llm.strict_schema(PROPOSAL_TOOL["function"]["parameters"])
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    skill = schema["properties"]["skill"]
    assert skill["additionalProperties"] is False and set(skill["required"]) == {"name", "description", "body"}
    theme = schema["properties"]["themes"]["items"]
    assert theme["additionalProperties"] is False and "criterion" in theme["required"]


def test_claude_is_the_default_preset():
    agent, judge = llm.PRESETS["anthropic"]
    assert agent == "anthropic:/claude-haiku-5-5" and judge == "anthropic:/claude-opus-5-5"


def test_direct_runtime_on_claude_uses_the_anthropic_sdk(tmp_path, monkeypatch):
    """The Claude path: strict tools, effort, caching, and assistant turns appended unchanged."""
    from types import SimpleNamespace

    from sia import config
    from sia.agent import run_direct
    from sia.workspace import build_workspace

    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                block = SimpleNamespace(type="tool_use", id="tu1", name="read_file", input={"path": "knowledge/billing.md"})
                return SimpleNamespace(stop_reason="tool_use", content=[block])
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Within 14 days.")])

    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path)
    monkeypatch.setattr(config, "AGENT_MODEL", "anthropic:/claude-haiku-5-5")
    monkeypatch.setattr(llm, "claude_client", lambda: SimpleNamespace(messages=FakeMessages()))
    ws = build_workspace("Read knowledge/ before answering.", "claude")

    assert run_direct("Refund window?", ws) == "Within 14 days."
    first, second = calls
    assert first["model"] == "claude-haiku-5-5" and first["output_config"] == {"effort": config.AGENT_EFFORT}
    assert all(t["strict"] for t in first["tools"]) and "tool_choice" not in first and "temperature" not in first
    assert first["cache_control"] == {"type": "ephemeral"}
    assistant, tool_result = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][0].id == "tu1"      # appended unchanged
    assert tool_result["content"][0]["tool_use_id"] == "tu1" and "14 days" in tool_result["content"][0]["content"]
