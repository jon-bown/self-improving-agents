"""Provider plumbing, tested with LiteLLM's mock responses (no keys, no network)."""

import json

import litellm
import pytest

from sia import llm


@pytest.mark.parametrize("uri, litellm_id, openclaw_id", [
    ("anthropic:/claude-haiku-4-5", "anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
    ("openai:/gpt-5.4-mini", "openai/gpt-5.4-mini", "openai/gpt-5.4-mini"),
    ("gemini:/gemini-3.5-flash", "gemini/gemini-3.5-flash", "google/gemini-3.5-flash"),
    ("ollama:/gpt-oss:20b", "ollama_chat/gpt-oss:20b", "ollama/gpt-oss:20b"),
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
    assert llm.has_credentials("ollama:/gpt-oss:20b")  # local, no key


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
