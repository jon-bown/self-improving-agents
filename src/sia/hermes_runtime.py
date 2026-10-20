"""Hermes Agent (Nous Research) as a third agent runtime: SIA_BACKEND=hermes.

Hermes is Python, so it runs in-process through its library API (`run_agent.AIAgent`), against
the same rendered workspace as OpenClaw:

    workspace/AGENTS.md      Hermes loads it from the working directory as project context
    workspace/knowledge/     read with Hermes' read_file tool
    workspace/skills/        copied into this agent version's own Hermes home (skills live there)

Isolation. Every agent version gets its own Hermes home under .hermes-state/homes/<workspace>/
(config.yaml + skills/), selected per run with Hermes' context-local home override, so your
~/.hermes is never touched and two versions never share skills or settings.

Tracing. MLflow's OpenAI autolog does see Hermes' model calls, but Hermes makes them on a worker
thread, so each one lands as its own disconnected trace. Instead, a small subclass wraps Hermes'
two model-call methods in LLM spans on the calling thread, and Hermes' tool callbacks become
TOOL spans -- one trace per answer, the same shape the OpenClaw runtime produces:

    acme_support_agent (AGENT)
      └─ hermes.agent (CHAIN)
           ├─ llm_call (CHAT_MODEL)   tokens, finish reason, tool calls requested
           ├─ read_file (TOOL)        {"path": "knowledge/billing.md"} -> contents
           └─ llm_call (CHAT_MODEL)

The subclass overrides private methods, which is why pyproject pins hermes-agent==0.19.0.

Claude. Hermes' native Anthropic provider (the Anthropic SDK), with your ANTHROPIC_API_KEY.

Ollama. Hermes refuses tool use below 64K tokens of context, and it talks to Ollama's
OpenAI-compatible /v1 endpoint, which ignores per-request num_ctx (Ollama's default is 4K, so
answers get silently truncated). sia.llm builds a local alias with the window baked in
(gemma4:e4b -> gemma4-e4b-64k, metadata only) the first time it's needed.

Hermes' own learning loop stays on but gated: skills.write_approval and memory.write_approval
are true, so anything Hermes decides to save is *staged* under <home>/pending/ for the MLflow
gate to approve or reject, never written straight into the agent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from sia import config, llm

# Hermes reads AGENTS.md from os.getcwd(), which is process-wide: one Hermes run at a time.
_LOCK = threading.Lock()


def traced_agent_class(base):
    """Subclass Hermes' AIAgent (or a test double) so its model calls become LLM spans."""

    class TracedAIAgent(base):
        """LLM spans on the caller's thread around Hermes' model calls."""

        _sia_parent = None

        def _interruptible_api_call(self, api_kwargs):
            return self._sia_traced(super()._interruptible_api_call, api_kwargs)

        def _interruptible_streaming_api_call(self, api_kwargs, **kw):
            parent_call = super()._interruptible_streaming_api_call
            return self._sia_traced(lambda a: parent_call(a, **kw), api_kwargs)

        def _sia_traced(self, call, api_kwargs):
            msgs = api_kwargs.get("messages") or []
            span = mlflow.start_span_no_context(
                "llm_call", span_type=SpanType.CHAT_MODEL, parent_span=self._sia_parent,
                inputs={"model": api_kwargs.get("model"), "messages": msgs[-3:], "n_messages": len(msgs)},
            )
            try:
                resp = call(api_kwargs)
            except Exception as e:
                span.end(status="ERROR", attributes={"error": str(e)[:500]})
                raise
            try:
                outputs, attrs = _response_summary(resp)
            except Exception:
                outputs, attrs = {"raw": str(resp)[:1000]}, {}
            attrs[SpanAttributeKey.MODEL] = api_kwargs.get("model") or self.model
            attrs[SpanAttributeKey.MODEL_PROVIDER] = getattr(self, "provider", None)
            span.end(outputs=outputs, attributes=attrs)
            return resp


    return TracedAIAgent


def _response_summary(resp) -> tuple[dict, dict]:
    """Outputs and MLflow usage attributes for a Hermes model response.

    Hermes returns an OpenAI-style ChatCompletion for OpenAI-compatible providers and the Anthropic
    SDK's Message for its native Anthropic provider (Claude)."""
    usage = getattr(resp, "usage", None)
    if getattr(resp, "choices", None):                       # OpenAI-compatible
        choice = resp.choices[0]
        outputs = {"content": choice.message.content, "finish_reason": choice.finish_reason,
                   "tool_calls": [tc.function.name for tc in (choice.message.tool_calls or [])]}
        tokens = {TokenUsageKey.INPUT_TOKENS: getattr(usage, "prompt_tokens", 0) or 0,
                  TokenUsageKey.OUTPUT_TOKENS: getattr(usage, "completion_tokens", 0) or 0} if usage else {}
    else:                                                     # Anthropic Message
        blocks = getattr(resp, "content", None) or []
        outputs = {"content": "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"),
                   "finish_reason": getattr(resp, "stop_reason", None),
                   "tool_calls": [b.name for b in blocks if getattr(b, "type", "") == "tool_use"]}
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tokens = {TokenUsageKey.INPUT_TOKENS: (getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write,
                  TokenUsageKey.OUTPUT_TOKENS: getattr(usage, "output_tokens", 0) or 0,
                  TokenUsageKey.CACHE_READ_INPUT_TOKENS: cache_read,
                  TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: cache_write} if usage else {}
    attrs = {}
    if tokens:
        tokens[TokenUsageKey.TOTAL_TOKENS] = tokens[TokenUsageKey.INPUT_TOKENS] + tokens[TokenUsageKey.OUTPUT_TOKENS]
        attrs[SpanAttributeKey.CHAT_USAGE] = tokens
    return outputs, attrs


@lru_cache(maxsize=1)
def _hermes():
    """Import Hermes lazily (it's an optional extra) with a repo-local default home."""
    os.environ.setdefault("HERMES_HOME", str(config.HERMES_STATE_DIR / "base"))
    # mlflow.genai.evaluate() switches OpenAI autolog on while it runs; Hermes' model calls happen
    # on a worker thread, so that would add one disconnected trace per call. Our LLM spans cover them.
    mlflow.openai.autolog(disable=True)
    (config.HERMES_STATE_DIR / "base").mkdir(parents=True, exist_ok=True)
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from run_agent import AIAgent
    except ImportError as e:
        raise RuntimeError('SIA_BACKEND=hermes needs the Hermes extra: pip install -e ".[hermes]"') from e

    _api_key_only()
    _keep_notebook_stdio()
    return traced_agent_class(AIAgent), set_hermes_home_override, reset_hermes_home_override


def _keep_notebook_stdio() -> None:
    """Inside Jupyter, stop AIAgent from wrapping sys.stdout/sys.stderr.

    AIAgent replaces both with a broken-pipe-proof wrapper (agent/process_bootstrap.py) whose
    `write` is read-only, when it's created and again in run_conversation(). IPython swaps `write`
    on its streams to capture each cell's output, so after the first Hermes run every later cell
    fails. Kernel streams don't break the way pipes do, so in a kernel the wrapper is skipped.
    """
    from agent import agent_init, conversation_loop, process_bootstrap

    install = process_bootstrap._install_safe_stdio
    if getattr(install, "_sia", False):
        return

    def _install_safe_stdio() -> None:
        if "ipykernel" not in type(sys.stdout).__module__:
            install()

    _install_safe_stdio._sia = True
    for module in (process_bootstrap, agent_init, conversation_loop):  # each imported it by name
        module._install_safe_stdio = _install_safe_stdio


def _api_key_only() -> None:
    """Make Hermes' native Anthropic provider use ANTHROPIC_API_KEY, as an API key, and nothing else.

    Hermes 0.19.0 (agent/anthropic_adapter.py) does two things a demo must not:
    * it treats every `sk-ant-` key except `sk-ant-api...` as a Claude subscription OAuth token, so
      newer Console keys (`sk-ant-usr-...`) are sent as subscription bearer tokens and rejected;
    * when it has no key, or after an auth error, it falls back to Claude Code's saved login
      (~/.claude/.credentials.json or the keychain), i.e. your Claude subscription.
    """
    from agent import anthropic_adapter as aa

    if getattr(aa._is_oauth_token, "_sia", False):
        return
    is_oauth = aa._is_oauth_token

    def _is_oauth_token(key) -> bool:
        if isinstance(key, str) and key and key == os.getenv("ANTHROPIC_API_KEY") and not key.startswith("sk-ant-oat"):
            return False
        return is_oauth(key)

    def resolve_anthropic_token():
        return os.getenv("ANTHROPIC_API_KEY") or None

    _is_oauth_token._sia = True
    aa._is_oauth_token = _is_oauth_token
    aa.resolve_anthropic_token = resolve_anthropic_token


@lru_cache(maxsize=4)
def _endpoint(model_uri: str) -> tuple:
    if llm.split(model_uri)[0] == "ollama":
        llm.ensure_ollama_ctx_variant(llm.split(model_uri)[1], llm.HERMES_NUM_CTX)
    ep = llm.hermes_endpoint(model_uri)
    return ep["model"], ep["provider"], ep["base_url"], ep["api_key"]


def fence(workspace: Path) -> Path:
    """Make the workspace its own git root, so Hermes treats it as the project.

    Hermes tells the model its "Workspace Root" is the enclosing git repository's top level. Every
    workspace here lives inside this repo, whose scenario/ holds the reviewers' answer key, so an
    unfenced Hermes is pointed at the answers. Hermes' file tools have no hard sandbox; this keeps
    the model's picture of the project to the workspace. (The folders are gitignored.)
    """
    if not (workspace / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(workspace)], check=False, capture_output=True)
    return workspace


def home_for(workspace: Path) -> Path:
    """This agent version's Hermes home: config.yaml + the workspace's skills."""
    home = config.HERMES_STATE_DIR / "homes" / workspace.name
    home.mkdir(parents=True, exist_ok=True)
    cfg = ("model:\n"
           f"  ollama_num_ctx: {llm.HERMES_NUM_CTX}\n"
           "skills:\n"
           "  write_approval: true      # Hermes' self-written skills are staged for the MLflow gate\n"
           "memory:\n"
           "  write_approval: true\n")
    (home / "config.yaml").write_text(cfg)
    skills = home / "skills"
    shutil.rmtree(skills, ignore_errors=True)
    if (workspace / "skills").is_dir():
        shutil.copytree(workspace / "skills", skills)
    return home


def run_hermes(question: str, workspace: Path) -> str:
    TracedAIAgent, set_home, reset_home = _hermes()
    model, provider, base_url, api_key = _endpoint(config.AGENT_MODEL)
    if provider == "anthropic" and not api_key:
        raise RuntimeError("Hermes on Claude needs ANTHROPIC_API_KEY (in .env)")
    session_id = f"sia-{uuid.uuid4().hex[:12]}"
    with mlflow.start_span("hermes.agent", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"message": question, "workspace": str(workspace)})
        home = home_for(workspace)
        open_tools: dict[str, object] = {}

        def tool_start(call_id, name, args, *_, **__):
            open_tools[call_id] = mlflow.start_span_no_context(
                name, span_type=SpanType.TOOL, parent_span=span,
                inputs=args if isinstance(args, dict) else {"args": str(args)})

        def tool_complete(call_id, name, args, result, *_, **__):
            tool = open_tools.pop(call_id, None)
            if tool is not None:
                tool.end(outputs={"result": str(result)[:4000]})

        toolsets = ["file"] + (["skills"] if (workspace / "skills").is_dir() else [])
        t0 = time.time()
        fence(workspace)
        with _LOCK:
            token, cwd = set_home(home), os.getcwd()
            os.chdir(workspace)
            try:
                agent = TracedAIAgent(
                    model=model, provider=provider, base_url=base_url, api_key=api_key,
                    enabled_toolsets=toolsets, skip_memory=True, quiet_mode=True,
                    max_iterations=config.HERMES_MAX_ITERATIONS, max_tokens=1024, session_id=session_id,
                    tool_start_callback=tool_start, tool_complete_callback=tool_complete,
                )
                agent._sia_parent = span
                out = agent.run_conversation(user_message=question)
            finally:
                os.chdir(cwd)
                reset_home(token)
        for tool in open_tools.values():  # tools that never reported completion
            tool.end(status="ERROR")
        text = out.get("final_response") or ""
        span.set_attributes({"hermes.session_id": session_id, "hermes.api_calls": out.get("api_calls"),
                             "hermes.completed": bool(out.get("completed")), "hermes.home": str(home),
                             "hermes.seconds": round(time.time() - t0, 1)})
        if not text and out.get("error"):
            raise RuntimeError(f"hermes failed: {str(out.get('error'))[:500]}")
        span.set_outputs({"text": text})
        return text


class HermesAgent:
    """Hermes Agent as a Python object, the same interface as sia.openclaw_client.OpenClawAgent.

        agent = HermesAgent(workspace)
        agent.set_agents_md(text)      # Hermes reads AGENTS.md from its working directory
        agent.ask("Refund window?")    # one MLflow trace: acme_support_agent > hermes.agent > ...
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def get_agents_md(self) -> str:
        path = self.workspace / "AGENTS.md"
        return path.read_text() if path.exists() else ""

    def set_agents_md(self, text: str) -> None:
        (self.workspace / "AGENTS.md").write_text(text)

    def ask(self, question: str, tags: dict | None = None) -> str:
        with mlflow.start_span("acme_support_agent", span_type=SpanType.AGENT) as root:
            root.set_inputs({"question": question})
            if tags:
                mlflow.update_current_trace(tags=tags)
            text = run_hermes(question, self.workspace)
            root.set_outputs(text)
            return text
