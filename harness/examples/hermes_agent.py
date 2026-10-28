"""Bring Hermes Agent (Nous Research): the python adapter around `run_agent.AIAgent`.

    SI_AGENT=python
    SI_AGENT_TARGET=examples.hermes_agent:answer
    SI_AGENT_MODEL=anthropic:/claude-haiku-5-5
    SI_INSTRUCTIONS_FILE=examples/instructions.md       # rendered as AGENTS.md for Hermes
    SI_WORKSPACE_DIR=examples/knowledge

    pip install hermes-agent==0.19.0

Hermes reads AGENTS.md from its working directory, so each call runs inside the workspace the
harness rendered for the instructions version under test. Hermes' tool callbacks become TOOL
spans under the harness's AGENT span. Hermes reads files with its own `read_file` tool, so give
it instructions that point at knowledge/<topic>.md (the example instructions name the support
agent's `lookup_policy` tool), and change the `requires_tool` check in criteria.yaml to read_file.

Isolation, because Hermes is usually someone's personal agent:
  * HERMES_HOME is a folder under .selfimprove/hermes/, never ~/.hermes;
  * only ANTHROPIC_API_KEY is used. Hermes 0.19 otherwise falls back to Claude Code's saved login
    and misreads newer Console keys (sk-ant-usr-...) as subscription tokens;
  * the workspace is made its own git root: Hermes tells the model the enclosing repository is
    its project, and this repository holds the reviewers' answer key (examples/cases.yaml).
Hermes' own learning loop stays gated: skills and memory writes need approval, so nothing it
decides to save changes the agent behind the harness's back.

These shims touch Hermes internals, which is why the version is pinned. The tutorial's
`notebooks/sdk/03_hermes_from_python.ipynb` (self-improving-harness branch) builds them step by step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from selfimprove import config, models

_LOCK = threading.Lock()  # Hermes reads AGENTS.md from os.getcwd(), which is process-wide


def _home(workspace: Path) -> Path:
    home = config.settings.state_dir / "hermes" / workspace.name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("skills:\n  write_approval: true\nmemory:\n  write_approval: true\n")
    return home


def _fence(workspace: Path) -> None:
    if not (workspace / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(workspace)], check=False, capture_output=True)


@lru_cache(maxsize=1)
def _hermes():
    os.environ.setdefault("HERMES_HOME", str(config.settings.state_dir / "hermes" / "base"))
    Path(os.environ["HERMES_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        from agent import anthropic_adapter as aa
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from run_agent import AIAgent
    except ImportError as e:
        raise RuntimeError("examples.hermes_agent needs `pip install hermes-agent==0.19.0`") from e

    is_oauth = aa._is_oauth_token

    def _is_oauth_token(key) -> bool:  # our API key is an API key
        if isinstance(key, str) and key and key == os.getenv("ANTHROPIC_API_KEY") and not key.startswith("sk-ant-oat"):
            return False
        return is_oauth(key)

    aa._is_oauth_token = _is_oauth_token
    aa.resolve_anthropic_token = lambda: os.getenv("ANTHROPIC_API_KEY") or None  # no saved-login fallback
    return AIAgent, set_hermes_home_override, reset_hermes_home_override


def answer(question: str, instructions: str, workspace: Path) -> str:
    AIAgent, set_home, reset_home = _hermes()
    provider, model = models.split(config.settings.agent_model)
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Hermes on Claude needs ANTHROPIC_API_KEY in .env")
    (workspace / "AGENTS.md").write_text(instructions)
    _fence(workspace)
    open_tools: dict[str, object] = {}

    with mlflow.start_span("hermes.agent", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"message": question, "workspace": str(workspace)})

        def tool_start(call_id, name, args, *_, **__):
            open_tools[call_id] = mlflow.start_span_no_context(
                name, span_type=SpanType.TOOL, parent_span=span,
                inputs=args if isinstance(args, dict) else {"args": str(args)})

        def tool_complete(call_id, name, args, result, *_, **__):
            if (tool := open_tools.pop(call_id, None)) is not None:
                tool.end(outputs={"result": str(result)[:4000]})

        with _LOCK:
            token, cwd = set_home(_home(workspace)), os.getcwd()
            os.chdir(workspace)
            try:
                agent = AIAgent(model=model, provider=provider, api_key=os.getenv("ANTHROPIC_API_KEY"),
                                enabled_toolsets=["file"], skip_memory=True, quiet_mode=True, max_iterations=8,
                                max_tokens=1024, session_id=f"si-{uuid.uuid4().hex[:12]}",
                                tool_start_callback=tool_start, tool_complete_callback=tool_complete)
                out = agent.run_conversation(user_message=question)
            finally:
                os.chdir(cwd)
                reset_home(token)
        for tool in open_tools.values():
            tool.end(status="ERROR")
        text = out.get("final_response") or ""
        if not text and out.get("error"):
            raise RuntimeError(f"hermes failed: {str(out['error'])[:500]}")
        span.set_outputs({"text": text})
        return text
