"""The Acme Cloud support agent: one traced entry point, three interchangeable runtimes.

    openclaw  - the real thing. Shells out to `openclaw agent --local ... --json` against an
                isolated OpenClaw profile, with the workspace rendered from the Prompt Registry.
    direct    - a small tool-use loop (any LiteLLM provider) that gives the model the same
                workspace files and the same "read a file" ability. Seconds per call instead of tens of seconds,
                which matters when an optimizer calls the agent hundreds of times.
    offline   - a deterministic simulator with no LLM at all. It obeys (a few) instructions in
                AGENTS.md so the whole loop can be rehearsed or unit-tested without API keys.

Every call produces one MLflow trace with an AGENT root span, tagged with topic, backend, and
the prompt version that produced it -- which is what lets later levels slice signal by version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from sia import config, llm
from sia.scenario import case_for_question, route_topic
from sia.workspace import build_workspace, workspace_files


# --------------------------------------------------------------------------------------------
# Prompt Registry <-> workspace
# --------------------------------------------------------------------------------------------
def load_agents_md(ref: str | int = "production"):
    """Load AGENTS.md from the Prompt Registry by alias ("production") or version number."""
    uri = f"prompts:/{config.PROMPT_NAME}/{ref}" if str(ref).isdigit() else f"prompts:/{config.PROMPT_NAME}@{ref}"
    return mlflow.genai.load_prompt(uri)


@lru_cache(maxsize=256)
def _workspace_for(text: str, label: str) -> Path:
    return build_workspace(text, label)


def workspace_for_prompt(prompt) -> Path:
    # prompt.template (not the registry copy) is what we render: during optimize_prompts MLflow
    # patches .template to return each *candidate*, so the hash keeps candidates apart on disk.
    text = prompt.template
    return _workspace_for(text, f"v{prompt.version}-{hashlib.sha1(text.encode()).hexdigest()[:8]}")


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------
def openclaw_env(workspace: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["OPENCLAW_STATE_DIR"] = str(config.OPENCLAW_STATE_DIR)
    env["OPENCLAW_CONFIG_PATH"] = str(config.OPENCLAW_STATE_DIR / "openclaw.json")
    if workspace is not None:
        env["OPENCLAW_WORKSPACE_DIR"] = str(workspace)
    return env


def _extract_text(payload: dict) -> str:
    """`openclaw agent --json` has shipped a few envelope shapes; accept all of them."""
    if isinstance(payload.get("final"), str):
        return payload["final"]
    for key in ("payloads", "messages"):
        items = payload.get(key) or (payload.get("result") or {}).get(key) or []
        texts = [i.get("text", "") for i in items if isinstance(i, dict)]
        if any(texts):
            return "\n".join(t for t in texts if t)
    for key in ("text", "reply", "output"):
        if isinstance(payload.get(key), str):
            return payload[key]
    raise ValueError(f"Could not find the reply in OpenClaw output: {json.dumps(payload)[:500]}")


def run_openclaw(question: str, workspace: Path) -> str:
    cmd = [
        config.OPENCLAW_BIN, "agent", "--local",
        "--agent", "main",
        "--session-id", f"sia-{uuid.uuid4().hex[:12]}",   # fresh session: no memory between questions
        "--model", llm.openclaw_model(config.AGENT_MODEL),
        "--thinking", "off",
        "--message", question,
        "--json",
    ]
    with mlflow.start_span("openclaw.agent", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"message": question, "workspace": str(workspace)})
        env = openclaw_env(workspace)
        # With the connectors/openclaw-mlflow plugin enabled, OpenClaw's own trace of this run is tagged
        # with ours, so the two can be joined (`tags.sia.parent_trace_id = '<trace id>'`).
        env["MLFLOW_OTEL_TRACE_TAGS"] = json.dumps({"sia.parent_trace_id": span.trace_id, "sia.workspace": workspace.name})
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"openclaw exited {proc.returncode}: {proc.stderr[-2000:]}")
        payload = json.loads(proc.stdout)
        meta = payload.get("meta", {})
        tools = payload.get("toolSummary") or meta.get("toolSummary") or {}
        usage = payload.get("usage") or (meta.get("agentMeta") or {}).get("usage") or {}
        span.set_attributes({
            "openclaw.tool_calls": tools.get("calls", 0),
            "openclaw.tools": ",".join(tools.get("tools", [])),
            "openclaw.usage": json.dumps(usage),
        })
        text = _extract_text(payload)
        span.set_outputs({"text": text})
        return text


_DIRECT_TOOLS = [
    llm.tool("list_files", "List the files in your workspace.", {"type": "object", "properties": {}}),
    llm.tool("read_file", "Read a file from your workspace, e.g. knowledge/billing.md.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
]


def run_direct(question: str, workspace: Path) -> str:
    """A minimal tool-use loop over the workspace files, on any provider LiteLLM supports."""
    files = workspace_files(workspace)
    messages = [{"role": "system", "content": files["AGENTS.md"]}, {"role": "user", "content": question}]

    @mlflow.trace(span_type=SpanType.TOOL)
    def list_files() -> str:
        return "\n".join(p for p in files if p != "AGENTS.md")

    @mlflow.trace(span_type=SpanType.TOOL)
    def read_file(path: str) -> str:
        return files.get(path.strip().removeprefix("./"), f"No such file: {path}")

    for _ in range(6):
        msg = llm.chat(config.AGENT_MODEL, messages, tools=_DIRECT_TOOLS, max_tokens=800)
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            out = list_files() if call.function.name == "list_files" else read_file(args.get("path", ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return "Sorry, I couldn't finish looking that up."


def run_offline(question: str, workspace: Path) -> str:
    """A rule-based stand-in for an LLM. It reads AGENTS.md and reacts to a few instructions:

    - references knowledge/<topic>.md (or all of knowledge/) -> answers correctly for that topic
    - otherwise                        -> answers from the stale "quick facts", or guesses
    - "Terms of Service" disclaimer    -> opens every answer with that disclaimer
    - "best guess"                     -> guesses when it doesn't know; otherwise escalates
    """
    agents_md = (workspace / "AGENTS.md").read_text()
    case = case_for_question(question)
    reads_topic = case is not None and (
        f"{case.topic}.md" in agents_md or re.search(r"every file in .?knowledge/", agents_md, re.I))
    if reads_topic:
        body = case.expected_response
    elif re.search(r"refund", question, re.I) and "30 days" in agents_md:
        body = "You can get a refund within 30 days of purchase."
    elif re.search(r"reset", question, re.I) and "24 hours" in agents_md:
        body = "Password reset links are valid for 24 hours."
    elif "best guess" in agents_md.lower():
        body = "Yes, that should be available on your plan. It usually takes about 24 hours."
    else:
        body = "I don't have that information, so I'm escalating this to a human teammate."
    if "terms of service" in agents_md.lower():
        body = ("Policies can vary by plan and region, so please double-check the Terms of Service. "
                + body)
    return body


BACKENDS = {"openclaw": run_openclaw, "direct": run_direct, "offline": run_offline}


# --------------------------------------------------------------------------------------------
# The traced entry point
# --------------------------------------------------------------------------------------------
@mlflow.trace(name="acme_support_agent", span_type=SpanType.AGENT)
def run_agent(question: str, prompt_ref: str | int = "production", backend: str | None = None,
              source: str = "eval") -> str:
    """Answer one customer question with the given AGENTS.md version. Produces one trace.

    source="production" marks real (simulated) customer traffic. Everything else -- evaluation,
    optimization, CI -- is "eval", so it never leaks back in as production signal.
    """
    backend = backend or config.BACKEND
    prompt = load_agents_md(prompt_ref)            # loading inside the trace links prompt <-> trace
    workspace = workspace_for_prompt(prompt)
    if mlflow.get_current_active_span() is not None:  # evaluate() probes predict_fn untraced once
        mlflow.update_current_trace(tags={
            "topic": route_topic(question),
            "backend": backend,
            "prompt_version": str(prompt.version),
            "source": source,
        })
    return BACKENDS[backend](question, workspace)


def predict_fn_for(prompt_ref: str | int, backend: str | None = None):
    """A predict_fn for mlflow.genai.evaluate() pinned to one prompt version."""

    def predict(question: str) -> str:
        return run_agent(question, prompt_ref=prompt_ref, backend=backend)

    return predict


def latest_version() -> int:
    versions = mlflow.MlflowClient().search_prompt_versions(config.PROMPT_NAME)
    return max(int(v.version) for v in versions)


def register_candidate(text: str, commit_message: str, author: str, level: int, **tags) -> int:
    """Register a new AGENTS.md version *without* promoting it. Returns the version number."""
    pv = mlflow.genai.register_prompt(
        name=config.PROMPT_NAME, template=text, commit_message=commit_message,
        tags={"author": author, "level": str(level), **{k: str(v) for k, v in tags.items()}},
    )
    return int(pv.version)
