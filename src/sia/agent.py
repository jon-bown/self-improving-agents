"""The Acme Cloud support agent: one traced entry point, four interchangeable runtimes.

    openclaw  - the real thing. Shells out to `openclaw agent --local ... --json` against an
                isolated OpenClaw profile, with the workspace rendered from the Prompt Registry.
    hermes    - Nous Research's Hermes Agent, in-process, against the same workspace
                (sia/hermes_runtime.py).
    direct    - a small tool-use loop (any LiteLLM provider) that gives the model the same
                workspace files and the same "read a file" ability. Seconds per call instead of tens of seconds,
                which matters when an optimizer calls the agent hundreds of times.
    offline   - a deterministic simulator with no LLM at all. It obeys (a few) instructions in
                AGENTS.md so the whole loop can be rehearsed or unit-tested without API keys.

Every call produces one MLflow trace with an AGENT root span, tagged with topic, backend, and
the AGENTS.md + skill versions that produced it -- which lets later levels slice signal by version.

What the agent *is* = an AgentSpec: one AGENTS.md version plus a set of skill versions. Both
live in the MLflow Prompt Registry (AGENTS.md as `acme-support-agents-md`, each skill as
`acme-skill-<name>`), so either can be the update target and both roll back by moving an alias.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
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


def _skill_uri(name: str, ref: str | int) -> str:
    full = f"{config.SKILL_PREFIX}{name}"
    return f"prompts:/{full}/{ref}" if str(ref).isdigit() else f"prompts:/{full}@{ref}"


def load_skill(name: str, ref: str | int = "production"):
    return mlflow.genai.load_prompt(_skill_uri(name, ref))


def skill_names() -> list[str]:
    """Every skill ever registered (any version), by short name."""
    found = mlflow.MlflowClient().search_prompts(filter_string=f"name LIKE '{config.SKILL_PREFIX}%'")
    return sorted(p.name.removeprefix(config.SKILL_PREFIX) for p in found)


def production_skills() -> dict[str, int]:
    """Skills currently serving: name -> version behind its @production alias."""
    out = {}
    for name in skill_names():
        try:
            out[name] = int(load_skill(name, "production").version)
        except Exception:
            pass  # registered but never promoted (e.g. a rejected proposal)
    return out


@dataclass(frozen=True)
class AgentSpec:
    """AGENTS.md version + skill versions. `skills` overrides/extends the production skill set;
    a version of None removes that skill."""

    agents: str | int = "production"
    skills: tuple[tuple[str, int | None], ...] = field(default_factory=tuple)

    def with_skill(self, name: str, version: int | None) -> "AgentSpec":
        rest = tuple((n, v) for n, v in self.skills if n != name)
        return AgentSpec(self.agents, rest + ((name, version),))

    @property
    def label(self) -> str:
        extra = "".join(f"+{n}@{v}" if v is not None else f"-{n}" for n, v in self.skills)
        return f"{'v' if str(self.agents).isdigit() else '@'}{self.agents}{extra}"


def as_spec(ref) -> AgentSpec:
    return ref if isinstance(ref, AgentSpec) else AgentSpec(agents=ref)


def pin(ref) -> AgentSpec:
    """Resolve aliases to concrete version numbers, so an evaluation always runs one fixed agent."""
    spec = as_spec(ref)
    skills = dict(production_skills())
    skills.update({n: v for n, v in spec.skills})
    return AgentSpec(int(load_agents_md(spec.agents).version),
                     tuple(sorted((n, v) for n, v in skills.items() if v is not None)))


def resolve(ref):
    """-> (AGENTS.md prompt, {skill name: skill prompt}) for the spec."""
    spec = as_spec(ref)
    agents = load_agents_md(spec.agents)
    skills = {n: load_skill(n, v) for n, v in pin(AgentSpec("production", spec.skills)).skills}
    return agents, skills


@lru_cache(maxsize=256)
def _workspace_for(text: str, skills: tuple[tuple[str, str], ...], label: str) -> Path:
    return build_workspace(text, label, skills=dict(skills))


def workspace_for(agents_prompt, skill_prompts: dict | None = None) -> Path:
    # prompt.template (not the registry copy) is what we render: during optimize_prompts MLflow
    # patches .template to return each *candidate*, so the hash keeps candidates apart on disk.
    text = agents_prompt.template
    skills = tuple(sorted((n, p.template) for n, p in (skill_prompts or {}).items()))
    digest = hashlib.sha1((text + json.dumps(skills)).encode()).hexdigest()[:8]
    return _workspace_for(text, skills, f"v{agents_prompt.version}-{digest}")


def workspace_for_prompt(prompt) -> Path:
    return workspace_for(prompt, {})


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------
_OPENCLAW_LOCK = threading.Lock()  # `openclaw agent --local` holds an exclusive lock on its state dir


def openclaw_env(workspace: Path | None = None) -> dict[str, str]:
    """Environment for the isolated profile. With a workspace, points OpenClaw at a per-workspace
    copy of the config: OpenClaw takes the workspace from config (agents.defaults.workspace) and
    ignores OPENCLAW_WORKSPACE_DIR once that key is set."""
    env = dict(os.environ)
    env["OPENCLAW_STATE_DIR"] = str(config.OPENCLAW_STATE_DIR)
    base = config.OPENCLAW_STATE_DIR / "openclaw.json"
    env["OPENCLAW_CONFIG_PATH"] = str(base)
    if workspace is not None:
        cfg = json.loads(base.read_text())
        cfg.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = str(workspace)
        per_ws = config.OPENCLAW_STATE_DIR / "configs" / f"{workspace.name}.json"
        per_ws.parent.mkdir(exist_ok=True)
        per_ws.write_text(json.dumps(cfg, indent=2))
        env["OPENCLAW_CONFIG_PATH"] = str(per_ws)
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
    session_id = f"sia-{uuid.uuid4().hex[:12]}"           # fresh session: no memory between questions
    cmd = [
        config.OPENCLAW_BIN, "agent", "--local",
        "--agent", "main",
        "--session-id", session_id,
        "--model", llm.openclaw_model(config.AGENT_MODEL),
        "--message", question,
        "--json",
    ]
    if llm.split(config.AGENT_MODEL)[0] != "anthropic":
        # Small local models answer faster without thinking. Current Claude models don't accept
        # disabled thinking, so for Claude we leave it to OpenClaw's provider defaults.
        cmd += ["--thinking", "off"]
    with mlflow.start_span("openclaw.agent", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"message": question, "workspace": str(workspace)})
        env = openclaw_env(workspace)
        with _OPENCLAW_LOCK:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            if config.OPENCLAW_SPANS:
                # Rebuild OpenClaw's own LLM + tool calls as child spans (see openclaw_trace.py).
                from sia.openclaw_trace import replay_session

                replay_session(session_id, env, span)
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
            "openclaw.session_key": f"agent:main:explicit:{session_id}",
        })
        text = _extract_text(payload)
        span.set_outputs({"text": text})
        return text


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    return dict(re.findall(r'^(\w+):\s*"?(.*?)"?\s*$', m.group(1), re.M)) if m else {}


def skills_index(files: dict[str, str]) -> str:
    """The skill list OpenClaw puts in its system prompt; the direct runtime mimics it."""
    rows = [f"- {fm.get('name', path.split('/')[1])}: {fm.get('description', '')} (read {path})"
            for path, text in files.items()
            if path.startswith("skills/") and path.endswith("/SKILL.md") and (fm := _frontmatter(text)) is not None]
    return ("\n\n## Skills\nBefore answering, read any skill whose description matches the question.\n"
            + "\n".join(rows)) if rows else ""


_DIRECT_TOOLS = [
    llm.tool("list_files", "List the files in your workspace.", {"type": "object", "properties": {}}),
    llm.tool("read_file", "Read a file from your workspace, e.g. knowledge/billing.md.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
]


_CLAUDE_TOOLS = [
    {"name": "list_files", "description": "List the files in your workspace.", "strict": True,
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "read_file", "description": "Read a file from your workspace, e.g. knowledge/billing.md.", "strict": True,
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
]


def run_direct_claude(question: str, workspace: Path, model: str) -> str:
    """The direct runtime on Claude: a manual tool-use loop with the Anthropic SDK."""
    files = workspace_files(workspace)
    system = files["AGENTS.md"] + skills_index(files)
    client = llm.claude_client()
    messages = [{"role": "user", "content": question}]

    @mlflow.trace(span_type=SpanType.TOOL)
    def list_files() -> str:
        return "\n".join(p for p in files if p != "AGENTS.md")

    @mlflow.trace(span_type=SpanType.TOOL)
    def read_file(path: str) -> str:
        return files.get(path.strip().removeprefix("./"), f"No such file: {path}")

    for _ in range(8):
        response = client.messages.create(
            model=model, max_tokens=16000, system=system, tools=_CLAUDE_TOOLS, messages=messages,
            output_config={"effort": config.AGENT_EFFORT},
            cache_control={"type": "ephemeral"},     # AGENTS.md + tools are a stable prefix
        )
        if response.stop_reason == "refusal":
            return "I'm sorry, I can't help with that request."
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")
        messages.append({"role": "assistant", "content": response.content})   # unchanged: keeps thinking valid
        results = []
        for block in response.content:
            if block.type == "tool_use":
                out = list_files() if block.name == "list_files" else read_file(block.input.get("path", ""))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})
    return "Sorry, I couldn't finish looking that up."


def run_direct(question: str, workspace: Path) -> str:
    """A minimal tool-use loop over the workspace files: the Anthropic SDK for Claude, LiteLLM for
    every other provider."""
    provider, model = llm.split(config.AGENT_MODEL)
    if provider == "anthropic":
        return run_direct_claude(question, workspace, model)
    files = workspace_files(workspace)
    system = files["AGENTS.md"] + skills_index(files)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

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
    - skills/*/SKILL.md that point at knowledge/<topic>.md count like AGENTS.md instructions
    """
    agents_md = (workspace / "AGENTS.md").read_text()
    skills_text = "\n".join(p.read_text() for p in workspace.glob("skills/*/SKILL.md"))
    case = case_for_question(question)
    reads_topic = case is not None and (
        f"{case.topic}.md" in agents_md + skills_text
        or re.search(r"every file in .?knowledge/", agents_md, re.I))
    if reads_topic:
        # Leave the same evidence a real runtime would: a TOOL span per file read.
        with mlflow.start_span("read", span_type=SpanType.TOOL) as tool:
            tool.set_inputs({"path": f"knowledge/{case.topic}.md"})
            tool.set_outputs({"result": f"(contents of knowledge/{case.topic}.md)"})
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


def run_hermes(question: str, workspace: Path) -> str:
    from sia.hermes_runtime import run_hermes as _run  # optional extra: imported only when used

    return _run(question, workspace)


BACKENDS = {"openclaw": run_openclaw, "hermes": run_hermes, "direct": run_direct, "offline": run_offline}


# --------------------------------------------------------------------------------------------
# The traced entry point
# --------------------------------------------------------------------------------------------
@mlflow.trace(name="acme_support_agent", span_type=SpanType.AGENT)
def run_agent(question: str, prompt_ref: "str | int | AgentSpec" = "production", backend: str | None = None,
              source: str = "eval") -> str:
    """Answer one customer question with the given agent (AGENTS.md version, or an AgentSpec with
    skills). Produces one trace.

    source="production" marks real (simulated) customer traffic. Everything else -- evaluation,
    optimization, CI -- is "eval", so it never leaks back in as production signal.
    """
    backend = backend or config.BACKEND
    prompt, skills = resolve(prompt_ref)         # loading inside the trace links prompts <-> trace
    workspace = workspace_for(prompt, skills)
    if mlflow.get_current_active_span() is not None:  # evaluate() probes predict_fn untraced once
        mlflow.update_current_trace(tags={
            "topic": route_topic(question),
            "backend": backend,
            "prompt_version": str(prompt.version),
            "skills": ",".join(f"{n}@{p.version}" for n, p in sorted(skills.items())),
            "source": source,
        })
    return BACKENDS[backend](question, workspace)


def predict_fn_for(prompt_ref: "str | int | AgentSpec", backend: str | None = None):
    """A predict_fn for mlflow.genai.evaluate() pinned to one agent version."""

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


def register_skill(name: str, text: str, commit_message: str, author: str, level: int, **tags) -> int:
    """Register a new version of a skill (full SKILL.md text) *without* promoting it."""
    pv = mlflow.genai.register_prompt(
        name=f"{config.SKILL_PREFIX}{name}", template=text, commit_message=commit_message,
        tags={"kind": "skill", "author": author, "level": str(level), **{k: str(v) for k, v in tags.items()}},
    )
    return int(pv.version)
