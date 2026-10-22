"""Bring your own agent. The harness calls it through one traced entry point, `run()`.

    run("How long do refunds take?")                  production instructions, source="eval"
    run(q, ref=7, source="eval")                       a candidate, as evaluate()/the gate call it
    run(q, source="production")                        real traffic: the loop learns from these

Every call is one MLflow trace with an AGENT root span named SI_AGENT_NAME, tagged with the
instructions version that produced it (`si_prompt_version`) and where the call came from
(`si_source`). Evaluation and optimizer traffic is tagged "eval", so it never comes back as
production signal. If your agent is already instrumented (autolog, @mlflow.trace), its spans
nest under this root.

An adapter is any callable `(question, instructions, workspace) -> str`. SI_AGENT picks one of
the built-ins in selfimprove/agents/; `python` imports yours from SI_AGENT_TARGET.
"""

from __future__ import annotations

import hashlib
import shutil
from functools import lru_cache
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from selfimprove import config, instructions


def adapter():
    """The configured adapter: (question, instructions, workspace) -> str."""
    kind = config.settings.agent
    if kind == "python":
        from selfimprove.agents.python import load_target

        return load_target(config.settings.agent_target)
    if kind == "http":
        from selfimprove.agents.http import call_http

        return call_http
    if kind == "command":
        from selfimprove.agents.command import call_command

        return call_command
    if kind == "litellm":
        from selfimprove.agents.litellm_agent import call_litellm

        return call_litellm
    if kind == "openclaw":
        from selfimprove.agents.openclaw import call_openclaw

        return call_openclaw
    raise ValueError(f"SI_AGENT={kind!r}: choose python, http, command, litellm or openclaw")


def instructions_filename() -> str:
    if config.settings.agent == "openclaw":
        return "AGENTS.md"
    f = config.settings.instructions_file
    return f.name if f else "INSTRUCTIONS.md"


@lru_cache(maxsize=256)
def _render(text: str) -> Path:
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    ws = config.settings.state_dir / "workspaces" / digest
    if not (ws / instructions_filename()).is_file():
        if ws.exists():
            shutil.rmtree(ws)
        src = config.settings.workspace_dir
        if src and src.is_dir():
            shutil.copytree(src, ws / src.name)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / instructions_filename()).write_text(text)
    return ws


def workspace_for(text: str) -> Path:
    """A directory holding this exact instructions text plus the fixed SI_WORKSPACE_DIR files.

    Keyed by a hash of the text, because during optimize_prompts MLflow swaps each candidate's
    text into the loaded prompt: candidates never share a workspace.
    """
    return _render(text)


def run(question: str, ref: str | int = "production", source: str = "eval", tags: dict | None = None) -> str:
    """Answer one question with the given instructions version. Produces one trace."""
    s = config.settings
    with mlflow.start_span(s.agent_name, span_type=SpanType.AGENT) as span:
        span.set_inputs({s.input_key: question})
        prompt = instructions.load(ref)  # loaded inside the trace: links prompt version <-> trace
        if mlflow.get_current_active_span() is not None:  # evaluate() probes predict_fn untraced once
            mlflow.update_current_trace(tags={
                "si_source": source, "si_prompt_version": str(prompt.version), "si_agent": s.agent,
                **(tags or {}),
            })
        text = prompt.template
        answer = adapter()(question, text, workspace_for(text))
        span.set_outputs(answer)
        return answer


def predict_fn(ref: str | int = "production", source: str = "eval"):
    """A predict_fn for mlflow.genai.evaluate() / optimize_prompts(), pinned to one version."""
    key = config.settings.input_key

    def predict(**inputs) -> str:
        return run(str(inputs[key]), ref=ref, source=source)

    predict.__signature__ = _signature(key)
    return predict


def _signature(key: str):
    import inspect

    return inspect.Signature([inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY)])
