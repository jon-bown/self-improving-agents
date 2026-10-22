"""SI_AGENT=python: call a function you wrote.

    SI_AGENT_TARGET=my_package.agent:answer

The function takes the question and the instructions text, and optionally the rendered
workspace directory:

    def answer(question: str, instructions: str) -> str: ...
    def answer(question: str, instructions: str, workspace: Path) -> str: ...   # gets the folder too

Wrap any SDK this way (OpenAI Agents, LangGraph, Hermes' AIAgent, the openclaw-sdk client, the
Claude Agent SDK). Instrument it with MLflow autolog or @mlflow.trace if you like; those spans
nest under the harness's AGENT span.
"""

from __future__ import annotations

import importlib
import inspect
import sys

from selfimprove import config


def load_target(target: str):
    if ":" not in target:
        raise ValueError(f"SI_AGENT_TARGET={target!r}: use 'module.path:function'")
    module_name, attr = target.split(":", 1)
    root = str(config.ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    fn = getattr(importlib.import_module(module_name), attr)
    params = inspect.signature(fn).parameters
    wants_workspace = "workspace" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

    def call(question, instructions, workspace):
        out = fn(question, instructions, workspace=workspace) if wants_workspace else fn(question, instructions)
        return out if isinstance(out, str) else str(out)

    call.__name__ = attr
    return call
