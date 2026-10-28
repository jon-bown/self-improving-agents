"""SI_AGENT=litellm: a no-code agent, for trying the harness before wiring up your own.

The instructions are the system prompt of SI_AGENT_MODEL (any LiteLLM provider). If the
workspace holds reference files (SI_WORKSPACE_DIR), the model gets two tools, list_files and
read_file, each traced as a TOOL span.
"""

from __future__ import annotations

import json

import mlflow
from mlflow.entities import SpanType

from selfimprove import config, models
from selfimprove.agents import instructions_filename

_TOOLS = [
    models.tool("list_files", "List the reference files you can read.", {"type": "object", "properties": {}}),
    models.tool("read_file", "Read one of your reference files by its path.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
]


def workspace_files(workspace) -> dict[str, str]:
    skip = instructions_filename()
    return {str(p.relative_to(workspace)): p.read_text(errors="replace")
            for p in sorted(workspace.rglob("*")) if p.is_file() and p.name != skip and ".git" not in p.parts}


def call_litellm(question, instructions, workspace, max_turns: int = 6):
    files = workspace_files(workspace)

    @mlflow.trace(span_type=SpanType.TOOL)
    def list_files() -> str:
        return "\n".join(files)

    @mlflow.trace(span_type=SpanType.TOOL)
    def read_file(path: str) -> str:
        return files.get(path.strip().removeprefix("./"), f"No such file: {path}")

    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": question}]
    for _ in range(max_turns):
        msg = models.chat(config.settings.agent_model, messages, tools=_TOOLS if files else None, max_tokens=2000)
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            out = list_files() if call.function.name == "list_files" else read_file(args.get("path", ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return "Sorry, I couldn't finish looking that up."
