"""SI_AGENT=command: your agent is a CLI.

    SI_AGENT_COMMAND=my-agent --system-file {instructions_file} --message {input}

Placeholders: {input} (the question), {instructions_file} (a file holding the instructions
version under test), {workspace} (the rendered workspace directory, also the working
directory). The command is split like a shell would split it, then each argument is filled in,
so a question is always one argument and is never interpreted by a shell.

The answer is stdout. If stdout is JSON, the first of "output", "text", "final", "result" is used.
"""

from __future__ import annotations

import json
import shlex
import subprocess

import mlflow
from mlflow.entities import SpanType

from selfimprove import config
from selfimprove.agents import instructions_filename


def render(template: str, question: str, workspace) -> list[str]:
    values = {"input": question, "instructions_file": str(workspace / instructions_filename()),
              "workspace": str(workspace)}
    return [arg.format_map(values) for arg in shlex.split(template)]


def call_command(question, instructions, workspace):
    s = config.settings
    if not s.agent_command:
        raise ValueError("SI_AGENT=command needs SI_AGENT_COMMAND")
    argv = render(s.agent_command, question, workspace)
    with mlflow.start_span("agent.command", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"argv": argv[:1] + ["..."], "workspace": str(workspace)})
        proc = subprocess.run(argv, cwd=workspace, capture_output=True, text=True, timeout=s.agent_timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{argv[0]} exited {proc.returncode}: {proc.stderr[-1500:]}")
        out = proc.stdout.strip()
        try:
            payload = json.loads(out)
            if isinstance(payload, dict):
                out = next((payload[k] for k in ("output", "text", "final", "result")
                            if isinstance(payload.get(k), str)), out)
        except json.JSONDecodeError:
            pass
        span.set_outputs({"text": out})
        return out
