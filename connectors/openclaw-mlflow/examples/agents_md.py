"""Keep an OpenClaw agent's AGENTS.md in the MLflow Prompt Registry, and run whatever @production says.

    python agents_md.py push AGENTS.md -m "v1: what the agent runs today"   # new version -> @production
    python agents_md.py pull ~/odsc-live/workspace                          # @production -> workspace/AGENTS.md
    python agents_md.py use 1                                               # @production -> v1 (a rollback)

`push` registers the file as a new prompt version and points the `production` alias at it. `pull`
writes the version `production` points at into the workspace OpenClaw reads, so the next
`openclaw agent` call runs it. Edit a version in the MLflow UI, move the alias there, then `pull`:
that's an update. Moving the alias back and pulling again is a rollback.

Set MLFLOW_TRACKING_URI first. Every command selects `--experiment` (default openclaw-demo), so the
prompt is linked to it and listed in that experiment's Prompts tab, next to the agent's traces.
Needs `mlflow` (the harness venv has it).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)  # keep the terminal output to one line


def push(path: Path, name: str, message: str) -> None:
    pv = mlflow.genai.register_prompt(name=name, template=path.read_text(), commit_message=message)
    mlflow.genai.set_prompt_alias(name, alias="production", version=pv.version)
    print(f"registered {name} v{pv.version}; @production -> v{pv.version}")


def pull(workspace: Path, name: str) -> None:
    pv = mlflow.genai.load_prompt(f"prompts:/{name}@production")
    (workspace / "AGENTS.md").write_text(pv.template)
    print(f"{workspace / 'AGENTS.md'} <- {name} v{pv.version} (@production): {pv.commit_message}")


def use(version: int, name: str) -> None:
    mlflow.genai.set_prompt_alias(name, alias="production", version=version)
    pv = mlflow.genai.load_prompt(f"prompts:/{name}@production")
    print(f"@production -> {name} v{pv.version}: {pv.commit_message}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="openclaw-agents-md", help="prompt name in the registry")
    ap.add_argument("--experiment", default="openclaw-demo",
                    help="experiment to link the prompt to (where the agent's traces are)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="register a file as a new version and point @production at it")
    p.add_argument("file", type=Path)
    p.add_argument("-m", "--message", default="new version")
    q = sub.add_parser("pull", help="write @production into a workspace's AGENTS.md")
    q.add_argument("workspace", type=Path)
    u = sub.add_parser("use", help="point @production at a version (moving it back is a rollback)")
    u.add_argument("version", type=int)
    args = ap.parse_args()
    # register_prompt and load_prompt link the prompt to the active experiment
    # (tag _mlflow_experiment_ids); without one it lands in the Default experiment.
    mlflow.set_experiment(args.experiment)
    if args.cmd == "push":
        push(args.file, args.name, args.message)
    elif args.cmd == "use":
        use(args.version, args.name)
    else:
        pull(args.workspace, args.name)


if __name__ == "__main__":
    main()
