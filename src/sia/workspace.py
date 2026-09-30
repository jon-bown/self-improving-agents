"""Materialize an OpenClaw workspace from a prompt version in the MLflow Prompt Registry.

OpenClaw's behavior is defined by plain files in its workspace: AGENTS.md (instructions),
skills/<name>/SKILL.md, and any reference files the agent can read. That makes the workspace
the natural *update target* for self-improvement. We keep AGENTS.md in the MLflow Prompt
Registry so every change is a numbered, diffable version with a commit message, and we render
a version to disk right before the agent runs:

    prompts:/acme-support-agents-md@production  ->  .workspaces/v3/AGENTS.md
                                                     .workspaces/v3/knowledge/*.md
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sia import config
from sia.scenario import KNOWLEDGE_DIR, WORKSPACE_V1_DIR


def build_workspace(agents_md: str, name: str, skills: dict[str, str] | None = None) -> Path:
    """Write a complete workspace directory and return its path.

    skills maps a skill name to the full text of its SKILL.md.
    """
    ws = config.WORKSPACES_DIR / name
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "knowledge").mkdir(parents=True)
    (ws / "AGENTS.md").write_text(agents_md)
    for f in KNOWLEDGE_DIR.glob("*.md"):
        shutil.copy(f, ws / "knowledge" / f.name)
    for skill_name, text in (skills or {}).items():
        (ws / "skills" / skill_name).mkdir(parents=True, exist_ok=True)
        (ws / "skills" / skill_name / "SKILL.md").write_text(text)
    return ws


def seed_agents_md() -> str:
    return (WORKSPACE_V1_DIR / "AGENTS.md").read_text()


def workspace_files(ws: Path) -> dict[str, str]:
    """Relative path -> text for every file the agent can see (used by the direct backend)."""
    return {
        str(p.relative_to(ws)): p.read_text()
        for p in sorted(ws.rglob("*"))
        if p.is_file() and ".openclaw" not in p.parts
    }
