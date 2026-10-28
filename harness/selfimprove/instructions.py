"""The update target: your agent's instructions, versioned in the MLflow Prompt Registry.

Whatever text defines your agent's behavior (a system prompt, AGENTS.md, a SKILL.md) is
registered as a prompt. The agent always runs a specific version, loaded by alias:

    prompts:/my-agent-instructions@production   what live traffic uses
    prompts:/my-agent-instructions@previous     one alias move away (rollback)
    prompts:/my-agent-instructions/7            a candidate under test

Promotion is an alias move, not a code change (Part 2, section 4).
"""

from __future__ import annotations

import difflib
import time

import mlflow

from selfimprove import config


def uri(ref: str | int = "production") -> str:
    name = config.settings.prompt_name
    return f"prompts:/{name}/{ref}" if str(ref).isdigit() else f"prompts:/{name}@{ref}"


def load(ref: str | int = "production"):
    return mlflow.genai.load_prompt(uri(ref))


def exists() -> bool:
    try:
        load("production")
        return True
    except Exception:
        return False


def seed(text: str | None = None, commit_message: str = "v1: initial instructions") -> int:
    """Register the seed instructions as v1 and point production at it (idempotent)."""
    if exists():
        return int(load("production").version)
    if text is None:
        path = config.settings.instructions_file
        if path is None or not path.is_file():
            raise FileNotFoundError(f"SI_INSTRUCTIONS_FILE not found: {path}")
        text = path.read_text()
    pv = mlflow.genai.register_prompt(name=config.settings.prompt_name, template=text,
                                      commit_message=commit_message, tags={"author": "seed"})
    mlflow.genai.set_prompt_alias(config.settings.prompt_name, alias="production", version=pv.version)
    return int(pv.version)


def register_candidate(text: str, commit_message: str, author: str, **tags) -> int:
    """Register a new version *without* promoting it. Returns the version number."""
    pv = mlflow.genai.register_prompt(
        name=config.settings.prompt_name, template=text, commit_message=commit_message[:1000],
        tags={"author": author, "status": "candidate", **{k: str(v)[:500] for k, v in tags.items()}},
    )
    return int(pv.version)


def set_tags(version: int, **tags) -> None:
    for k, v in tags.items():
        mlflow.genai.set_prompt_version_tag(config.settings.prompt_name, version, k, str(v)[:500])


def promote(version: int, reason: str, by: str = "human", **tags) -> int:
    """Move production to `version`; the version it replaces becomes `previous`. Returns that."""
    name = config.settings.prompt_name
    previous = int(load("production").version)
    mlflow.genai.set_prompt_alias(name, alias="previous", version=previous)
    mlflow.genai.set_prompt_alias(name, alias="production", version=version)
    set_tags(version, status="production", promoted_by=by, promoted_because=reason,
             promoted_at=int(time.time() * 1000), replaced_version=previous, **tags)
    set_tags(previous, status="retired")
    return previous


def rollback(reason: str = "manual rollback") -> int:
    """Point production back at the version it replaced. Returns the restored version."""
    current = load("production")
    restore = int(current.tags.get("replaced_version") or load("previous").version)
    mlflow.genai.set_prompt_alias(config.settings.prompt_name, alias="production", version=restore)
    set_tags(int(current.version), status="rolled_back", rolled_back_because=reason)
    set_tags(restore, status="production")
    return restore


def latest_version() -> int:
    versions = mlflow.MlflowClient().search_prompt_versions(config.settings.prompt_name)
    return max(int(v.version) for v in versions)


def diff(a: str | int, b: str | int) -> str:
    old, new = load(a), load(b)
    return "".join(difflib.unified_diff(
        old.template.splitlines(keepends=True), new.template.splitlines(keepends=True),
        fromfile=f"v{old.version}", tofile=f"v{new.version}"))
