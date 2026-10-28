"""Load the Acme Cloud scenario: test cases, knowledge base, and workspace files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "scenario"
KNOWLEDGE_DIR = SCENARIO_DIR / "knowledge"
WORKSPACE_V1_DIR = SCENARIO_DIR / "workspace_v1"
TOPICS = ("billing", "account_access", "product_howto", "support_process")


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    topic: str
    question: str
    expected_response: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


def load_cases(split: str | None = None) -> list[Case]:
    raw = yaml.safe_load((SCENARIO_DIR / "cases.yaml").read_text())["cases"]
    cases = [Case(**c) for c in raw]
    return [c for c in cases if split is None or c.split == split]


def case_for_question(question: str) -> Case | None:
    return next((c for c in load_cases() if c.question == question), None)


def load_knowledge() -> dict[str, str]:
    """topic -> markdown text of the current (correct) policy."""
    return {p.stem: p.read_text() for p in sorted(KNOWLEDGE_DIR.glob("*.md"))}


def route_topic(question: str) -> str:
    """A deliberately simple keyword router, used to tag traces by topic."""
    q = question.lower()
    if any(w in q for w in ("discount", "pricing", "ship a", "roadmap", "real person", "contract", "human")):
        return "support_process"
    if any(w in q for w in ("refund", "invoice", "billing", "payment", "card", "dollars", "pay ")):
        return "billing"
    if any(w in q for w in ("password", "log in", "login", "locked", "2fa", "authenticator", "sso", "sign-on", "backup code", "phone")):
        return "account_access"
    return "product_howto"
