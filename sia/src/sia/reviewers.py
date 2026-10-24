"""Simulated human reviewers.

A live workshop can label traces by hand in the MLflow UI. For a repeatable demo we also ship
three simulated reviewers. Each one knows the answer key in scenario/cases.yaml, but they grade
against slightly different rubrics -- which is exactly the "criteria drift" and "reviewer
disagreement" problem you hit once more than one person labels traces.

    maria  - checks facts only.
    dev    - checks facts, and fails answers that open with a disclaimer.
    sam    - checks facts, and fails answers longer than ~120 words.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from sia.scenario import Case

CAVEAT_OPENERS = re.compile(
    r"(policies (can|may) vary|terms of service|double[- ]check|please note|"
    r"it'?s important to|great question|thanks for reaching out|i understand)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Verdict:
    reviewer: str
    passed: bool
    rationale: str


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+|\n", text.strip(), maxsplit=1)
    return parts[0] if parts else ""


def fact_problems(case: Case, response: str) -> list[str]:
    problems = []
    for pattern in case.must_not_include:
        if m := re.search(pattern, response):
            problems.append(f"Stated something wrong or outdated ('{m.group(0)}').")
    for pattern in case.must_include:
        if not re.search(pattern, response, re.IGNORECASE):
            problems.append(f"Missing a key fact (expected something matching /{pattern}/).")
    return problems


def review(reviewer: str, case: Case, response: str) -> Verdict:
    problems = fact_problems(case, response)
    if reviewer == "dev" and CAVEAT_OPENERS.search(first_sentence(response)):
        problems.append("Buries the answer: opens with a disclaimer instead of the answer.")
    if reviewer == "sam" and len(response.split()) > 120:
        problems.append(f"Too long ({len(response.split())} words) for a simple question.")
    if problems:
        return Verdict(reviewer, False, " ".join(problems) + f" Correct answer: {case.expected_response}")
    return Verdict(reviewer, True, "Correct and to the point.")


REVIEWERS = ("maria", "dev", "sam")


def assign_reviewer(case: Case) -> str:
    """Spread cases across reviewers deterministically, like a review queue would."""
    return REVIEWERS[zlib.crc32(case.id.encode()) % len(REVIEWERS)]


def log_review(trace_id: str, case: Case, response: str) -> Verdict:
    """Label one trace the way a reviewer would: a verdict, plus the answer key if they have it.

    feedback     quality            True/False with a rationale
    expectation  expected_response  the correct answer, in words
    expectation  key_facts          patterns a correct answer must contain (a team "answer key")
    """
    import mlflow

    from sia import tracking

    reviewer = assign_reviewer(case)
    v = review(reviewer, case, response)
    src = tracking.human(reviewer)
    mlflow.log_feedback(trace_id=trace_id, name="quality", value=v.passed, rationale=v.rationale, source=src)
    mlflow.log_expectation(trace_id=trace_id, name="expected_response", value=case.expected_response, source=src)
    mlflow.log_expectation(trace_id=trace_id, name="key_facts", value=list(case.must_include), source=src)
    return v
