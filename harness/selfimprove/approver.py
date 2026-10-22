"""Approve the tests (Part 3, section 6): explicit checks in place of a human reviewer.

A proposed test joins the dataset only if it
  1. reproduces the failure it came from (its source trace still fails a verifier),
  2. is not a near duplicate of an existing test,
  3. stays under the per-theme cap, and
  4. passes a reviewer agent that checks it against written criteria. The reviewer is a separate
     prompt from the analyst that proposed the test, and neither sees the holdout.
Anything that fails a check goes to the human queue instead of being discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow

from selfimprove import config, models, signals, store, traces

REVIEW_CRITERIA = """\
A good regression test for an AI agent:
- has an input a real user could plausibly send;
- has an expected response that is specific, checkable and consistent with the reviewers'
  rationales (not merely the agent's own output);
- tests one behavior, named by its theme;
- would not reward an answer that games the judge (e.g. by being longer or more hedged)."""

REVIEW_TOOL = models.tool("review_test", "Approve or reject the proposed test.", {
    "type": "object",
    "properties": {"approve": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["approve", "reason"],
})


@dataclass
class Verdict:
    record: dict
    approved: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def reproduces_failure(record: dict) -> tuple[bool, str]:
    trace_id = record["tags"].get("source_trace")
    if not trace_id:
        return True, "hand-written (no source trace)"
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        return False, f"source trace {trace_id} not found"
    if traces.failed(trace, signals.must_pass()):
        return True, "source trace fails"
    return False, "source trace no longer has a failing verdict"


def reviewer_approves(record: dict, criteria: str = REVIEW_CRITERIA) -> tuple[bool, str]:
    if not config.settings.has_llm:
        return False, "no LLM for the reviewer agent: a person must review"
    user = (f"Criteria:\n{criteria}\n\nProposed test:\n"
            f"input: {store.question_of(record)}\nexpectations: {record.get('expectations')}\n"
            f"theme: {record['tags'].get('theme')}\nwhy proposed: {record['tags'].get('reason', '')}")
    out = models.structured(config.settings.judge_model,
                            "You review proposed regression tests for an AI agent. Be strict.", user, REVIEW_TOOL,
                            max_tokens=2000)
    return bool(out.get("approve")), str(out.get("reason", ""))


def approve(proposed: list[dict], cap: int | None = None, use_reviewer: bool = True) -> list[Verdict]:
    cap = config.settings.tests_per_theme_cap if cap is None else cap
    existing = store.records()
    verdicts = []
    for rec in proposed:
        v = Verdict(rec, approved=True)
        ok, why = reproduces_failure(rec)
        v.checks["reproduces_failure"] = ok
        v.reasons.append(why)
        dup = store.near_duplicate(store.question_of(rec), existing)
        v.checks["not_near_duplicate"] = dup is None
        if dup:
            v.reasons.append(f"near duplicate of: {dup[:80]}")
        theme = rec["tags"].get("theme")
        under_cap = theme is None or store.count_in_theme(theme, existing) < cap
        v.checks["under_theme_cap"] = under_cap
        if not under_cap:
            v.reasons.append(f"theme '{theme}' already has {cap} tests")
        if use_reviewer and all(v.checks.values()):
            ok, why = reviewer_approves(rec)
            v.checks["reviewer_agent"] = ok
            v.reasons.append(f"reviewer: {why}")
        v.approved = all(v.checks.values())
        if v.approved:
            existing.append(rec)  # later proposals are checked against this one too
        verdicts.append(v)
    return verdicts
