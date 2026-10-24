"""Signal: every automated verdict we can compute about a response.

Three kinds, from cheapest/most trustworthy to most flexible:

    code scorers   deterministic checks (regex, length). Cheap, exact, but narrow.
    guidelines     LLM judges with a pass/fail requirement written in plain English.
    quality judge  a make_judge() judge named "quality" -- the same name the human reviewers
                   use -- so it can be *aligned* to their labels in Level 4.
"""

from __future__ import annotations

import re

from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from sia import config

# Facts that used to be true. Any answer containing one is quoting stale policy.
STALE_FACTS = {
    r"30[- ]day": "refund window is 14 days since July 2026, not 30",
    r"24 hours? (link|valid)|valid for 24 hours": "password reset links expire after 15 minutes",
}


@scorer
def no_stale_policy(outputs) -> Feedback:
    text = str(outputs)
    for pattern, why in STALE_FACTS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return Feedback(value=False, rationale=f"Quotes outdated policy: {why}.")
    return Feedback(value=True, rationale="No known-stale facts.")


@scorer
def no_disclaimer_opener(outputs) -> Feedback:
    """A cheap code proxy for the leads_with_answer judge."""
    first = re.split(r"(?<=[.!?])\s+|\n", str(outputs).strip(), maxsplit=1)[0]
    if re.search(r"policies (can|may) vary|terms of service|double[- ]check", first, re.IGNORECASE):
        return Feedback(value=False, rationale=f"Opens with a disclaimer: '{first[:80]}'")
    return Feedback(value=True, rationale="Opens with content, not a disclaimer.")


@scorer
def has_key_facts(outputs, expectations) -> Feedback:
    """Reference check against the reviewers' answer key (expectation "key_facts")."""
    facts = (expectations or {}).get("key_facts") or []
    missing = [f for f in facts if not re.search(f, str(outputs), re.IGNORECASE)]
    if missing:
        return Feedback(value=False, rationale=f"Missing key facts: {missing}")
    return Feedback(value=True, rationale="All key facts present." if facts else "No answer key for this row.")


@scorer
def concise(outputs) -> Feedback:
    n = len(str(outputs).split())
    return Feedback(value=n <= 120, rationale=f"{n} words (limit 120).")


leads_with_answer = Guidelines(
    name="leads_with_answer",
    guidelines=(
        "The first sentence of the response must directly answer the customer's question. "
        "FAIL if it opens with a disclaimer, a reminder to check the Terms of Service, "
        "a pleasantry, or restating the question."
    ),
)

no_guessing = Guidelines(
    name="no_guessing",
    guidelines=(
        "The response must not present a guess as fact. If the agent is unsure it must say so "
        "or offer to escalate. FAIL if it confidently states plan limits, timelines, or feature "
        "availability with hedges like 'should', 'usually', or 'I believe'."
    ),
)

QUALITY_INSTRUCTIONS = """\
You are reviewing a customer-support answer for Acme Cloud.

Question: {{ inputs }}
Answer: {{ outputs }}

Decide whether a careful support lead would approve this answer as-is.
Return true if it is correct, current, and gets to the point; otherwise false.
"""


def quality_judge(model: str | None = None):
    """An LLM judge whose name matches the human feedback name, so it can be aligned later."""
    return make_judge(
        name="quality",
        instructions=QUALITY_INSTRUCTIONS,
        feedback_value_type=bool,
        model=model or config.JUDGE_MODEL,
    )


def code_scorers():
    """Reference-free deterministic checks: safe to run on any production trace."""
    return [no_stale_policy, no_disclaimer_opener, concise]


def judge_scorers():
    return [leads_with_answer, no_guessing] if config.HAS_LLM else []


def reference_scorers():
    """Scorers that need expectations (only usable on a curated dataset)."""
    return [has_key_facts] + ([Correctness()] if config.HAS_LLM else [])


def theme_scorers():
    """Guidelines judges generated from failure themes (Level 3+), stored in MLflow's scorer registry.

    The verifier set grows with the evidence: each accepted theme adds a registered judge named
    theme_<slug>, and every later gate evaluates against it.
    """
    if not config.HAS_LLM:
        return []
    from mlflow.genai.scorers import list_scorers

    try:
        return [s for s in list_scorers() if s.name.startswith("theme_")]
    except Exception:
        return []


def register_theme(slug: str, criterion: str):
    judge = Guidelines(name=f"theme_{slug}", guidelines=criterion)
    return judge.register(name=f"theme_{slug}")


def all_scorers(quality=None):
    extra = [quality] if (quality is not None and config.HAS_LLM) else []
    return code_scorers() + judge_scorers() + reference_scorers() + theme_scorers() + extra
