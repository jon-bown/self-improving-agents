"""Signal: every automated verdict we can compute about a response.

Three kinds, from cheapest/most trustworthy to most flexible:

    code scorers   deterministic checks (regex, length). Cheap, exact, but narrow.
    guidelines     LLM judges with a pass/fail requirement written in plain English.
    quality judge  a make_judge() judge named "quality" -- the same name the human reviewers
                   use -- so it can be *aligned* to their labels in Level 4.
"""

from __future__ import annotations

import re

from mlflow.entities import Feedback, SpanType
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


# ---- Tool-level signal: *how* the agent got its answer ----------------------------------------
# These read the TOOL spans in the trace: the direct runtime's read_file calls, and OpenClaw's own
# tool calls replayed from its session trajectory (sia/openclaw_trace.py).
def knowledge_reads(trace) -> list[str]:
    reads = []
    for span in sorted(trace.data.spans, key=lambda sp: sp.start_time_ns):  # in the order they happened
        if span.span_type == SpanType.TOOL and isinstance(span.inputs, dict):
            path = str(span.inputs.get("path") or span.inputs.get("file_path") or "")
            if "knowledge/" in path:
                reads.append(path[path.index("knowledge/"):])
    return reads


@scorer
def read_right_policy(trace) -> Feedback:
    """Did the agent read the policy file for the question's topic before answering?"""
    topic = trace.info.tags.get("topic")
    if not topic:
        return Feedback(value=True, rationale="No topic tag on this trace.")
    expected, reads = f"knowledge/{topic}.md", knowledge_reads(trace)
    if expected in reads:
        return Feedback(value=True, rationale=f"Read {expected}.")
    if not reads:
        return Feedback(value=False, rationale=f"Answered without reading any policy file (expected {expected}).")
    return Feedback(value=False, rationale=f"Read {', '.join(dict.fromkeys(reads))} but not {expected}.")


@scorer
def focused_reading(trace) -> Feedback:
    """Shotgun reading (every policy file, every time) is slow and usually means the agent
    doesn't know where to look. One or two files is enough for any question in this scenario."""
    reads = list(dict.fromkeys(knowledge_reads(trace)))
    return Feedback(value=len(reads) <= 2, rationale=f"Read {len(reads)} policy file(s): {', '.join(reads) or 'none'}.")


def tool_scorers():
    """Deterministic tool checks, plus MLflow's LLM-based ToolCallCorrectness when SIA_TOOL_JUDGE=1."""
    import os

    out = [read_right_policy, focused_reading]
    if config.HAS_LLM and os.getenv("SIA_TOOL_JUDGE") == "1":
        from mlflow.genai.scorers import ToolCallCorrectness

        out.append(ToolCallCorrectness())
    return out


def code_scorers():
    """Reference-free deterministic checks: safe to run on any production trace."""
    return [no_stale_policy, no_disclaimer_opener, concise, read_right_policy, focused_reading]


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
    llm_tool = [s for s in tool_scorers() if s not in code_scorers()]
    return code_scorers() + llm_tool + judge_scorers() + reference_scorers() + theme_scorers() + extra
