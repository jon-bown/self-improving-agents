"""Signal: the verifier. Every scorer comes from SI_CRITERIA_FILE or MLflow's scorer registry.

From cheapest and most trustworthy to most flexible (Part 1, section 2.3):

    checks      code scorers from regexes, word limits and tool-span rules. Free, exact, narrow.
    guidelines  Guidelines judges: a requirement in plain English becomes a pass/fail judge.
    quality     a make_judge() judge named "quality", the same name reviewers use, so its
                agreement with them can be measured and it can be aligned (Part 3, section 5).
    reference   scorers that need expectations; they only run on the curated dataset.
    themes      Guidelines judges the triage step writes from failure themes, registered in
                MLflow's scorer registry as theme_<slug>. The verifier grows with the evidence.
    grounded    span checks on a tool call, then a judge on whether the answer stays within
                what the tool returned (Part 3, section 3).

With SI_FAST_JUDGE_MODEL (e.g. typesafe:/jev-latest) the quality and grounded judges become a
cascade: the decision model answers first, with a probability and no rationale; verdicts whose
pass probability falls inside SI_FAST_JUDGE_BAND are re-judged by SI_JUDGE_MODEL, which writes
the rationale. A decision model's verdict is never an expectation.

Mark a scorer `must_pass: true` and no candidate that fails it on any row can ship.
"""

from __future__ import annotations

import importlib
import json
import re
from functools import lru_cache

import yaml
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, SpanStatusCode, SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from selfimprove import config

DEFAULT_QUALITY = """\
You are reviewing one response from an AI agent.

Request: {{ inputs }}
Response: {{ outputs }}

Would the person this agent works for accept this response as-is? Return true if it is correct,
current, and does what was asked; otherwise false. Explain your reasoning briefly."""


@lru_cache(maxsize=1)
def criteria() -> dict:
    path = config.settings.criteria_file
    if path is None or not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+|\n", text.strip(), maxsplit=1)[0]


def failed_call(span) -> bool:
    """A tool call that errored, including tools that report errors in their output.

    OpenClaw's `read` returns {"error": ...} for a directory and a result with details.kind
    "not_found" for a missing file, with an OK span status.
    """
    if span.status.status_code == SpanStatusCode.ERROR:
        return True
    out = span.outputs
    if not isinstance(out, dict):
        return False
    if out.get("error"):
        return True
    details = out.get("result", {}).get("details") if isinstance(out.get("result"), dict) else None
    return isinstance(details, dict) and str(details.get("kind") or details.get("status")).lower() in ("not_found", "error")


def make_check(spec: dict):
    """A code scorer from a criteria.yaml `checks:` entry.

        must_match / must_not_match   regexes over the response (a list means all / any)
        first_sentence_not_match      regexes the opening sentence must avoid
        max_words                     word limit
        requires_tool                 a TOOL span with this name must exist and succeed
          tool_input_match            ...and one of them must have inputs matching this regex

    Tool spans come from the trace and from any trace the agent logged itself for the same call
    (the OpenClaw MLflow plugin's, found by session key; see traces.linked).
    """
    name = spec["name"]
    why = spec.get("rationale", "")

    if "requires_tool" in spec:
        tool, input_match = spec["requires_tool"], spec.get("tool_input_match")

        @scorer(name=name)
        def tool_check(trace) -> Feedback:
            from selfimprove import traces

            calls = [s for s in traces.spans(trace, span_type=SpanType.TOOL) if tool in ("*", s.name)]
            if input_match:
                calls = [s for s in calls if re.search(input_match, json.dumps(s.inputs, default=str), re.I)]
            if not calls:
                target = f"{tool} on /{input_match}/" if input_match else tool
                return Feedback(value=False, rationale=f"The agent answered without calling {target}. {why}".strip())
            if all(failed_call(c) for c in calls):
                return Feedback(value=False, rationale=f"Every {tool} call failed. {why}".strip())
            return Feedback(value=True, rationale=f"Called {', '.join(sorted({c.name for c in calls}))}.")

        return tool_check

    def as_list(v):
        return [v] if isinstance(v, str) else list(v or [])

    must, must_not = as_list(spec.get("must_match")), as_list(spec.get("must_not_match"))
    opener_not, max_words = as_list(spec.get("first_sentence_not_match")), spec.get("max_words")

    @scorer(name=name)
    def check(outputs) -> Feedback:
        text = str(outputs)
        for p in must_not:
            if re.search(p, text, re.IGNORECASE):
                return Feedback(value=False, rationale=f"Matches /{p}/. {why}".strip())
        for p in must:
            if not re.search(p, text, re.IGNORECASE):
                return Feedback(value=False, rationale=f"Missing /{p}/. {why}".strip())
        opener = _first_sentence(text)
        for p in opener_not:
            if re.search(p, opener, re.IGNORECASE):
                return Feedback(value=False, rationale=f"Opens with '{opener[:80]}'. {why}".strip())
        if max_words and len(text.split()) > max_words:
            return Feedback(value=False, rationale=f"{len(text.split())} words (limit {max_words}).")
        return Feedback(value=True, rationale="Check passed.")

    return check


@scorer(name="has_key_facts")
def key_facts(outputs, expectations) -> Feedback:
    """Reference check: every regex in the expectation `key_facts` appears in the response.

    Not `expected_facts`: MLflow's Correctness judge reads that name too, and refuses rows that
    carry it alongside `expected_response`. And the scorer is not named `key_facts`: evaluate()'s
    result_df gives a feedback the same name as an expectation a NaN value, so the gate would
    silently skip every row that has facts to check. (criteria.yaml still lists it as key_facts.)
    """
    facts = (expectations or {}).get("key_facts") or []
    missing = [f for f in facts if not re.search(f, str(outputs), re.IGNORECASE)]
    if missing:
        return Feedback(value=False, rationale=f"Missing expected facts: {missing}")
    return Feedback(value=True, rationale="All expected facts present." if facts else "No expected facts on this row.")


def checks():
    return [make_check(c) for c in criteria().get("checks", [])]


def guidelines():
    if not config.settings.has_llm:
        return []
    return [Guidelines(name=g["name"], guidelines=g["guidelines"]) for g in criteria().get("guidelines", [])]


def quality_judge(registered: bool = True):
    """The alignable judge. Uses the registered (possibly aligned) version when there is one."""
    if not config.settings.has_llm:
        return None
    if registered:
        from mlflow.genai.scorers import get_scorer

        try:
            return get_scorer(name="quality")
        except Exception:
            pass
    spec = criteria().get("quality") or {}
    return make_judge(name="quality", instructions=spec.get("instructions", DEFAULT_QUALITY),
                      feedback_value_type=bool, model=config.settings.judge_model)


# --------------------------------------------------------------------------------------------
# First-pass decision model (Jev) and the cascade to the text judge
# --------------------------------------------------------------------------------------------
PROBABILITY = "typesafe.probability"
FIRST_PASS = "quality_first_pass"   # the decision model's raw verdict when the server samples it


def fast_judge(name: str, instructions: str):
    """A judge on SI_FAST_JUDGE_MODEL: a yes/no with a probability, no rationale, no {{ trace }}."""
    return make_judge(name=name, instructions=instructions, feedback_value_type=bool,
                      model=config.settings.fast_judge_model)


def probability(fb) -> float | None:
    """The decision model's probability that the answer passes, from the feedback metadata."""
    raw = (getattr(fb, "metadata", None) or {}).get(PROBABILITY)
    return None if raw is None else float(json.loads(raw) if isinstance(raw, str) else raw)


def uncertain(p: float | None) -> bool:
    lo, hi = config.settings.fast_judge_band
    return p is None or lo <= p <= hi


def cascade(fast, text):
    """Ask `fast` first; re-ask `text` when fast's pass probability is in the uncertain band.

    Both judges are callables taking inputs= and outputs=. The result is an LLM_JUDGE feedback
    with the fast model's probability in its metadata either way, and si.escalated saying
    which judge decided.
    """
    def judge(inputs, outputs) -> Feedback:
        first = fast(inputs=inputs, outputs=outputs)
        p = probability(first)
        meta = {k: str(v) for k, v in (first.metadata or {}).items()}
        if not uncertain(p):
            return Feedback(value=first.value, metadata={**meta, "si.escalated": "false"},
                            rationale=f"{config.settings.fast_judge_model} p(pass)={p:.2f}, outside the "
                                      f"uncertain band; not re-judged.",
                            source=AssessmentSource(AssessmentSourceType.LLM_JUDGE, config.settings.fast_judge_model))
        second = text(inputs=inputs, outputs=outputs)
        why = f"[re-judged: {config.settings.fast_judge_model} p(pass)={p if p is None else round(p, 2)}] "
        return Feedback(value=second.value, rationale=why + (second.rationale or ""),
                        metadata={**meta, "si.escalated": "true"},
                        source=AssessmentSource(AssessmentSourceType.LLM_JUDGE, config.settings.judge_model))
    return judge


def quality_scorer(text=None):
    """The `quality` verdict for scoring: the text judge, or the Jev-first cascade when configured.

    Alignment always works on the text judge (the decision model has no rationale to align).
    """
    text = text if text is not None else quality_judge()
    if text is None or not config.settings.has_fast_judge:
        return text
    spec = criteria().get("quality") or {}
    fast = fast_judge("quality", spec.get("fast_instructions") or spec.get("instructions", DEFAULT_QUALITY))
    run = cascade(fast, text)

    @scorer(name="quality")
    def quality(inputs, outputs) -> Feedback:
        return run(inputs, outputs)

    return quality


def make_grounded(spec: dict):
    """Span checks on one tool, then a judge on whether the answer stays within its output.

        - name: grounded_in_policy
          tool: lookup_policy          # the TOOL span to check
          requires_input: topic        # it must have been called with this argument
          tool_input_match: knowledge/ # optional: only calls whose inputs match this regex
          instructions: "Does {{ outputs }} make only claims that the policy in {{ inputs }} supports?"

    The span checks cost nothing and catch a missing, failed, or empty lookup before any model
    is asked. The judge sees the question and the tool's output as its inputs.
    """
    name, tool, needs = spec["name"], spec["tool"], spec.get("requires_input")
    instructions = spec["instructions"]
    text = make_judge(name=name, instructions=instructions, feedback_value_type=bool,
                      model=config.settings.judge_model) if config.settings.has_llm else None
    judge = cascade(fast_judge(name, spec.get("fast_instructions", instructions)), text) \
        if (text is not None and config.settings.has_fast_judge) else text

    input_match = spec.get("tool_input_match")

    @scorer(name=name)
    def grounded(trace) -> Feedback:
        from selfimprove import traces

        calls = traces.spans(trace, span_type=SpanType.TOOL, name=tool)
        if input_match:
            calls = [c for c in calls if re.search(input_match, json.dumps(c.inputs, default=str), re.I)]
        if not calls:
            return Feedback(value=False, rationale=f"The agent answered without calling {tool}.")
        ok = [c for c in calls if not failed_call(c)]
        if not ok:
            return Feedback(value=False, rationale=f"{tool} failed, so the answer has nothing behind it.")
        ok = [c for c in ok if c.outputs and (not needs or (c.inputs or {}).get(needs))]
        if not ok:
            return Feedback(value=False, rationale=f"{tool} was called without {needs or 'input'} or returned nothing.")
        if judge is None:
            return Feedback(value=True, rationale=f"{tool} returned content (no judge model: span checks only).")
        # Every successful call: an agent that reads several files may answer from any of them.
        looked_up = "\n\n".join(str(c.outputs) for c in ok)
        return judge(inputs={"question": trace.data.request, tool: looked_up}, outputs=trace.data.response)

    return grounded


def grounded():
    return [make_grounded(g) for g in criteria().get("grounded", [])]


def reference():
    names = criteria().get("reference", ["key_facts", "correctness"])
    out = []
    if "key_facts" in names:
        out.append(key_facts)
    if "correctness" in names and config.settings.has_llm:
        out.append(Correctness())
    return out


def themes():
    """theme_<slug> judges from MLflow's scorer registry (written by triage, kept by approve)."""
    if not config.settings.has_llm:
        return []
    from mlflow.genai.scorers import list_scorers

    try:
        return [s for s in list_scorers() if s.name.startswith("theme_")]
    except Exception:
        return []


def register_theme(slug: str, criterion: str):
    judge = Guidelines(name=f"theme_{slug}", guidelines=criterion)
    return judge.register(name=f"theme_{slug}")


def extra():
    """Your own scorers: criteria.yaml `python: my_pkg.scorers:SCORERS` (a list, or a function)."""
    target = criteria().get("python")
    if not target:
        return []
    module, attr = target.split(":", 1)
    obj = getattr(importlib.import_module(module), attr)
    return list(obj() if callable(obj) and not hasattr(obj, "name") else obj)


def must_pass() -> tuple[str, ...]:
    c = criteria()
    names = [x["name"] for group in ("checks", "guidelines", "grounded") for x in c.get(group, []) if x.get("must_pass")]
    return tuple(names + list(c.get("must_pass", [])))


def live(quality=None):
    """Reference-free scorers: safe on any production trace (no expectations needed)."""
    q = quality_scorer(quality)
    return checks() + grounded() + guidelines() + themes() + extra() + ([q] if q is not None else [])


def full(quality=None):
    """Everything, for the curated dataset: live scorers plus reference scorers."""
    return live(quality) + reference()


def reload():
    criteria.cache_clear()
