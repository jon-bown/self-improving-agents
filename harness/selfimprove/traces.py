"""MLflow setup and the trace queries every workflow reuses.

Assessment conventions (Part 1, section 2.2):
    feedback     "quality"            bool + rationale. HUMAN from reviewers, LLM_JUDGE from the
                                      quality judge (same name on purpose, so they can be compared
                                      and the judge aligned), CODE from checks.
    expectation  "expected_response"  the right answer, from a person or a golden case. A judge
                                      verdict never becomes an expectation (Part 3, section 3).
Boolean feedback is stored as text: filter with feedback.quality = 'false', not = False.
"""

from __future__ import annotations

import json
import time

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Trace

from selfimprove import config

QUALITY = "quality"
EXPECTED = "expected_response"
FAILED = f"feedback.{QUALITY} = 'false'"
PASSED = f"feedback.{QUALITY} = 'true'"
HAS_EXPECTATION = f"expectation.{EXPECTED} IS NOT NULL"


def setup() -> str:
    """Point MLflow at the store and experiment from .env. Returns the experiment id."""
    s = config.settings
    mlflow.set_tracking_uri(s.tracking_uri)
    if s.tracking_uri.startswith("sqlite:") and mlflow.get_experiment_by_name(s.experiment) is None:
        # keep a local store's artifacts next to it, not in ./mlruns of whatever folder we run from
        mlflow.create_experiment(s.experiment, artifact_location=(s.state_dir / "artifacts").resolve().as_uri())
    exp = mlflow.set_experiment(s.experiment)
    if s.agent == "litellm" or s.agent_target.startswith("examples.support_agent"):
        mlflow.litellm.autolog()  # LLM spans under the AGENT span for the built-in agents
    return exp.experiment_id


def human(reviewer: str) -> AssessmentSource:
    return AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=reviewer)


def now_ms() -> int:
    return int(time.time() * 1000)


def hours_ago_ms(hours: float) -> int:
    return int((time.time() - hours * 3600) * 1000)


def search(filter_string: str | None = None, max_results: int = 1000) -> list[Trace]:
    mlflow.flush_trace_async_logging()  # traces this process just wrote must be visible
    return mlflow.search_traces(filter_string=filter_string, max_results=max_results, return_type="list",
                                order_by=["timestamp_ms DESC"])


def production(extra: str | None = None, since_ms: int | None = None, max_results: int = 1000) -> list[Trace]:
    """Production traffic only. Evaluation, optimizer and CI traces are excluded.

    Traces the harness created carry tags.si_source = 'production'. Traces your agent logs on its
    own are selected with SI_PRODUCTION_FILTER instead.
    """
    base = config.settings.production_filter or "tags.si_source = 'production'"
    parts = [base] + ([extra] if extra else []) + ([f"trace.timestamp_ms > {since_ms}"] if since_ms else [])
    traces = search(" AND ".join(parts), max_results)
    return [t for t in traces if "mlflow.sourceRun" not in t.info.trace_metadata]


def _request(trace: Trace) -> dict:
    req = trace.data.request
    try:
        req = json.loads(req) if isinstance(req, str) else req
    except json.JSONDecodeError:
        return {config.settings.input_key: req}
    return req if isinstance(req, dict) else {config.settings.input_key: req}


def question_of(trace: Trace) -> str:
    req = _request(trace)
    key = config.settings.input_key
    if key in req:
        return str(req[key])
    return str(next(iter(req.values()), "")) if req else ""


def response_of(trace: Trace) -> str:
    out = trace.data.response
    if isinstance(out, str):
        try:
            parsed = json.loads(out)
            return parsed if isinstance(parsed, str) else out
        except json.JSONDecodeError:
            return out
    return "" if out is None else str(out)


def is_fail(value) -> bool:
    return str(value).strip().lower() in ("false", "no", "fail", "0", "0.0")


def _valid(a) -> bool:
    return a.valid is not False


def feedback(trace: Trace, name: str = QUALITY, source_type: str | None = None):
    """Latest valid feedback with this name (optionally from one source type), or None."""
    fb = [a for a in trace.info.assessments
          if a.name == name and a.feedback is not None and _valid(a)
          and (source_type is None or a.source.source_type == source_type)]
    return fb[-1] if fb else None


def human_feedback(trace: Trace, name: str = QUALITY):
    return feedback(trace, name, AssessmentSourceType.HUMAN)


def human_labels(trace: Trace, name: str = QUALITY) -> dict[str, bool]:
    """reviewer -> their latest verdict. More than one distinct value is a contradiction."""
    out = {}
    for a in trace.info.assessments:
        if a.name == name and a.feedback is not None and _valid(a) \
                and a.source.source_type == AssessmentSourceType.HUMAN:
            out[a.source.source_id or "?"] = not is_fail(a.value)
    return out


def expectation(trace: Trace, name: str = EXPECTED):
    ex = [a for a in trace.info.assessments if a.name == name and a.expectation is not None and _valid(a)]
    return ex[-1].expectation.value if ex else None


def expectations(trace: Trace) -> dict:
    out = {}
    for a in trace.info.assessments:
        if a.expectation is not None and _valid(a):
            out[a.name] = a.expectation.value
    return out


def failed(trace: Trace, must_pass: tuple[str, ...] = ()) -> bool:
    """A trace fails if its human label, the quality judge, or a must-pass check says so.

    A human label overrides the judge: if a person passed it, the judge's FAIL does not count.
    """
    human_fb = human_feedback(trace)
    if human_fb is not None:
        if is_fail(human_fb.value):
            return True
    elif (judge := feedback(trace, QUALITY)) is not None and is_fail(judge.value):
        return True
    return any(is_fail(a.value) for a in trace.info.assessments
               if a.feedback is not None and _valid(a) and a.name in must_pass)


def has_verdict(trace: Trace, must_pass: tuple[str, ...] = ()) -> bool:
    return any(a.feedback is not None and (a.name == QUALITY or a.name in must_pass)
               for a in trace.info.assessments)


def unscored(trace: Trace) -> bool:
    return not any(a.source.source_type in (AssessmentSourceType.CODE, AssessmentSourceType.LLM_JUDGE)
                   for a in trace.info.assessments)


def failing_verdicts(trace: Trace) -> list[dict]:
    return [{"name": a.name, "source": f"{a.source.source_type}:{a.source.source_id}",
             "rationale": (a.rationale or "")[:400]}
            for a in trace.info.assessments
            if a.feedback is not None and _valid(a) and is_fail(a.value)]


def tool_calls(trace: Trace) -> list[dict]:
    from mlflow.entities import SpanType

    return [{"name": s.name, "inputs": s.inputs, "status": str(s.status.status_code)}
            for s in trace.search_spans(span_type=SpanType.TOOL)]
