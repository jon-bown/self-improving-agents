"""MLflow setup and the handful of trace queries every level reuses."""

from __future__ import annotations

import time

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Trace

from sia import config


def setup(experiment: str | None = None) -> str:
    """Point MLflow at the tracking server and experiment. Returns the experiment id."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    exp = mlflow.set_experiment(experiment or config.EXPERIMENT)
    if config.BACKEND == "direct":
        mlflow.litellm.autolog()    # LLM spans (any provider via LiteLLM) under our AGENT span
        mlflow.anthropic.autolog()  # ... and for Claude, which goes through the Anthropic SDK
    return exp.experiment_id


def human(reviewer: str) -> AssessmentSource:
    return AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=reviewer)


def code(name: str) -> AssessmentSource:
    return AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=name)


def since_ms(hours: float) -> int:
    return int((time.time() - hours * 3600) * 1000)


def search(filter_string: str | None = None, max_results: int = 500) -> list[Trace]:
    return mlflow.search_traces(filter_string=filter_string, max_results=max_results, return_type="list")


def agent_traces(filter_string: str | None = None, max_results: int = 500) -> list[Trace]:
    """Production traffic only: excludes traces from evaluation, optimization, and CI."""
    base = "trace.name = 'acme_support_agent' AND tags.source = 'production'"
    traces = search(f"{base} AND {filter_string}" if filter_string else base, max_results)
    return [t for t in traces if "mlflow.sourceRun" not in t.info.trace_metadata]


def question_of(trace: Trace) -> str:
    return _inputs(trace).get("question", "")


def response_of(trace: Trace) -> str:
    out = trace.data.response
    return out if isinstance(out, str) else str(out)


def _inputs(trace: Trace) -> dict:
    import json

    req = trace.data.request
    return json.loads(req) if isinstance(req, str) else (req or {})


def human_feedback(trace: Trace, name: str = "quality"):
    """Latest human feedback with this name on a trace, or None."""
    fb = [
        a for a in trace.info.assessments
        if a.name == name and a.feedback is not None and a.source.source_type == AssessmentSourceType.HUMAN
        and a.valid is not False
    ]
    return fb[-1] if fb else None


def expectation(trace: Trace, name: str = "expected_response"):
    ex = [a for a in trace.info.assessments if a.name == name and a.expectation is not None]
    return ex[-1].expectation.value if ex else None
