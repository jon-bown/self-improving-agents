"""Store: turn labeled traces into a curated, versioned evaluation dataset."""

from __future__ import annotations

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

from sia import config, tracking
from sia.scenario import load_cases

# Boolean feedback is stored as text, so filters compare against 'true' / 'false'.
FAILED = "feedback.quality = 'false'"
PASSED = "feedback.quality = 'true'"
HAS_EXPECTATION = "expectation.expected_response IS NOT NULL"


def get_or_create_dataset(experiment_id: str):
    try:
        return get_dataset(name=config.DATASET_NAME)
    except Exception:
        return create_dataset(name=config.DATASET_NAME, experiment_id=[experiment_id],
                              tags={"purpose": "regression"})


def _human_labeled(value: bool, extra_filter: str | None):
    # The search filter matches *any* "quality" feedback, including the LLM judge's (same name on
    # purpose, for alignment). Keep only traces whose latest HUMAN label has this value.
    f = f"{PASSED if value else FAILED} AND {HAS_EXPECTATION}" + (f" AND {extra_filter}" if extra_filter else "")
    return [t for t in tracking.agent_traces(f)
            if (fb := tracking.human_feedback(t)) is not None and bool(fb.value) is value]


def labeled_failures(extra_filter: str | None = None):
    return _human_labeled(False, extra_filter)


def labeled_passes(extra_filter: str | None = None):
    return _human_labeled(True, extra_filter)


def trace_to_record(trace, source: str) -> dict:
    """A dataset row: the question as inputs, the reviewer's answer as the expectation."""
    return {
        "inputs": {"question": tracking.question_of(trace)},
        "expectations": {k: v for k in ("expected_response", "key_facts")
                         if (v := tracking.expectation(trace, k)) is not None},
        "tags": {"source": source, "source_trace": trace.info.trace_id,
                 "topic": trace.info.tags.get("topic", "")},
    }


def dedupe(records: list[dict]) -> list[dict]:
    """Forty thumbs-down on one question are one test, not forty."""
    seen, out = set(), []
    for r in records:
        q = r["inputs"]["question"].strip().lower()
        if q not in seen:
            seen.add(q)
            out.append(r)
    return out


def rows(dataset) -> list[dict]:
    """Dataset -> list of {inputs, expectations} dicts that evaluate() and optimize_prompts() accept."""
    df = dataset.to_df()
    return [{"inputs": r["inputs"], "expectations": r["expectations"]} for _, r in df.iterrows()]


def holdout_rows() -> list[dict]:
    """Hand-written golden cases that no optimizer ever sees. Checks that a fix generalizes."""
    return [
        {"inputs": {"question": c.question},
         "expectations": {"expected_response": c.expected_response, "key_facts": list(c.must_include)}}
        for c in load_cases("holdout")
    ]


def growth_report(dataset) -> str:
    df = dataset.to_df()
    by_source = df["tags"].map(lambda t: (t or {}).get("source", "?")).value_counts().to_dict()
    return f"{len(df)} records {by_source}"


def flush():
    mlflow.flush_trace_async_logging()
