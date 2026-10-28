"""Human signal: the ground truth the rest of the loop is checked against.

For a personal agent the best verifier is the owner's own judgment, so labeling has to be cheap:

    label()          one call per verdict (from code, a notebook, a chat UI's thumbs button)
    label_terminal() go through unlabeled production traces in the terminal: p / f / s / q
    review_queue()   label schemas + an MLflow review queue, for labeling in the MLflow UI

A label is a `quality` feedback (pass/fail + rationale) and, for a failure, an
`expected_response` expectation. Feedback says a trace is worth testing; the expectation says
what correct means. A trace with both is one call away from being a dataset record.
"""

from __future__ import annotations

import getpass
import os

import mlflow

from selfimprove import traces


def reviewer_name() -> str:
    return os.getenv("SI_REVIEWER") or getpass.getuser()


def label(trace_id: str, passed: bool, rationale: str = "", expected_response: str | None = None,
          reviewer: str | None = None, **expectations) -> None:
    source = traces.human(reviewer or reviewer_name())
    mlflow.log_feedback(trace_id=trace_id, name=traces.QUALITY, value=bool(passed),
                        rationale=rationale or None, source=source)
    if expected_response:
        mlflow.log_expectation(trace_id=trace_id, name=traces.EXPECTED, value=expected_response, source=source)
    for name, value in expectations.items():
        mlflow.log_expectation(trace_id=trace_id, name=name, value=value, source=source)


def unlabeled(limit: int = 50):
    return [t for t in traces.production(max_results=limit * 4) if traces.human_feedback(t) is None][:limit]


def label_terminal(limit: int = 20) -> int:
    todo = unlabeled(limit)
    print(f"{len(todo)} unlabeled production traces. p = pass, f = fail, s = skip, q = quit\n")
    done = 0
    for t in todo:
        judge = traces.feedback(t, traces.QUALITY)
        print("-" * 100)
        print(f"trace {t.info.trace_id}   instructions v{t.info.tags.get('si_prompt_version', '?')}")
        print(f"\nQ: {traces.question_of(t)}\n\nA: {traces.response_of(t)}\n")
        if judge is not None:
            print(f"(judge says {'PASS' if not traces.is_fail(judge.value) else 'FAIL'}: {(judge.rationale or '')[:200]})\n")
        choice = input("[p/f/s/q] > ").strip().lower()
        if choice == "q":
            break
        if choice not in ("p", "f"):
            continue
        rationale = input("why? > ").strip()
        expected = input("correct answer (enter to skip) > ").strip() if choice == "f" else ""
        label(t.info.trace_id, passed=choice == "p", rationale=rationale, expected_response=expected or None)
        done += 1
    return done


def review_queue(name: str = "selfimprove-triage", users: list[str] | None = None, limit: int = 100) -> str:
    from mlflow.genai.label_schemas import InputPassFail, InputText, create_label_schema
    from mlflow.genai.review_queues import add_items_to_review_queue, create_review_queue

    quality = create_label_schema(
        traces.QUALITY, type="feedback", input=InputPassFail(), enable_comment=True, overwrite=True,
        instruction="Would you accept this response as-is? Say why in the comment.")
    expected = create_label_schema(
        traces.EXPECTED, type="expectation", input=InputText(), overwrite=True,
        instruction="If the response is wrong, write the correct one.")
    queue = create_review_queue(name, queue_type="custom", users=users or None,
                                schema_ids=[quality.schema_id, expected.schema_id])
    todo = unlabeled(limit)
    if todo:
        add_items_to_review_queue(queue.queue_id, item_ids=[t.info.trace_id for t in todo])
    return f"review queue '{queue.name}' has {len(todo)} traces; label them in the MLflow UI (Traces -> Review Queues)"
