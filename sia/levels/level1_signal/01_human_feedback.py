"""Level 1 · Signal from people: reviewers label traces with feedback and expectations.

Two ways to run it:

  default   three simulated reviewers (maria, dev, sam) label every unlabeled trace. They
            share an answer key but grade with slightly different rubrics -- on purpose.
  --live    create a label schema and a review queue so real people can label in the MLflow UI
            (Traces -> Review Queues). Run 02/03 afterwards on whatever they labeled.

Every label is an MLflow *assessment* on the trace:
    feedback     name="quality"            value=True/False + rationale, source=HUMAN:<reviewer>
    expectation  name="expected_response"  the answer the reviewer says is correct
    expectation  name="key_facts"          patterns a correct answer must contain

    python levels/level1_signal/01_human_feedback.py [--live]
"""

import argparse

from sia import tracking
from sia.reviewers import log_review
from sia.scenario import case_for_question

parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true", help="create a review queue for real reviewers instead")
parser.add_argument("--reviewers", default="", help="comma-separated usernames for --live")
args = parser.parse_args()

experiment_id = tracking.setup()
unlabeled = [t for t in tracking.agent_traces() if tracking.human_feedback(t) is None]
print(f"{len(unlabeled)} unlabeled traces")

if args.live:
    from mlflow.genai.label_schemas import InputPassFail, InputText, create_label_schema
    from mlflow.genai.review_queues import add_items_to_review_queue, create_review_queue

    quality = create_label_schema(
        "quality", type="feedback", input=InputPassFail(), enable_comment=True, overwrite=True,
        instruction="Would you send this answer to the customer as-is? Explain in the comment.",
    )
    expected = create_label_schema(
        "expected_response", type="expectation", input=InputText(), overwrite=True,
        instruction="If the answer is wrong, write the correct answer.",
    )
    queue = create_review_queue(
        "acme-support-triage", queue_type="custom",
        users=[u for u in args.reviewers.split(",") if u] or None,
        schema_ids=[quality.schema_id, expected.schema_id],
    )
    add_items_to_review_queue(queue.queue_id, item_ids=[t.info.trace_id for t in unlabeled])
    print(f"Review queue '{queue.name}' has {len(unlabeled)} traces. Label them in the MLflow UI.")
    raise SystemExit

counts = {}
for trace in unlabeled:
    question, answer = tracking.question_of(trace), tracking.response_of(trace)
    case = case_for_question(question)
    if case is None:
        continue
    verdict = log_review(trace.info.trace_id, case, answer)
    reviewer = verdict.reviewer
    counts[reviewer] = counts.get(reviewer, [0, 0])
    counts[reviewer][0 if verdict.passed else 1] += 1
    mark = "PASS" if verdict.passed else "FAIL"
    print(f"  {reviewer:<6} {mark}  {question[:60]:<60}  {'' if verdict.passed else verdict.rationale[:70]}")

print("\nreviewer  pass  fail")
for r, (p, f) in sorted(counts.items()):
    print(f"{r:<9} {p:>4}  {f:>4}")
print("\nIn the UI: filter traces with  feedback.quality = 'false'")
