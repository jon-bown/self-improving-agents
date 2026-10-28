"""Log human feedback and an expectation

From Self-Improvement & MLflow, Part 1:
https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/
"""

from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.log_feedback(
    trace_id="tr-7f3a9c",
    name="quality",
    value=False,
    rationale="Quoted last year's refund policy; the window changed in July.",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="maria"
    ),
)
mlflow.log_expectation(
    trace_id="tr-7f3a9c",
    name="expected_response",
    value="Refunds are available within 14 days of purchase.",
)
