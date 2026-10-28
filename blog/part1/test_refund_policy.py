"""Write a regression test

From Self-Improvement & MLflow, Part 1:
https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/

Not defined here: leads_with_answer comes from scorers.py.
"""

import mlflow
from mlflow.genai.scorers import RegexMatch

from support_agent import run_agent     # the production, instrumented entry point


@mlflow.test
def test_should_state_current_refund_policy():
    result = mlflow.genai.evaluate(
        predict_fn=run_agent,
        data=[{"inputs": {"question": "How long can I wait to request a refund?"}}],
        scorers=[
            RegexMatch(name="mentions_window", pattern=r"14 days"),
            leads_with_answer,
        ],
    )
    assert result.passed, result.reason
