"""Run the dataset as tests

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: run_agent comes from support_agent.py; the scorers from Part 1's
scorers.py.
"""

import mlflow
import pytest
from mlflow.genai.datasets import get_dataset

dataset = get_dataset(name="support_regressions")


@mlflow.test
@pytest.mark.parametrize(
    "row", dataset.df.to_dict("records"),
    ids=lambda r: r["inputs"]["question"][:40],
)
def test_support_agent_regressions(row):
    result = mlflow.genai.evaluate(
        predict_fn=run_agent,
        data=[row],
        scorers=[cites_current_policy, leads_with_answer, Correctness()],
    )
    assert result.passed, result.reason
