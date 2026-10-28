"""6. Automate the Approval

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: get_dataset is from mlflow.genai.datasets; run_agent is your agent;
grounded_in_policy comes from grounded_in_policy.py.
"""

# tests/regression/test_support_agent.py
import mlflow
from mlflow.genai.scorers import Correctness

@mlflow.test
def test_support_regressions():
    dataset = get_dataset(name="support_regressions")
    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=run_agent,
        scorers=[grounded_in_policy, Correctness()],
    )
    assert result.passed, result.reason
