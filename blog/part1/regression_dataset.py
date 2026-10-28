"""Build a regression dataset from failing traces

From Self-Improvement & MLflow, Part 1:
https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/

Not defined here: experiment_id is your experiment's id.
"""

from mlflow.genai.datasets import create_dataset

dataset = create_dataset(
    name="billing_regressions", experiment_id=[experiment_id]
)

failing = mlflow.search_traces(
    filter_string="feedback.quality = 'Fail' "
                  "AND expectation.expected_response IS NOT NULL",
    return_type="list",
)
dataset = dataset.merge_records(failing)
