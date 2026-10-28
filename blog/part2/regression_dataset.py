"""Grow the regression dataset

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: experiment_id is your experiment's id.
"""

from mlflow.genai.datasets import create_dataset

dataset = create_dataset(
    name="support_regressions", experiment_id=[experiment_id]
)

failing = mlflow.search_traces(
    filter_string="feedback.quality = 'Fail' "
                  "AND expectation.expected_response IS NOT NULL",
    return_type="list",
)
dataset = dataset.merge_records(failing)

dataset = dataset.merge_records([
    {"inputs": {"question": "How long can I wait to request a refund?"},
     "expectations": {"expected_response": "Within 14 days of purchase."}},
    {"inputs": {"question": "I can't log in after resetting my password."},
     "expectations": {"expected_response": "Clear the browser cache and retry the "
                                           "reset link within 15 minutes."}},
])
