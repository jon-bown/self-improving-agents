"""2. Schedule the Trace Ingestion: checkpoints

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: EXPERIMENT_ID is your experiment's id.
"""

from mlflow import MlflowClient

client = MlflowClient()

def load_checkpoint(name, experiment_id=EXPERIMENT_ID):
    tags = client.get_experiment(experiment_id).tags
    return int(tags.get(f"checkpoint.{name}", 0))

def save_checkpoint(name, timestamp_ms, experiment_id=EXPERIMENT_ID):
    client.set_experiment_tag(experiment_id, f"checkpoint.{name}", str(timestamp_ms))
