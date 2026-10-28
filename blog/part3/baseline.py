"""1. Track the Baseline

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: get_dataset is from mlflow.genai.datasets; load_checkpoint is in
checkpoints.py; tests_per_theme, human_edits_per_test and traces_per_theme are your own
helpers.
"""

import mlflow

dataset = get_dataset(name="support_regressions")
last_run_ms = load_checkpoint("triage")                  # our helper, defined in Section 2
new_traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {last_run_ms}",
    return_type="list")

with mlflow.start_run(run_name="loop-baseline"):
    mlflow.log_metrics({
        "tests_per_theme": tests_per_theme(dataset),           # our helper
        "human_edits_per_test": human_edits_per_test(dataset), # our helper
        "traces_per_theme": traces_per_theme(new_traces),      # our helper
        "incoming_traces": len(new_traces),
    })
