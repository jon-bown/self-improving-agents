"""2. Schedule the Trace Ingestion: tag the dataset

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: dataset, records and triage_run_id come from the triage run.
"""

dataset = dataset.merge_records(records)
dataset = dataset.set_tags({
    "generated_by": "triage-agent",
    "source_run": triage_run_id,
    "tests_per_theme_cap": "3",
})
