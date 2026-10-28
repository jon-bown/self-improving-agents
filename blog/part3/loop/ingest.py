"""2. Schedule the Trace Ingestion

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: load_checkpoint and save_checkpoint are in checkpoints.py;
TRIAGE_THRESHOLD and run_triage_agent are your own.
"""

last_run_ms = load_checkpoint("triage")   # checkpoint helper
new_failures = mlflow.search_traces(
    filter_string=(f"feedback.quality = 'Fail' "
                   f"AND expectation.expected_response IS NOT NULL "
                   f"AND attributes.timestamp_ms > {last_run_ms}"),
    return_type="list")

if len(new_failures) >= TRIAGE_THRESHOLD:
    dataset = get_dataset(name="support_regressions")
    dataset = dataset.merge_records(new_failures)
    save_checkpoint("triage", int(time.time() * 1000))   # our helper
    run_triage_agent(dataset_version=dataset.version, traces=new_failures)
