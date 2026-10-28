"""7. Improve Other Components: the improvement loop

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: skill_loaded and dataset_created come from skill_judges.py; SKILL_DIR,
MAX_ROUNDS, SKILL_CONFIGS, CLAUDE_EXPERIMENT_ID and the helpers are your own.
"""

candidate = copy_skill(SKILL_DIR)                           # our helper

for _ in range(MAX_ROUNDS):
    round_start = int(time.time() * 1000)
    for config in SKILL_CONFIGS:                            # our helper: one prompt and project per case
        run_claude(config, skills=candidate)                # our helper: runs claude -p with tracing on, in a fresh evaluation experiment tagged on the trace

    traces = mlflow.search_traces(
        experiment_ids=[CLAUDE_EXPERIMENT_ID],
        filter_string=f"attributes.timestamp_ms > {round_start}",
        return_type="list")
    result = mlflow.genai.evaluate(
        data=traces, scorers=[skill_loaded, dataset_created])

    if result.passed:
        open_pull_request(candidate)                        # our helper
        break
    ask_claude(f"These judges failed: {result.reason}. Fix SKILL.md.",
               cwd=candidate)                               # our helper
