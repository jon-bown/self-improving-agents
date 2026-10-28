"""Level 2 · Store: curate labeled traces into an evaluation dataset.

Traces are raw signal; the dataset is the curated store. We take:
  * every trace a human FAILED that also has an expected_response  -> "this must get fixed"
  * a sample of traces a human PASSED                              -> "this must not break"
and merge them (deduplicated by question) into a versioned MLflow evaluation dataset.

The holdout cases in scenario/cases.yaml are deliberately *not* added: they are how we find
out later whether a change generalizes or just memorized the dataset.

    python levels/level2_manual_loop/01_build_dataset.py
"""

from sia import data, tracking

experiment_id = tracking.setup()
dataset = data.get_or_create_dataset(experiment_id)
before = len(dataset.to_df())

failures = data.labeled_failures()
passes = data.labeled_passes()
records = data.dedupe(
    [data.trace_to_record(t, "human_fail") for t in failures]
    + [data.trace_to_record(t, "human_pass") for t in passes[:10]]
)
existing = {r["question"].strip().lower() for r in dataset.to_df()["inputs"]}
new = [r for r in records if r["inputs"]["question"].strip().lower() not in existing]
if new:
    dataset.merge_records(new)

print(f"Found {len(failures)} human-failed and {len(passes)} human-passed traces with expectations")
print(f"Dataset '{dataset.name}': {before} -> {len(dataset.to_df())} records  ({data.growth_report(dataset)})")
print("\nRead the failing rationales before writing a fix. In this scenario they cluster into:")
print("  1. stale facts (30-day refunds, 24-hour reset links)")
print("  2. answers buried under a Terms-of-Service disclaimer")
print("  3. confident guesses for things not in the quick facts (SSO, Jira, 2FA)")
print("\nNext: 02_write_fix.py")
