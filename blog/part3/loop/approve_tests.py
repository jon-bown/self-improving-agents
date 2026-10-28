"""6. Automate the Approval: approve proposed tests

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: proposed_tests, dataset, CRITERIA, TESTS_PER_THEME_CAP and the check
helpers are your own.
"""

def approve(test):
    return all([
        fails_on_source_trace(test),                      # our helper: reproduces the failure
        not_near_duplicate(test, dataset),                # our helper
        count_in_theme(dataset, test.theme) < TESTS_PER_THEME_CAP,   # our helper
        reviewer_agent_approves(test, CRITERIA),          # our helper: not the proposing agent
    ])

approved = [t for t in proposed_tests if approve(t)]
flagged = [t for t in proposed_tests if t not in approved]

dataset = dataset.merge_records(approved)
dataset = dataset.set_tags({"holdout_fraction": "0.2"})
send_to_human_queue(flagged)                              # our helper
