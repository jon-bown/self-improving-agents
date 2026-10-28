"""Level 2 · Measure, then promote (a person decides).

Runs production and the candidate against the same dataset and the same scorers, plus the
holdout set, and applies the promotion rules in sia/gate.py. Every evaluate() call is an
MLflow run, nested under one "gate" run, so the before/after sit side by side in the UI.

Three things to check before trusting the numbers:
  * the rows that failed before now pass            (the change works)
  * the holdout rows still pass                     (the change generalizes)
  * read at least one candidate answer by eye       (a green scorer != a good answer)

    python levels/level2_manual_loop/03_measure_and_promote.py [--candidate N] [--promote]
"""

import argparse

from sia import data, gate, tracking
from sia.agent import latest_version

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=int, default=None, help="prompt version (default: latest)")
parser.add_argument("--promote", action="store_true", help="move the production alias if the gate passes")
args = parser.parse_args()

experiment_id = tracking.setup()
candidate = args.candidate or latest_version()
dataset = data.get_or_create_dataset(experiment_id)

decision = gate.compare(candidate, "production", train=data.rows(dataset), holdout=data.holdout_rows())
print("\n" + decision.table())
print(f"\nGate: {'PASS' if decision.passed else 'FAIL'}", *(f"\n  - {r}" for r in decision.reasons))

if decision.passed and args.promote:
    gate.promote(candidate, reason=f"Level 2 manual fix; gate run {decision.run_id}")
    print(f"\nproduction -> v{candidate}. Rollback: python -c 'from sia import gate; gate.rollback()'")
elif decision.passed:
    print("\nRe-run with --promote to move the production alias (you are the approver at this level).")
print("Also: make test  -- the same dataset as a CI gate (see .github/workflows).")
