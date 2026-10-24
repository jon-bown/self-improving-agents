"""Level 4 · Gate the optimizer's candidate. Promotion is still a human decision.

Same rules as every level (sia/gate.py): must-pass scorers at 100%, better than production on
the dataset, not worse on the holdout. For optimizer output, also read the diff: optimizers
sometimes "win" by adding text that pleases the judge rather than the customer.

    python levels/level4_optimizer/03_gate.py [--candidate N] [--promote]
"""

import argparse
import difflib

from mlflow.genai.scorers import get_scorer

from sia import config, data, gate, signals, tracking
from sia.agent import latest_version, load_agents_md

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=int)
parser.add_argument("--promote", action="store_true")
args = parser.parse_args()

experiment_id = tracking.setup()
candidate = args.candidate or latest_version()
prod = load_agents_md("production")
print("".join(difflib.unified_diff(prod.template.splitlines(True), load_agents_md(candidate).template.splitlines(True),
                                   f"v{prod.version} (production)", f"v{candidate} (candidate)")))

quality = None
if config.HAS_LLM:
    try:
        quality = get_scorer(name="quality")
    except Exception:
        quality = signals.quality_judge()

decision = gate.compare(candidate, "production", train=data.rows(data.get_or_create_dataset(experiment_id)),
                        holdout=data.holdout_rows(), scorers=signals.all_scorers(quality))
print("\n" + decision.table())
print(f"\nGate: {'PASS' if decision.passed else 'FAIL'}", *(f"\n  - {r}" for r in decision.reasons))
if decision.passed and args.promote:
    gate.promote(candidate, reason=f"Level 4 optimizer candidate, approved by a human; gate {decision.run_id}")
    print(f"production -> v{candidate}")
