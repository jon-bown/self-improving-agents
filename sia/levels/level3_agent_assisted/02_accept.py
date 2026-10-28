"""Level 3 · Accept a reviewed proposal: register the candidate, grow the verifiers, run the gate.

Accepting a proposal does three things, and only the last one can ship anything:
  1. registers the proposed AGENTS.md as a new prompt version (tagged author=analyst),
  2. registers each non-noise theme criterion as a Guidelines judge in MLflow's scorer registry
     (theme_<slug>), so every future gate checks it -- the test suite grows with the evidence,
  3. adds the representative failing traces to the regression dataset and runs the gate.
Promotion still needs --promote: at this level a person presses the button.

    python levels/level3_agent_assisted/02_accept.py proposals/<ts> [--promote]
"""

import argparse
import json
from pathlib import Path

import mlflow

from sia import config, data, gate, signals, tracking
from sia.agent import register_candidate

parser = argparse.ArgumentParser()
parser.add_argument("proposal_dir")
parser.add_argument("--promote", action="store_true")
args = parser.parse_args()

experiment_id = tracking.setup()
d = Path(args.proposal_dir)
themes = [t for t in json.loads((d / "themes.json").read_text()) if not t["is_reviewer_noise"]]

version = register_candidate(
    (d / "AGENTS.md").read_text(),
    commit_message=f"Analyst proposal {d.name}: " + "; ".join(t["slug"] for t in themes),
    author="analyst", level=3, proposal=d.name,
)
print(f"Registered candidate v{version}")

if config.HAS_LLM:
    for t in themes:
        signals.register_theme(t["slug"], t["criterion"])
        print(f"  + scorer theme_{t['slug']}: {t['criterion']}")

dataset = data.get_or_create_dataset(experiment_id)
records = [data.trace_to_record(trace, f"theme:{t['slug']}")
           for t in themes for i in t["trace_ids"] if (trace := mlflow.get_trace(i)) is not None]
records = [r for r in data.dedupe(records) if r["expectations"].get("expected_response")]
if records:
    dataset.merge_records(records)
print(f"Dataset: {data.growth_report(dataset)}")

decision = gate.compare(version, "production", train=data.rows(dataset), holdout=data.holdout_rows())
print("\n" + decision.table())
print(f"\nGate: {'PASS' if decision.passed else 'FAIL'}", *(f"\n  - {r}" for r in decision.reasons))
if decision.passed and args.promote:
    gate.promote(version, reason=f"Level 3 analyst proposal {d.name}, approved by a human; gate {decision.run_id}")
    print(f"production -> v{version}")
