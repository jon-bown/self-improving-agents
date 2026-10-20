"""Level 4 · Let an optimizer write the candidate; the gate judges it; a person promotes.

mlflow.genai.optimize_prompts() with the GEPA optimizer proposes AGENTS.md variants, runs the
agent on the regression dataset with each one, scores them with *our* scorers (code checks,
guideline judges, theme judges from Level 3, and the aligned quality judge from 01), reflects on
the failures, and keeps the best. The winner is registered as a new prompt version.

Then the same gate as Levels 2-3 runs, including the holdout set the optimizer never saw.
That is the check against overfitting and against gaming a judge.

Use SIA_BACKEND=direct here: GEPA calls the agent dozens of times, and the direct backend
runs the same AGENTS.md + knowledge/ in seconds instead of spawning OpenClaw each time.
The gate can then re-check the winner on OpenClaw:  SIA_BACKEND=openclaw ... 03_gate.py

    SIA_BACKEND=direct python levels/level4_optimizer/02_optimize.py [--budget N]
"""

import argparse

from sia import config, data, improvers, signals, tracking
from sia.agent import load_agents_md, register_candidate

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=config.GEPA_BUDGET,
                    help="GEPA max_metric_calls (default 16 on Ollama, 60 on hosted providers)")
args = parser.parse_args()

experiment_id = tracking.setup()
train = data.rows(data.get_or_create_dataset(experiment_id))
prod = load_agents_md("production")
print(f"Optimizing AGENTS.md v{prod.version} on {len(train)} dataset rows (backend: {config.BACKEND})")

if not config.HAS_LLM:
    # Offline rehearsal: stand in for GEPA with the rule-based analyst so the loop still closes.
    report = improvers.failure_report(tracking.agent_traces(f"tags.prompt_version = '{prod.version}'"))
    text = improvers.offline_analyze(report, prod.template)["agents_md"]
    version = register_candidate(text, "offline stand-in for GEPA", author="offline-optimizer", level=4)
else:
    from mlflow.genai.scorers import get_scorer

    try:
        quality = get_scorer(name="quality")            # the aligned judge from 01
    except Exception:
        quality = signals.quality_judge()
    scorers = signals.code_scorers() + signals.judge_scorers() + signals.theme_scorers() + [quality]
    result = improvers.gepa_optimize(train, scorers, max_metric_calls=args.budget)
    best = result.optimized_prompts[0]
    print(f"GEPA score: {result.initial_eval_score} -> {result.final_eval_score}")
    version = register_candidate(best.template, f"GEPA from v{prod.version}: "
                                 f"{result.initial_eval_score:.2f} -> {result.final_eval_score:.2f}",
                                 author="gepa", level=4)

print(f"Registered candidate v{version}. Next: 03_gate.py --candidate {version}")
