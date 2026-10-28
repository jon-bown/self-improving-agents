"""Level 4 · Improve the verifier first: align the LLM judge to your reviewers.

An optimizer maximizes whatever its scorer rewards. Before handing the update step to an
optimizer, make the scorer agree with people. The "quality" judge shares its name with the
human feedback, so judge.align() can learn from those labels.

  * split human-labeled traces 70/30 into align / check
  * measure judge-vs-human agreement on the check split, before and after alignment
  * register the aligned judge as "quality" in MLflow's scorer registry (used by Levels 4-5)
    only if it agrees more often than the unaligned one did

    python levels/level4_optimizer/01_align_judge.py
"""

import random

from sia import config, improvers, llm, signals, tracking

if not config.HAS_LLM:
    raise SystemExit(f"Judge alignment needs an LLM: {llm.missing_key_hint(config.JUDGE_MODEL)} (see .env.example).")

tracking.setup()
labeled = [t for t in tracking.agent_traces() if tracking.human_feedback(t) is not None]
random.Random(0).shuffle(labeled)
cut = int(len(labeled) * 0.7)
align_set, check_set = labeled[:cut], labeled[cut:]
print(f"{len(labeled)} human-labeled traces: {len(align_set)} to align, {len(check_set)} to check")
if len(align_set) < 10:
    raise SystemExit("Need at least 10 labeled traces to align. Run Level 0 with --repeat 2, then Level 1.")

judge = signals.quality_judge()
before = improvers.judge_agreement(judge, check_set)
print(f"agreement before alignment: {before:.0%}")

aligned = improvers.align_judge(judge, align_set)
after = improvers.judge_agreement(aligned, check_set)
print(f"agreement after alignment:  {after:.0%}")

if after >= before:
    aligned.register(name="quality")
    print("Registered the aligned judge as 'quality'. Later gates and monitors will use it.")
else:
    print("Alignment did not help on held-out traces; keeping the unaligned judge. "
          "Look for reviewer disagreement (Level 1, 03_signal_quality.py) before trying again.")
