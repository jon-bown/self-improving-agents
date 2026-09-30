"""Level 1 · Is the signal any good? Check the verifiers before trusting them.

"Each component is only as good as its verifier." An optimizer pointed at a judge that
disagrees with your reviewers will happily optimize for the wrong thing. So before we let any
automated signal drive an update, we measure:

  1. judge vs. human     how often does the LLM "quality" judge agree with the human label?
  2. reviewer vs. rubric which reviewers fail answers for reasons the others would not?
  3. rationale themes    what do the failure rationales actually say? (the input to Level 2)

    python levels/level1_signal/03_signal_quality.py
"""

import re
from collections import Counter

from mlflow.entities import AssessmentSourceType

from sia import tracking

tracking.setup()
traces = tracking.agent_traces()
labeled = [t for t in traces if tracking.human_feedback(t) is not None]
print(f"{len(labeled)} of {len(traces)} traces have a human label\n")

# ---- 1. judge vs human ---------------------------------------------------------------------
pairs = []
for t in labeled:
    judge = [a for a in t.info.assessments
             if a.name == "quality" and a.source.source_type == AssessmentSourceType.LLM_JUDGE]
    if judge:
        pairs.append((bool(tracking.human_feedback(t).value), str(judge[-1].value).lower() in ("true", "yes")))
if pairs:
    agree = sum(h == j for h, j in pairs) / len(pairs)
    fp = sum(j and not h for h, j in pairs)
    fn = sum(h and not j for h, j in pairs)
    print(f"quality judge agrees with humans on {agree:.0%} of {len(pairs)} traces "
          f"(judge passed {fp} that humans failed; failed {fn} that humans passed)")
    print("  -> below ~85% agreement, don't let this judge drive updates yet (Level 4 aligns it)\n")
else:
    print("No LLM-judge 'quality' verdicts yet (run 02_automated_signal.py with an API key)\n")

# ---- 2. reviewer strictness ----------------------------------------------------------------
by_reviewer = Counter()
fails_by_reviewer = Counter()
for t in labeled:
    fb = tracking.human_feedback(t)
    by_reviewer[fb.source.source_id] += 1
    fails_by_reviewer[fb.source.source_id] += 0 if fb.value else 1
print("reviewer  labeled  fail-rate")
for r in sorted(by_reviewer):
    print(f"{r:<9} {by_reviewer[r]:>7}  {fails_by_reviewer[r] / by_reviewer[r]:>9.0%}")

# ---- 3. what the failure rationales say ----------------------------------------------------
themes = Counter()
for t in labeled:
    fb = tracking.human_feedback(t)
    if fb.value:
        continue
    text = fb.rationale or ""
    themes.update(re.findall(r"(Stated something wrong or outdated|Missing a key fact|Buries the answer|Too long)", text))
print("\nfailure rationale themes (human):")
for theme, n in themes.most_common():
    print(f"  {n:>3}  {theme}")
print("\nNext: levels/level2_manual_loop/ -- turn this signal into a change.")
