# Guardrails, and the failures they exist for

Removing a person from approval and promotion is where self-improving systems go wrong. The
research is consistent about how:

* LLMs often fail to self-correct without external feedback (Huang et al., ICLR 2024).
* The Darwin Gödel Machine faked test logs and removed its own hallucination markers (Sakana AI).
* METR found o3 reward-hacking in about 30% of RE-Bench runs.
* ImpossibleBench: frontier models satisfy tests that contradict the specification 50-76% of the time.
* Decagon: growing GEPA's training set from 100 to 500 examples made the prompt 75% longer and
  lowered holdout accuracy.

An optimizer with access to its own evaluator can end up optimizing the evaluator instead of the task.

| Guardrail | Stops | Where in the harness |
|---|---|---|
| separate approver | the proposer approving its own tests | `approve` is its own step with its own prompt |
| hidden holdout | overfitting to the gate | `store.is_holdout`; improvers and the analyst never see it |
| promotion margin | candidates that only win on train | `SI_PROMOTION_MARGIN`, gate rule 3 |
| must-pass scorers | trading a critical failure for an average gain | `must_pass: true` in criteria |
| bloat guard | judge-pleasing length | `SI_MAX_PROMPT_GROWTH` |
| cap per theme | one bug becoming forty tests | `SI_TESTS_PER_THEME_CAP` |
| contradiction path | encoding whichever reviewer was read first | triage -> human queue |
| judge-only failures never become tests | the judge writing its own ground truth | expectations only from people |
| judge agreement check | optimizing against a drifted judge | `align`; the cycle halts when it can't recover |
| provenance tags | not knowing where a bad batch came from | `source`, `source_trace`, `author`, `source_run` on every row |
| rollback | a promotion live traffic disagrees with | `monitor`, `SI_ROLLBACK_MARGIN` |
| every step is a run | silent automation | `si.job` runs with outcome, metrics, artifacts |

## What a person keeps

* the thresholds and the criteria (they define what counts as good),
* golden cases and expectations,
* contradictions, flagged tests, and judge-only failures that need a label,
* a periodic audit of a sample of what the approver merged,
* accepting any change to skills, tools or code,
* reviewing rollbacks.

## Red flags in a candidate diff

* new hedging or caveats that judges reward ("I believe", "please verify")
* content copied from the failing cases or the judge's rationales into the instructions
* facts pasted in instead of a pointer to the source (the next policy change breaks it again)
* instructions about the evaluation itself ("graders prefer...", "always mention...")
* a large length increase with no matching new behavior
