# The framework

From the *Self-Improvement & MLflow* series (Parts 1-3) and the ODSC West 2026 tutorial.

## Self-improvement vs self-evolution

* **Self-improvement**: a component gets measurably better at its job over time by consuming
  signal about its own performance. A person or another agent can drive the update and check it.
* **Self-evolution**: the stronger claim that the update runs continuously and autonomously.
  Gao et al. call fully autonomous self-evolution "an aspirational goal rather than the current
  norm."

Neither is limited to the agent. The judge, the dataset, the test suite and the prompt can each
improve from the same signal, and mature systems improve several of them together.

## The loop

| Role | MLflow component | Entry point |
|---|---|---|
| store · raw | traces | `@mlflow.trace`, `mlflow.<flavor>.autolog()`, `mlflow.start_span` |
| signal · human | feedback / expectations | `mlflow.log_feedback`, `mlflow.log_expectation` |
| signal · structured | label schemas, review queues | `create_label_schema`, `create_review_queue` |
| signal · automated | scorers and judges | `@scorer`, `Guidelines`, `make_judge`, `Correctness` |
| signal · learns | judge alignment | `judge.align(traces, MemAlignOptimizer() / SIMBAAlignmentOptimizer())` |
| store · curated | evaluation datasets | `create_dataset`, `merge_records`, `set_dataset_tags` |
| query | trace search | `mlflow.search_traces`, `mlflow traces search` |
| gate | regression tests | `@mlflow.test` + `mlflow.genai.evaluate` |
| signal · production | automatic evaluation | `scorer.register().start(ScorerSamplingConfig(...))` |
| signal · cheap first pass | decision-model judges (Jev) | `make_judge(model="typesafe:/jev-latest")`; probability in `metadata["typesafe.probability"]` |
| update · prompt | prompt optimization | `mlflow.genai.optimize_prompts` (GEPA, MetaPrompt) |
| update · code | coding agent + MLflow skills | `mlflow agent setup` |
| ship | Prompt Registry aliases | `register_prompt`, `set_prompt_alias` |

## Update targets, easiest to hardest to audit and roll back

```
prompt ──▶ memory ──▶ skill / code ──▶ weights ──▶ architecture
```

Start at the left. A prompt version is a numbered, diffable artifact whose rollback is an alias
move. That is why the harness only promotes prompts automatically.

## The verifier comes first

LLMs cannot reliably self-correct without external feedback (Huang et al.). Whatever performs
the update takes the verifier's judgment as its input, so a weak or gameable verifier gets
optimized against, and a verifier the agent can influence gets influenced. When progress
plateaus, improve the verifier before the agent:

* judges that track the reviewers (measured agreement, alignment),
* a dataset that grows from real failures (with a cap per theme, so forty complaints about one
  bug become one test),
* a test suite that gates every change, plus a holdout nothing upstream has seen.

## Why the manual loop stops scaling (Part 2)

With more reviewers and more traffic four problems show up:

* **criteria drift**: reviewers discover their criteria by grading (Shankar et al.), so ten
  people grade against ten rubrics. A label schema narrows the spread.
* **reviewer disagreement**: the same trace passes for one reviewer and fails for another. Route
  contradictions to a person; never turn them into tests.
* **duplicate signal**: forty thumbs-downs are one bug. Deduplicate and cap tests per theme.
* **judge drift**: the judge was never checked against people, and LLM judges have position and
  verbosity biases (Zheng et al.). Its verdicts soon outnumber the human ones.

The bottleneck is reading. A coding agent with MLflow's skills, or the harness's trace analyst,
reads the failures and proposes themes, criteria, tests and a fix for a person to approve.

## Sorting the steps (Part 3)

| Step | Scheduled | Optimizer | Stays with a person |
|---|---|---|---|
| Reading the themes | triage job with a cap, contradiction path, provenance | | flagged contradictions |
| Keeping signal flowing | sampled evaluation, a cheap first-pass judge | | golden cases and expectations |
| Keeping the judge aligned | nightly agreement check | `align()` with MemAlign, SIMBA or GEPA | the agreement threshold |
| Approving the tests | approver checks, then automatic merge | | flagged tests, audit sample, the criteria |
| Changing the prompt | candidate scored on CI and holdout | `optimize_prompts` | the promotion margin |
| Changing a skill or tool | | | accepting the agent's proposal |
| Promoting to production | alias moves when both gates pass | | rollback review |

## The four baseline numbers

Record them on every triage run (`selfimprove report`):

1. **traces in**: too few means too little signal for downstream automation.
2. **themes out**: many traces in a few consistent themes means automating theme assignment pays.
3. **tests generated**: many per theme means the triage agent writes one test per complaint; cap it.
4. **tests a person edited before merging**: many means the criteria or the judge do not match
   what reviewers want yet. Fix that before automating more.

## Personal agents

Public benchmarks (WildClawBench, PersonaMem, PrefEval) score a model or harness on other people's
tasks and personas. They may not say how well an agent serves *you*. The best ground truth is the
owner's own judgment, so improving a personal agent starts with a verifier built from the owner's
feedback: traces and assessments for signal, a dataset and judges for the store, and the prompt,
skills and tools as update targets that count as improvements only when the verifier says so.
