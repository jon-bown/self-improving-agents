# Workflows: running the loop on a schedule

Part 3 of the framework sorts the loop's steps into jobs that can run without a person, and jobs
that must keep one. The harness ships those jobs three ways: as CLI commands, as one `cycle`, and
as GitHub Actions. Every job is an MLflow run tagged `si.job`, with its outcome, metrics and
artifacts, so unattended work is never silent.

## Contents

* `cycle`: the whole loop once
* GitHub Actions
* Auto-promotion and rollback
* What a person keeps, and how they record it
* Ungated outside changes
* Resuming an interrupted run
* A precomputed store for demos

## `cycle`: the whole loop once

```
score -> monitor -> align -> triage -> approve -> [enough failures?] -> improve -> gate
```

`selfimprove cycle` runs these as one parent MLflow run with each job nested inside it. Its own
guardrails make it safe to schedule:

| Guardrail | Setting (default) |
|---|---|
| no signal, no change: `improve` runs only above a failure rate, with enough traces | `SI_FAIL_THRESHOLD` (0.15), `SI_MIN_TRACES` (8); `--force` skips both |
| verifier health: halt and queue for a person if the judge disagrees with people and re-alignment fails | `SI_AGREEMENT_THRESHOLD` (0.75) |
| triage waits for enough new failures | `SI_TRIAGE_THRESHOLD` (5) |
| one theme can't flood the dataset | `SI_TESTS_PER_THEME_CAP` (3) |
| the gate's own rules | `SI_PROMOTION_MARGIN`, `SI_MAX_PROMPT_GROWTH` (2.0), must-pass checks, the holdout |
| promotion stays with a person | `SI_AUTO_PROMOTE=false` |

`--improver` picks who writes the candidate (`analyst`, `metaprompt`, `gepa`, `command`); the
default is `SI_IMPROVER`.

## GitHub Actions

Three workflows at the repository root, all running in `harness/`:

| Workflow | Runs | When |
|---|---|---|
| `selfimprove-loop.yml` | nightly: `score`, `monitor`, `align`, `triage`, `approve`; weekly: `cycle`; or any one job by hand | 06:17 UTC daily, 07:43 UTC Mondays, `workflow_dispatch` |
| `selfimprove-regression.yml` | every dataset row as an `@mlflow.test` case against the agent and judges | pull requests touching `harness/`, or by hand for any version or alias |
| `selfimprove-offline.yml` | the harness's own tests, offline, no keys | every change to `harness/` |

To turn the loop on:

1. Repository variable `SELFIMPROVE_ENABLED=true` (the loop and regression jobs are off until then).
2. Secret `MLFLOW_TRACKING_URI`: a server with a database backend, shared with production,
   because the jobs read production traces and write the dataset and Prompt Registry there.
3. Secrets for your provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TYPESAFE_API_KEY` ...).
4. Optional variables: `SI_EXPERIMENT`, `SI_IMPROVER`, `SI_AUTO_PROMOTE`, `SI_FAST_JUDGE_MODEL`.
   The workflow copies `harness/.env.example` to `.env`, so commit your agent settings there or
   set them as variables.

The human queue is a file (`.selfimprove/human_queue.jsonl`). The loop workflow caches it
between runs and uploads it as an artifact; point `SI_STATE_DIR` at shared storage if several
people work the queue. Each run's `selfimprove report` goes to the job summary.

## Auto-promotion and rollback

With `SI_AUTO_PROMOTE=true`, a candidate that passes the gate becomes `production` with no person
involved. Turn it on only after levels 1-3 have run several times without a person overriding the
gate. The promotion records the version it replaced and the live failure rate at the time.

`monitor` then compares live traffic on the new version with that recorded rate, and rolls back if
it got worse by more than `SI_ROLLBACK_MARGIN` (0.10). It needs fresh scored traffic on the new
version, so it runs nightly, before triage. A person reviews every rollback.

## What a person keeps, and how they record it

Even at level 4, people keep: the thresholds and criteria, golden cases and expectations, labels on
a sample of traffic, the human queue (contradictions, flagged tests, judge-only failures), an audit
sample of what the approver merged, and every change to skills, tools or code.

Record these decisions in MLflow, not in chat, so the next person can see why:

* `promote N --reason "..."` for a promotion;
* `queue accept|reject|edit <id> --note "..."` for a test;
* a guardrail change (for example raising `SI_MAX_PROMPT_GROWTH` because a candidate's extra length
  is a real new section, not padding) or a verifier fix: change the setting or the criteria file,
  re-gate the same version, and note who decided and why with that gate run.

## Ungated outside changes

Real teams edit instructions outside the loop: someone appends a section to production and ships
it. Treat that version like any other production version. It gets traffic, the scorers catch what
it broke, and the loop fixes it through the gate. Register it as a new version with its author and
message rather than overwriting the old one, so the history shows where the regression came from.
`harness/session` loops 3 and 4 start this way.

## Resuming an interrupted run

A machine that sleeps or a killed process can stop a `cycle` partway. Its runs stay `RUNNING`;
mark them `KILLED` so the history is honest. Then run the next `cycle` against the stored state.
Don't replay the traffic or re-apply an outside change that already happened: the dataset, traces
and versions are already in MLflow. A candidate written but never gated stays an ungated version;
the next cycle's gate decides about whatever it writes. Long runs on a laptop need sleep disabled.

## A precomputed store for demos

For a talk or a demo, run the loops ahead of time into a store served by its own MLflow server,
then present read-only notebooks against it, so nothing calls a model or an agent live.
`harness/session` does this: `make session-server`, `make session-notebooks`, and
`python -m session.precompute [--loops N]` to rebuild it (about two hours for four loops).
