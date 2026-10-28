---
name: self-improving-agent
description: >-
  Sets up, runs and automates a self-improvement loop for an AI agent with MLflow: signal
  (traces, human feedback, code checks, judges) -> store (an evaluation dataset with a hidden
  holdout) -> update (a new version of the agent's instructions) -> gate (promote only if it
  measurably helps, with rollback). Covers the `selfimprove` harness and its CLI jobs, the
  scheduled workflows (`cycle`, GitHub Actions), writing the verifier (criteria files, tool-span
  checks, aligned judges), handling gate rejections, and bringing any agent: a Python function,
  an HTTP service, a CLI, OpenClaw (with the MLflow OpenClaw connector), Hermes, the Claude
  Agent SDK. Use this whenever the user wants an agent to "improve itself", "learn from
  feedback" or "get better over time"; to build a verifier, regression suite or quality gate
  from traces; to trace OpenClaw runs into MLflow; to automate prompt optimization, judge
  alignment or promotion; to decide how much of the loop to automate; or to debug why a gate
  rejected a candidate. Also use it when an agent is asked to propose changes to its own
  instructions or skills.
---

# Self-improving agent with MLflow

A component improves when it gets measurably better at its job by consuming signal about its
own performance. The loop has three steps and one rule:

* **signal**: how the agent is doing (a reviewer's pass/fail, an expectation, a check, a judge's verdict)
* **store**: where that signal accumulates into something reusable (traces, an evaluation dataset, the Prompt Registry)
* **update**: what consumes it and changes a component (a rewritten prompt, a new test, an aligned judge)
* **rule**: an update is only as good as its verifier. Fix the verifier first, and never let
  the thing being improved grade itself.

`references/framework.md` has the full framework and the reasoning behind every guardrail.
Read it before designing anything beyond step 2 below.

## Step 0: find out where the user is

Check in this order and stop at the first gap. That gap is the work.

1. **Is the agent traced?** One MLflow trace per request, with the input, the output and a span
   per tool call. Without tool spans you can't check *how* the agent answered (did it read the
   policy?), only *what* it said. For OpenClaw, install the connector (`references/openclaw.md`).
   Otherwise use the harness's adapters (`references/adapters.md`) or MLflow's
   `instrumenting-with-mlflow-tracing` skill.
2. **What is the update target?** The text that defines behavior: a system prompt, `AGENTS.md`,
   a `SKILL.md`. It lives in the MLflow Prompt Registry with a `production` alias, and the agent
   loads it by alias. Code, tools and weights are harder to audit and roll back; leave them to a
   proposal a person accepts.
3. **Is there human signal?** `quality` feedback (pass/fail plus a rationale) and
   `expected_response` expectations on real traces. For a personal agent the owner's judgment is
   the ground truth, so make labeling one keystroke (`selfimprove label`).
4. **Is there a verifier?** Code checks, judges whose agreement with people has been measured,
   and a dataset with a holdout. `references/verifiers.md` covers the criteria file.
5. **What is automated today?** Use the levels below, and move up one at a time.

## Pick a path

| The user has | Use |
|---|---|
| their own agent, and wants the loop around it | the **harness** (`harness/`, the `selfimprove` CLI): the rest of this skill |
| no agent yet, and wants to learn the loop step by step | the **level-based tutorial** in `sia/`: set a provider and key in `sia/.env`, then `make level0 level1 ...` (its README) |
| no harness, and wants to write MLflow code directly | `references/mlflow-recipes.md`: raw MLflow for every step |

`harness/session/` is a complete worked example: four loops on a real OpenClaw agent, with the
criteria files, traffic, reviewer and runbook for each (`RUNBOOK.md`). Copy its patterns.

## The harness

`selfimprove` implements the whole loop for an agent the user brings, configured from `.env`.
Check for it (`selfimprove doctor`) before writing MLflow code by hand. To set it up:

```bash
cd harness && uv venv -p 3.11 .venv && uv pip install -p .venv -e .
cp .env.example .env      # SI_AGENT + its target, SI_INSTRUCTIONS_FILE, SI_CRITERIA_FILE
.venv/bin/selfimprove doctor --smoke "a typical request"
.venv/bin/selfimprove init  # instructions v1 -> production; golden cases -> dataset
```

| Step | Command | Writes |
|---|---|---|
| traffic | `ask "..."`, `replay file.txt` | traces tagged `si_source=production` |
| signal (people) | `label [--why] [--expected]`, `review-queue` | `quality` feedback, `expected_response` |
| signal (automated) | `score`, `online` | check and judge verdicts on traces |
| verifier health | `align [--force]` | judge-vs-human agreement; an aligned `quality` judge |
| store | `triage [--force]`, then `approve` | dataset rows, `theme_*` judges, queued tests |
| a person's queue | `queue list`, `queue accept/reject/edit <id> --note ...` | dataset rows with `approved_by=human` |
| update | `improve [--improver analyst/metaprompt/gepa/command] [--from-file F] [--budget N]` | a candidate version, never promoted |
| inspect | `diff production latest` | the change, to read before gating |
| gate | `gate [N] [--promote]` | scorecards, a decision, optionally the alias move |
| ship / undo | `promote N --reason ...`, `rollback`, `monitor` | alias moves with a record |
| everything | `cycle [--improver ...]`, `report` | one MLflow run per job, nested |

Every job is an MLflow run, so the history of what changed, why, and who decided is in MLflow,
not in someone's terminal.

## Levels of automation

Every level runs the same loop. Two things change: who writes the candidate, and who approves it.

| Level | Candidate written by | Approved by | Move up when |
|---|---|---|---|
| 1 manual | a person (`improve --from-file`) | a person (`promote`) | always start here |
| 2 assisted | the trace analyst (`improve --improver analyst`) or a coding agent | a person reads the diff | reading traces is the bottleneck |
| 3 optimizer | MetaPrompt, then GEPA (`--improver metaprompt`, `gepa`) | a person | the judge agrees with people (>= 75%) and there are >= ~20 train rows |
| 4 scheduled | any of the above, from `cycle` | the gate (`SI_AUTO_PROMOTE=true`), with `monitor` | levels 1-3 have run several times without a person overriding the gate |

Decide from `selfimprove report`'s four baseline numbers: traces in, themes out, tests
generated, tests a person edited. Many edits means the criteria or the judge are wrong: fix the
verifier, don't automate further. `references/workflows.md` covers scheduling and CI.

## When the gate rejects a candidate

A rejection is the gate doing its job. Read *which rule* failed before doing anything:

| Rejected for | Ask | Then |
|---|---|---|
| worse than production on train | Are the train rows right? | Review the human queue (`queue list`); accept the tests that are right, reject the rest; run the improver again |
| a must-pass check below 100% | Open the failing row. Is the agent wrong, or the check? | Agent wrong: improve again. Check wrong (it flagged a correct answer, like a refusal matching a "send me your password" regex): fix the check, record why, and re-gate the same version |
| no gain on the holdout | Is the candidate overfitting train? | Try another improver; never show the holdout to the improver |
| bloat (longer than `SI_MAX_PROMPT_GROWTH` times production) | Is the extra length new behavior, or padding and hedging? | Padding: reject. Real content on a short prompt: a person may raise the limit for this agent, with a written reason, then `gate <version>` again |

Two lines not to cross: loosening a threshold or editing a check just so a candidate passes,
and letting the improver (or you, as the improver) make that call. A guardrail change and a
verifier fix are a person's decisions, recorded with who and why, like a promotion.

## Rules (also when you are the agent being improved)

1. **The proposer never approves.** Whoever writes a candidate or a test doesn't decide whether
   it ships. Use the gate and the separate approver step.
2. **The holdout is hidden.** No improver, analyst, approver or coding agent reads holdout rows,
   or traces whose inputs are holdout cases. Only the gate scores them.
3. **A judge verdict is never an expectation.** Expectations come from people or golden cases,
   including when a decision model is confident.
4. **Measure the judge before trusting it.** Agreement with people on held-out labels; align
   only with >= 10 labels and >= 30% in each class; register the aligned judge only if agreement
   improved. For a decision model's re-judge band, use where its misses fall (`align` suggests one).
5. **Promotion is an alias move with a record.** Tag the version it replaced and the live failure
   rate at the time, so rollback is one alias move and can be automatic (`monitor`).
6. **Keep a person on what defines "good".** Thresholds, criteria, golden cases, contradictions,
   flagged tests, and an audit sample of what the automation approved.
7. **Prompts may ship automatically; skills, tools and code may not.** For those, a coding agent
   proposes from the failing traces and a person accepts.
8. **Read the diff.** Optimizers win by pleasing the judge: longer, more hedged, judge-flattering
   text. The bloat guard catches some of it; a person reading the diff catches the rest.

`references/guardrails.md` explains the failure modes these rules exist for.

## When you are asked to improve your own instructions

You are the improver, so rules 1 and 2 bind you. Work only from the failure report and the train
split: `selfimprove improve --improver command` hands you exactly that in a scratch folder
(`TASK.md`, `failure_report.json`, the instructions file). Edit only the instructions file, keep
the change minimal and evidence-based, point at sources instead of pasting facts, and say in two
or three sentences what you changed and why. Then stop: the gate decides.

## Reference files

Read the one that matches the task:

* `references/framework.md`: the framework, self-improvement vs self-evolution, the baseline numbers
* `references/verifiers.md`: the criteria file (checks, tool-span checks, grounded judges, guidelines, must-pass), choosing and testing checks, a fast first-pass judge
* `references/workflows.md`: `cycle`, scheduled jobs and GitHub Actions, auto-promotion and rollback, ungated outside changes, resuming an interrupted run
* `references/adapters.md`: bringing an agent (`SI_AGENT=python|http|command|litellm|openclaw`), Hermes, keeping the user's own setup untouched
* `references/openclaw.md`: OpenClaw with the harness, and the MLflow OpenClaw connector (install, what it records, sessions, linking traces)
* `references/guardrails.md`: the failures the rules prevent, and red flags in a candidate diff
* `references/troubleshooting.md`: environment and MLflow gotchas, with fixes
* `references/mlflow-recipes.md`: every step in raw MLflow, without the harness

## Related MLflow skills

Load these for the detailed mechanics; this skill decides *what* to do and in which order.

* `instrumenting-with-mlflow-tracing` for tracing an agent
* `build-a-scorer` for choosing and writing checks and judges
* `retrieving-mlflow-traces`, `analyze-mlflow-trace`, `debug-agent` for reading failures
* `agent-evaluation`, `fix-agent-issue` for the evaluate-change-verify loop on code
