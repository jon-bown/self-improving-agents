---
name: self-improving-agent
description: >-
  Sets up and runs a self-improvement loop for an AI agent with MLflow: signal (traces, human
  feedback, judges) -> store (an evaluation dataset with a hidden holdout) -> update (a new
  version of the agent's instructions) -> gate (promote only if it measurably helps). Works
  with any agent: a Python function, an HTTP service, a CLI, OpenClaw, Hermes, the Claude Agent
  SDK. Use when the user wants their agent to "improve itself", "learn from feedback", "get
  better over time", to build a verifier, regression suite or quality gate from traces, to
  automate prompt optimization or judge alignment, or to decide how much of that loop to
  automate. Also use when an agent is asked to propose changes to its own instructions or skills.
---

# Self-improving agent with MLflow

A component improves when it gets measurably better at its job by consuming signal about its
own performance. The loop has three steps and one rule:

* **signal**: how the agent is doing (a reviewer's pass/fail, an expectation, a judge's verdict, a test result)
* **store**: where that signal accumulates into something reusable (traces, an evaluation dataset, the Prompt Registry)
* **update**: what consumes it and changes a component (a rewritten prompt, a new test, an aligned judge)
* **rule**: an update is only as good as its verifier. Fix the verifier first, and never let
  the thing being improved grade itself.

`references/framework.md` has the full framework and the reasoning behind every guardrail.
Read it before you design anything beyond step 2.

## Step 0: find out where the user is

Ask or check, in this order, and stop at the first gap. That gap is the work.

1. **Is the agent traced?** One MLflow trace per request, with the input, output and tool spans.
   If not, load MLflow's `instrumenting-with-mlflow-tracing` skill (`mlflow agent setup` installs
   MLflow's skills), or use the harness's adapters (`references/adapters.md`).
2. **What is the update target?** The text that defines behavior: a system prompt, `AGENTS.md`,
   a `SKILL.md`. It must live in the MLflow Prompt Registry with a `production` alias, and the
   agent must load it by alias. Code, tools and weights are harder to audit and roll back; leave
   them to a human-accepted proposal.
3. **Is there human signal?** `quality` feedback (pass/fail + rationale) and `expected_response`
   expectations on real traces. For a personal agent the owner's judgment is the ground truth;
   make labeling one keystroke (`selfimprove label`).
4. **Is there a verifier?** Code checks, Guidelines judges, a `quality` judge whose agreement
   with people has been measured, and a dataset with a holdout. When traffic outgrows a text
   judge, add a cheap first pass: a decision model such as Jev (`typesafe:/jev-latest`, MLflow
   3.17+) answers yes/no with a probability, and only its uncertain band goes to the text judge
   (`SI_FAST_JUDGE_MODEL`, `SI_FAST_JUDGE_BAND`). Check spans before asking any model: a missing
   or failed tool call is a free failure.
5. **What is automated today?** Use the levels below. Move up one at a time.

## The harness (preferred when available)

The `selfimprove` package in this repository's `harness/` folder implements the whole loop for an
agent the user brings, configured from `.env`. Check for it (`selfimprove doctor`) before writing
MLflow code by hand. If the user has not set it up:

```bash
cd harness && uv venv -p 3.11 .venv && uv pip install -p .venv -e .
cp .env.example .env      # SI_AGENT + its target, SI_INSTRUCTIONS_FILE, SI_CRITERIA_FILE
.venv/bin/selfimprove doctor --smoke "a typical request"
.venv/bin/selfimprove init
```

| Step | Command | Writes |
|---|---|---|
| traffic | `selfimprove ask "..."`, `selfimprove replay file.txt` | traces tagged `si_source=production` |
| signal (human) | `selfimprove label`, `selfimprove review-queue` | `quality` feedback, `expected_response` |
| signal (auto) | `selfimprove score`, `selfimprove online` | judge and check verdicts on traces |
| verifier health | `selfimprove align` | agreement metric; an aligned `quality` judge |
| store | `selfimprove triage` then `selfimprove approve` | dataset rows, `theme_*` judges, queue items |
| a person's queue | `selfimprove queue list / accept / edit / reject` | dataset rows with `approved_by=human` |
| update | `selfimprove improve [--improver analyst/gepa/metaprompt/command] [--from-file]` | a candidate version (never promoted) |
| gate | `selfimprove gate [N] [--promote]` | scorecards, a decision, the alias move |
| ship / undo | `selfimprove promote N`, `selfimprove rollback`, `selfimprove monitor` | alias moves |
| everything | `selfimprove cycle`, `selfimprove report` | one run per job, nested |

Without the harness, `references/mlflow-recipes.md` has the raw MLflow code for every step.

## Levels of automation

Every level runs the same loop. Only two things change: who writes the candidate, and who
approves it.

| Level | Candidate written by | Approved by | Add when |
|---|---|---|---|
| 1 manual | a person (`improve --from-file`) | a person | always start here |
| 2 agent-assisted | the trace analyst (`improve --improver analyst`) or a coding agent | a person reviews the diff | reading traces is the bottleneck |
| 3 optimizer | GEPA or MetaPrompt (`--improver gepa`) | a person | the judge agrees with people (>= 75%) and the dataset has >= ~20 train rows |
| 4 scheduled | any of the above, from `cycle` | the gate (`SI_AUTO_PROMOTE=true`) | levels 1-3 have run several times without a human overriding the gate |

Decide from the four baseline numbers (`selfimprove report`): traces in, themes out, tests
generated, tests a person edited. Many edits means the criteria or the judge are wrong: fix the
verifier, do not automate further.

## Rules (non-negotiable, also when you are the agent being improved)

1. **The proposer never approves.** Whoever writes a candidate or a test does not decide whether
   it ships. Use the gate and the separate approver step.
2. **The holdout is hidden.** No improver, analyst, approver, or coding agent reads holdout rows or
   traces whose inputs are holdout cases. Only the gate scores them.
3. **A judge verdict is never an expectation.** Expectations come from people or golden cases.
   This includes a decision model's verdict, however confident.
4. **Measure the judge before trusting it.** Judge-vs-human agreement on held-out labels; align
   only with >= 10 labels and >= 30% in each class; register the aligned judge only if
   agreement improved. For a decision model, choose the re-judge band from where its misses
   against human labels fall (`selfimprove align` suggests one); don't guess it.
5. **Promotion is an alias move with a record.** Tag the version it replaced and the live failure
   rate at the time, so rollback is one alias move and can be triggered automatically.
6. **Keep a person on what defines "good".** Thresholds, criteria, golden cases, contradictions,
   flagged tests, and an audit sample of what the automation approved.
7. **Prompts may ship automatically; skills, tools and code may not.** For those, a coding
   agent proposes from the failing traces and a person accepts.
8. **Read the diff.** Optimizers win by pleasing the judge: longer, more hedged, judge-flattering
   text. The bloat guard catches some of it; a person reading the diff catches the rest.

`references/guardrails.md` explains the failure modes these rules exist for (reward hacking,
tests that contradict the spec, self-correction without external feedback).

## When you are asked to improve your own instructions

You are the improver, so rules 1 and 2 bind you. Work only from the failure report and the
train split (`selfimprove improve --improver command` hands you exactly that in a scratch
folder: `TASK.md`, `failure_report.json`, and the instructions file). Edit only the instructions
file, keep the change minimal and evidence-based, point at sources instead of pasting facts, and
say in two or three sentences what you changed and why. Then stop: the gate decides.

## Bringing an agent

`references/adapters.md` shows how to wire up each kind of agent (`SI_AGENT=python|http|command|litellm|openclaw`),
how to keep the user's own agent configuration untouched (isolated state directories, separate
ports, API keys from `.env` only), and how to point the harness at traces an agent already logs
(`SI_PRODUCTION_FILTER`).

## Related MLflow skills

Load these for the detailed mechanics; this skill decides *what* to do and in which order.

* `instrumenting-with-mlflow-tracing` for step 1
* `build-a-scorer` for choosing and writing checks and judges
* `retrieving-mlflow-traces`, `analyze-mlflow-trace`, `debug-agent` for reading failures
* `agent-evaluation`, `fix-agent-issue` for the evaluate-change-verify loop on code
