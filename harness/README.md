# selfimprove: an MLflow harness for self-improving agents

The framework from the *Self-Improvement & MLflow* blog series, packaged for **an agent you
bring**. Your agent's instructions become a versioned prompt in MLflow; its traffic becomes
traces; people and judges turn traces into signal; the signal becomes a regression suite with a
hidden holdout; and a candidate (written by you, an LLM analyst, an optimizer, or a coding agent)
ships only when the gate says it measurably helps.

```
        ┌──────── signal ────────┐     ┌──── store ────┐     ┌──── update ────┐
traces ─┤ human labels           ├─────▶ dataset       ├─────▶ candidate      ├──▶ gate ──▶ promote
        │ code checks · judges   │     │ (+ holdout,   │     │ instructions   │   (CI +     (alias
        └────────────────────────┘     │  provenance)  │     │ version        │   holdout)   move)
                                       └───────────────┘     └────────────────┘
```

| | what | where |
|---|---|---|
| **workflows** | the scheduled jobs from Part 3: score, triage, approve, align, improve, gate, monitor, cycle | `selfimprove/workflows/`, the `selfimprove` CLI, `.github/workflows/selfimprove-*.yml` |
| **notebooks** | the manual steps, one idea at a time, with the raw MLflow calls in the cells | `notebooks/` |
| **config** | bring your own agent, models, criteria and guardrails | `.env` (from `.env.example`), `examples/criteria.yaml` |
| **skill** | an agent skill that applies the framework with MLflow and the agent of your choice | [`../skills/`](../skills/README.md), at the repo root |

## Quickstart

```bash
cd harness
uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[notebooks]"
cp .env.example .env            # add ANTHROPIC_API_KEY (or another provider), or leave it empty
.venv/bin/selfimprove doctor    # shows the config; checks that your agent loads
.venv/bin/selfimprove init      # instructions v1 -> production; golden cases -> dataset
```

The defaults run the example agent (`examples/support_agent.py`, a support agent for a fictional
product with three seeded flaws). With no API key it falls back to a rule-based stand-in and code
checks, so every command and notebook still runs.

```bash
selfimprove replay examples/questions.txt --repeat 2     # a week of production traffic
selfimprove label                                        # you are the reviewer: p / f / s / q
python -m examples.reviewer --all                        # ...or let three simulated reviewers do it
selfimprove score                                        # judges and checks on the new traces
selfimprove triage --force && selfimprove approve        # failures -> themes -> tests -> dataset
selfimprove improve                                      # the analyst writes a candidate
selfimprove gate --promote                               # ships only if it passes
selfimprove report                                       # the baseline numbers, the human queue
```

Or all of it, in order, as one run: `selfimprove cycle`.

## Bring your own agent

Set `SI_AGENT` and one target in `.env`:

| `SI_AGENT` | you provide | example |
|---|---|---|
| `python` | `def answer(question, instructions) -> str` (optionally `workspace`) | `SI_AGENT_TARGET=my_pkg.agent:answer` |
| `http` | a service: POST `{"input", "instructions"}` → `{"output"}` | `SI_AGENT_URL=http://127.0.0.1:8000/answer` |
| `command` | a CLI; `{input}`, `{instructions_file}`, `{workspace}` are filled in | `SI_AGENT_COMMAND=my-agent --system-file {instructions_file} --message {input}` |
| `litellm` | nothing: a chat model with the instructions as system prompt | `SI_AGENT_MODEL=anthropic:/claude-haiku-5-5` |
| `openclaw` | OpenClaw; the instructions become its `AGENTS.md` | `SI_AGENT_MODEL=anthropic:/claude-haiku-5-5` |

Then point `SI_INSTRUCTIONS_FILE` at the text that defines your agent's behavior and
`SI_CRITERIA_FILE` at what "good" means for it. `examples/hermes_agent.py` wraps Hermes Agent's
`AIAgent`; the same pattern wraps any SDK. Already tracing your agent yourself? Set
`SI_PRODUCTION_FILTER` and the harness reads those traces as production signal.

Personal agents are someone's real setup, so the adapters are isolated: OpenClaw runs against a
profile in `.selfimprove/openclaw/` (never `~/.openclaw`, never your gateway's port), Hermes against
a home in `.selfimprove/hermes/` with only `ANTHROPIC_API_KEY`, and the harness's MLflow store is a
local SQLite file unless you point `MLFLOW_TRACKING_URI` elsewhere.

For OpenClaw's own spans (each model call, each tool call, token usage), set
`SI_OPENCLAW_MLFLOW=1` with an HTTP `MLFLOW_TRACKING_URI` and run `selfimprove doctor`. It builds
the MLflow tracing plugin in [`connectors/openclaw-mlflow`](../connectors/openclaw-mlflow/README.md)
and installs it into the isolated profile. OpenClaw then logs a second trace per call to the same
experiment, linked by `openclaw.session_key`. The same `install.sh` connects your everyday OpenClaw
to MLflow.

## The verifier

`examples/criteria.yaml`, cheapest first:

```yaml
quality:            # the alignable judge: same name as the reviewers' feedback
  instructions: "... {{ inputs }} ... {{ outputs }} ..."
checks:             # code: regexes, word limits, required tool spans
  - {name: no_stale_policy, must_not_match: ['30[- ]day'], must_pass: true}
  - {name: looked_up_policy, requires_tool: lookup_policy}
grounded:           # span checks on a tool call, then a judge on the tool's output
  - {name: grounded_in_policy, tool: lookup_policy, requires_input: topic, instructions: "Does ..."}
guidelines:         # LLM judges from plain-English requirements
  - {name: leads_with_answer, guidelines: "The first sentence must directly answer ..."}
reference: [key_facts, correctness]     # need expectations; dataset only
python: my_pkg.scorers:SCORERS               # anything YAML can't express
```

Triage adds `theme_<slug>` judges to MLflow's scorer registry as failure themes are confirmed, so
the verifier grows with the evidence. `must_pass` scorers block any candidate that fails them on any row.

### A cheap first pass with Jev

Set `SI_FAST_JUDGE_MODEL=typesafe:/jev-latest` and `TYPESAFE_API_KEY` (MLflow 3.17+). Jev is a
decision model: a yes/no with a probability, no generated text, cheap enough to judge most
production traffic (Part 3, §3). The `quality` judge and the `grounded` checks then become a
cascade: Jev answers first, and verdicts whose pass probability falls inside `SI_FAST_JUDGE_BAND`
(default `0.5,0.8`) are re-judged by `SI_JUDGE_MODEL`, which writes the rationale. Every verdict
keeps `typesafe.probability` and `si.escalated` in its metadata.

* `selfimprove align` reports how often Jev agrees with people and where its misses fall, and
  suggests the narrowest band that contains every miss. You set the band; alignment (which needs
  rationales) applies to the text judge.
* `selfimprove online` samples Jev server-side at 100% into `quality_first_pass`; `selfimprove score`
  turns each of those into the loop's `quality` verdict, re-judging the uncertain ones.
* Jev's verdicts never become expectations, and triage ignores judge-only failures while the
  judge disagrees with people.

## Workflows

Each job is one MLflow run (tag `si.job`) with its outcome, metrics and artifacts.

| job | Part 3 | does | a person keeps |
|---|---|---|---|
| `score` | §3 | checks + judges (Jev first, if set) on new production traces, no re-run; re-judges the server's uncertain Jev verdicts; `online` registers judges for server-side sampling | golden cases, expectations |
| `triage` | §2 | checkpointed read of new failures; contradictions → queue; analyst themes + criteria; proposes tests | contradictions |
| `approve` | §6 | reproduces? near duplicate? under the theme cap? reviewer agent? merge or queue; registers theme judges | flagged tests, audit sample |
| `align` | §5 | judge-vs-human agreement on a held-out slice; aligns (MemAlign / SIMBA) and registers only if it helped; calibrates Jev's band | the threshold, the band |
| `improve` | §4 | candidate from `analyst`, `gepa`, `metaprompt`, `command` (a coding agent), or `--from-file` | accepting skill/tool changes |
| `gate` | §7 | candidate vs production on train + holdout; must-pass, margin, bloat; promotes if allowed | the margin; `SI_AUTO_PROMOTE` |
| `monitor` | §8 | rolls back a gate promotion if live failures rise by `SI_ROLLBACK_MARGIN` | rollback review |
| `cycle` | all | score → monitor → align → triage → approve → improve → gate | |
| `report` | §1 | traces in, themes out, tests generated, tests a person edited; the queue | |

The human queue (`selfimprove queue list | accept | edit | reject`) holds everything the
automation refuses to decide: reviewer contradictions, failures without an expected answer,
flagged tests, judge-only failures that need a label, and halts.

Guardrails (`.env`): `SI_TRIAGE_THRESHOLD`, `SI_TESTS_PER_THEME_CAP`, `SI_HOLDOUT_FRACTION`,
`SI_AGREEMENT_THRESHOLD`, `SI_MIN_ALIGNMENT_TRACES`, `SI_PROMOTION_MARGIN`, `SI_MAX_PROMPT_GROWTH`,
`SI_ROLLBACK_MARGIN`, `SI_FAIL_THRESHOLD`, `SI_MIN_TRACES`, `SI_AUTO_PROMOTE`.

### Scheduling

`.github/workflows/` at the repository root:

* `selfimprove-offline.yml`: the whole loop offline on every change to `harness/` (no keys)
* `selfimprove-regression.yml`: every dataset row as an `@mlflow.test` case on pull requests
* `selfimprove-loop.yml`: nightly score/monitor/align/triage/approve, a weekly `cycle`, and a
  manual trigger for any job

The last two are off until you set the repository variable `SELFIMPROVE_ENABLED=true`, the secret
`MLFLOW_TRACKING_URI` (a server with a database backend, shared with production), and a provider
key. Promotion stays with a person unless `SI_AUTO_PROMOTE=true`.

## Notebooks

| notebook | blog | shows |
|---|---|---|
| [00 · Bring your agent](notebooks/00_bring_your_agent.ipynb) | Pt 2 §1 | the adapter contract, the Prompt Registry, one trace per request, traffic |
| [01 · Signal](notebooks/01_signal.ipynb) | Pt 1 §2, Pt 2 §1, Pt 3 §3 | `log_feedback`/`log_expectation`, checks, Jev first pass, grounded span checks, Guidelines, `make_judge`, `evaluate()` on existing traces |
| [02 · Store](notebooks/02_store.ipynb) | Pt 2 §2 | dataset from labeled failures, dedupe, golden cases, hash-based holdout, provenance |
| [03 · Update and gate](notebooks/03_update_and_gate.ipynb) | Pt 2 §3-4 | a hand-written v2, side-by-side evaluation, the gate, `@mlflow.test`, promotion |
| [04 · Triage and approve](notebooks/04_triage_and_approve.ipynb) | Pt 2 §5-6, Pt 3 §2, §6 | reviewer disagreement, the failure report, themes, the approver, the human queue, the four numbers |
| [05 · Verifier and optimizer](notebooks/05_verifier_and_optimizer.ipynb) | Pt 3 §3-5, §7 | judge agreement and alignment, calibrating Jev's band, online scoring, `optimize_prompts`, the holdout margin |
| [06 · The loop runs itself](notebooks/06_automate.ipynb) | Pt 3 | `cycle()`, auto-promotion, rollback, scheduling, what stays with a person |

```bash
make notebooks        # JupyterLab on notebooks/ with the "selfimprove (.venv)" kernel
```

They share one store in `.selfimprove/notebooks/` (notebook 00 starts it fresh), separate from the
CLI's. A live pass on the default models (Claude Haiku 5.5 agent, Claude Opus 5.5 judges) costs a
few dollars, most of it in notebooks 03 and 05.

## The skill

The `self-improving-agent` skill now lives at the repo root, in [`skills/`](../skills/README.md):
it teaches a coding agent to apply this harness, its workflows and the OpenClaw connector.

## Layout

```
selfimprove/
  config.py        .env -> settings            agents/       adapters + the traced entry point
  instructions.py  Prompt Registry, promote, rollback
  traces.py        queries, assessment conventions
  signals.py       the verifier from criteria.yaml + theme judges
  store.py         dataset, holdout, checkpoints, the human queue
  improvers.py     failure report, analyst, optimizers, coding agent
  approver.py      approval checks                gate.py      scorecards + promotion rules
  feedback.py      human labels, review queues    workflows/   the scheduled jobs
  cli.py           `selfimprove <command>`
examples/          the support agent, its instructions, criteria, golden cases, reviewers, Hermes
notebooks/         the walkthrough
tests/             offline loop (CI) + the regression suite (SI_REGRESSION=1)
```

## Notes

* Requires MLflow 3.17.2 or later (TypeSafe/Jev judges). `EvaluationResult.passed` treats
  expectation columns as assertions, so the gate and the regression suite check scorer columns
  only (`gate.scorer_failures`).
* The regex answer key is the expectation `key_facts`, not `expected_facts`: MLflow's
  Correctness judge reads `expected_facts` and refuses rows that also have `expected_response`.
  The scorer that checks it is named `has_key_facts`: in `evaluate()`'s `result_df`, a feedback
  with the same name as an expectation comes back as NaN, so the gate would skip those rows.
* litellm 1.106.1 imports `mcp` without depending on it, so the harness stays below 1.106. Some
  litellm versions can deadlock in a finalizer at interpreter exit; the CLI flushes MLflow's
  trace queue and exits immediately, so scheduled jobs never hang after finishing.
* Code `@scorer`s cannot be registered for server-side sampling on open-source MLflow; `score`
  runs them, and `online` registers only the LLM judges.
