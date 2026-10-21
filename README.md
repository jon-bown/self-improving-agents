# Build Your Own Self-Improving Agent with OpenClaw and MLflow

Companion repo for the ODSC West 2026 tutorial and the *Self-Improvement & MLflow* blog series.

An [OpenClaw](https://github.com/openclaw/openclaw) support agent for a fictional product, Acme Cloud, goes from
"a person reads traces and edits a prompt" to "the loop notices a problem, fixes it, proves the fix,
and ships it unattended". Along the way MLflow provides the traces, the signal, the store, and the gate.

```
        ┌────────── signal ──────────┐      ┌──── store ────┐      ┌───── update ─────┐
traces ─┤ human feedback / expectations├─────▶ eval dataset  ├─────▶ new AGENTS.md    ├──▶ gate ──▶ promote
        │ code scorers · LLM judges    │      │ (grows from   │      │ version in the   │    (dataset +   (alias
        └──────────────────────────────┘      │  failures)    │      │ Prompt Registry) │     holdout)    move)
                                              └───────────────┘      └──────────────────┘
```

## The levels

Every level runs the same loop. Two things change: **who writes the candidate** and **who approves it**.

| Level | Folder | Who writes the change | Who promotes | MLflow pieces introduced |
|---|---|---|---|---|
| 0 | `level0_instrument` | – | – | Prompt Registry, tracing, `update_current_trace` tags |
| 1 | `level1_signal` | – | – | `log_feedback` / `log_expectation`, label schemas + review queues, `@scorer`, `Guidelines`, `make_judge`, `evaluate()` on existing traces |
| 2 | `level2_manual_loop` | a person | a person | evaluation datasets, side-by-side `evaluate()`, `@mlflow.test` regression suite, alias promotion |
| 3 | `level3_agent_assisted` | an LLM trace analyst | a person (reviews a proposal) | scorer registry (theme judges grow with evidence), `mlflow agent setup` skills |
| 4 | `level4_optimizer` | GEPA (`optimize_prompts`) | a person | judge alignment (`judge.align`, SIMBA / MemAlign), `optimize_prompts` |
| 5 | `level5_autonomous` | GEPA, the analyst, **or OpenClaw editing itself** | the gate | online scoring (`register().start()`), loop runs as MLflow runs, rollback |

The update target throughout is the agent's `AGENTS.md`. OpenClaw's behavior is defined by plain files in its
workspace. We keep `AGENTS.md` in the MLflow Prompt Registry and render the chosen version into a fresh
workspace right before the agent runs (`src/sia/workspace.py`). Every change is then a numbered version with a
commit message, and every rollback is an alias move. The policy documents in `knowledge/` sit next to it in the
workspace and stay fixed.

## Quickstart

```bash
git clone <this repo> && cd self-improving-agent
make install                      # uv venv + pip install -e ".[dev]"
cp .env.example .env              # pick SIA_PROVIDER and add that provider's API key
make server                       # MLflow on http://127.0.0.1:5000 (leave running)
```

In a second terminal:

```bash
make level0 level1 level2         # instrument, collect signal, first manual loop
make drift level3                 # new kind of traffic; analyst writes proposals/<ts>/
python levels/level3_agent_assisted/02_accept.py proposals/<ts> --promote
make level4                       # align the judge, run GEPA, gate
make reset setup level5           # start over and let the loop do all of it by itself
```

**No API key or OpenClaw yet?** `make rehearse` runs every level with a rule-based stand-in agent and code-only
scorers in a throwaway store. It takes about three minutes. It's how CI checks the loop mechanics, and it's a
dry run for the workshop.

### Choosing an LLM provider

Every model is named in MLflow's URI format, `<provider>:/<model>`. MLflow's judges, GEPA and judge alignment
take these URIs directly. The repo's own model calls (the `direct` runtime and the trace analyst) go through
[LiteLLM](https://docs.litellm.ai/), so any provider it supports works. For the most common ones, one variable
picks a preset (`src/sia/llm.py`):

| `SIA_PROVIDER` | agent model | judge / analyst / optimizer | key | judge alignment |
|---|---|---|---|---|
| `anthropic` (default) | `claude-haiku-4-5` | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` | SIMBA |
| `openai` | `gpt-5.4-mini` | `gpt-5.5` | `OPENAI_API_KEY` | MemAlign |
| `gemini` | `gemini-3.5-flash` | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | MemAlign |
| `mistral` | `mistral-small-latest` | `mistral-large-latest` | `MISTRAL_API_KEY` | MemAlign |
| `groq` | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | SIMBA |
| `ollama` | `gpt-oss:20b` | `gpt-oss:20b` | none (runs locally) | SIMBA |

To override either model, or to mix providers (for example, a local agent graded by a hosted judge):

```bash
SIA_AGENT_MODEL=ollama:/gpt-oss:20b
MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=openai:/gpt-5.5
```

The OpenClaw runtime gets the same model, translated to OpenClaw's naming (`gemini:/…` → `google/…`).
OpenClaw reads the provider key from the environment. The judge model needs to handle tool calling and
long structured output reasonably well. A small local model may make a weak judge; the Level 1 agreement check
is where that would show up.

### Agent runtimes (`SIA_BACKEND`)

| Backend | What runs | Use for |
|---|---|---|
| `openclaw` (default) | `openclaw agent --local --json` against an **isolated** profile in `.openclaw-state/`. Your `~/.openclaw` is never touched. | the real demo |
| `direct` | a small tool-use loop (any provider, via LiteLLM) over the same workspace files (`list_files`, `read_file`) | GEPA and CI, where the agent is called dozens of times |
| `offline` | a deterministic simulator that follows a few instructions in `AGENTS.md` | rehearsals and tests without keys |

OpenClaw needs Node and `npm i -g openclaw`. `make setup` writes the isolated config and validates it. Optional:
`python levels/level0_instrument/01_setup.py --mlflow-plugin` also traces each OpenClaw run from the inside:
prompt, every model request, every tool call, and token usage. The trace is tagged with the
`acme_support_agent` trace that ran it. This uses the dependency-free plugin in `connectors/openclaw-mlflow`.
`--mlflow-otel` instead streams OpenClaw's diagnostic spans via the `diagnostics-otel` plugin.

### Trace your own agent

The same connectors work for your everyday OpenClaw or [Hermes Agent](https://github.com/NousResearch/hermes-agent):

```bash
make server                       # MLflow on :5000
make connect-openclaw             # or: python connectors/connect.py openclaw | hermes | status
```

Each OpenClaw run or Hermes turn becomes an MLflow trace, grouped by session. See
[`connectors/README.md`](connectors/README.md).

## Walkthrough

### Level 0: instrument
`01_setup.py` registers `scenario/workspace_v1/AGENTS.md` as v1 and points the `production` alias at it.
`02_first_traces.py` sends a week of questions. Each one becomes a trace tagged with `topic`, `backend`,
`prompt_version`, and `source=production`. Evaluation and optimizer traffic is tagged `source=eval`, so it can
never be mistaken for real signal. v1 has three seeded problems: stale facts (30-day refunds), a mandatory
Terms-of-Service disclaimer, and "give your best guess".

### Level 1: signal
* `01_human_feedback.py` has three simulated reviewers label every trace: `quality` feedback, an
  `expected_response`, and a `key_facts` answer key. They grade with slightly different rubrics on purpose,
  because that is what real reviewer pools do. `--live` creates a label schema and review queue so people can
  label in the MLflow UI instead.
* `02_automated_signal.py` runs code scorers and LLM judges on the existing traces (`evaluate()` with no
  `predict_fn`) and prints failure rate by topic.
* `03_signal_quality.py` checks the verifiers before trusting them: judge-vs-human agreement, reviewer
  strictness, and failure themes. *An optimizer is only as good as its verifier.*

### Level 2: the manual loop
Build the dataset from human-failed traces, register a hand-written fix
(`scenario/fixes/level2_manual_AGENTS.md`) as v2 without promoting it, then run the gate:

```
scorer                    train/baseline   train/candidate  holdout/baseline holdout/candidate
has_key_facts                         0%              100%                0%              100%
no_disclaimer_opener                  0%              100%                0%              100%
no_stale_policy                      64%              100%               83%              100%
```

The gate rules live in `src/sia/gate.py` and every later level reuses them:
1. must-pass scorers (`no_stale_policy`) are at 100%,
2. the candidate beats production on the dataset,
3. the candidate is not worse on the **holdout**, which no optimizer ever sees.

`make test` runs the same dataset as a pytest suite with `@mlflow.test`, and `.github/workflows/agent-regression.yml`
runs it on every PR.

### Level 3: agent-assisted
With ten reviewers and a week of traffic, the bottleneck is reading. `01_propose.py` sends every failing trace,
with its rationales, to an LLM analyst. The analyst returns up to five themes, one criterion per theme, and a
proposed `AGENTS.md`, and writes them all to `proposals/<ts>/` for review. `02_accept.py` registers the candidate
and **registers each theme criterion as a Guidelines judge** in MLflow's scorer registry (`theme_<slug>`), so the
test suite grows with the evidence. It then adds the representative traces to the dataset and runs the gate.
Prefer Claude Code? `mlflow agent setup --agent claude` installs MLflow's trace-analysis skills; give it the same job.

### Level 4: optimizer
* `01_align_judge.py` aligns the `quality` judge to the reviewers' labels (`judge.align`). It uses SIMBA by
  when the provider has no embedding model (Anthropic, Groq), and MemAlign otherwise. Agreement is measured on held-out labels, and the
  aligned judge is registered only if it helped.
* `02_optimize.py` runs `mlflow.genai.optimize_prompts` with `GepaPromptOptimizer`. Its scorers are the code
  checks, the guideline and theme judges, and the aligned judge. Run it with `SIA_BACKEND=direct`.
* `03_gate.py` shows the diff and runs the gate. Read the diff: optimizers can win by pleasing the judge.

### Level 5: autonomous
`02_run_loop.py` runs `sia.loop.cycle()`. Each cycle is one MLflow run:

1. **Signal:** read the production traces that arrived since the last cycle, and score any that are unscored.
2. **Verifier health:** if the judge's agreement with new human labels drops below 75%, re-align it, or **halt and ask for a person**.
3. **Store:** add human-labeled failures to the dataset. Failures only a scorer caught wait for a human label.
4. **Update:** `--improver gepa | analyst | openclaw`. With `openclaw`, `openclaw agent exec` edits a copy of its own workspace, driven by the failure report.
5. **Gate:** the same rules as Level 2, plus a bloat guard.
6. **Ship:** move the alias, and record the failure rate it replaced. A later cycle rolls back if live results are worse.

```
======== cycle 1 ========   production v1: failure rate 62%  → proposed v2 → gate 35%→100% → PROMOTED
======== cycle 2 ========   drift traffic starts (new support_process policy)   → healthy (6%)
======== cycle 3 ========   production v2: failure rate 19%  → proposed v3 → gate 93%→100% → PROMOTED
======== cycle 4 ========   production v3: failure rate 0%   → healthy
```

`01_monitor.py` turns on server-side online scoring, so judges label production traffic as it arrives. It needs
`mlflow server` with a database backend. Custom `@scorer` code can't be registered on OSS MLflow, so the loop runs
the code checks itself.

Humans stay in the loop where they can't be replaced: labeling a sample of traffic. Those labels are the ground
truth that keeps the judge honest and the dataset growing.

## Repository layout

```
scenario/            knowledge base, flawed v1 AGENTS.md, the Level 2 fix, cases.yaml (traffic / holdout / drift)
src/sia/
  agent.py           traced entry point + openclaw / direct / offline runtimes, prompt <-> workspace
  signals.py         code scorers, Guidelines judges, the alignable "quality" judge, theme judges
  data.py            labeled traces -> evaluation dataset; holdout rows
  gate.py            scorecards, promotion rules, promote / rollback
  improvers.py       failure report, LLM analyst, OpenClaw self-edit, judge alignment, GEPA
  loop.py            one autonomous cycle
  reviewers.py       simulated reviewers (with deliberate disagreement)
  traffic.py         simulated production traffic + labeling
levels/              one folder per level; each script's docstring explains the step
connectors/          OpenClaw + Hermes plugins that trace any run to MLflow over OTLP; connect.py wires them up
tests/unit           offline smoke tests (CI, no keys)
tests/agent          the regression gate (@mlflow.test over the dataset + holdout)
```

## Notes on MLflow 3.16.1

We hit three issues while building this. The workarounds are in the code:
* `EvaluationResult.passed` treats expectation columns (such as `expected_response/value`) as assertions, so
  `assert result.passed` fails on any dataset row that has an expected response. `sia.gate.scorer_failures()`
  checks only the scorers' own columns.
* Usage telemetry can deadlock on a lazy import when `evaluate()` runs scorers in worker threads. `import sia`
  sets `MLFLOW_DISABLE_TELEMETRY=true`.
* Registering the MLflow pytest plugin in a root `conftest.py` hung subprocess-based tests. The agent suite
  enables it with `pytest -p mlflow.pytest.plugin tests/agent`.

Boolean feedback is stored as text. Filter with `feedback.quality = 'false'`, not `= False`.
