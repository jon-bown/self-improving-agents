# Self-Improving Agents with MLflow & OpenClaw

Companion repo for my talk at **ODSC AI West 2026**, *Build Your Own Self-Improving Agent with
OpenClaw and MLflow*, and for the [*Self-Improvement & MLflow* blog series](#the-blog-series).

Every part of this repo runs the same loop, with MLflow providing the traces, the signal, the
store, and the gate:

```
        ┌────────── signal ──────────┐      ┌──── store ────┐      ┌───── update ─────┐
traces ─┤ human feedback / expectations├─────▶ eval dataset  ├─────▶ new AGENTS.md    ├──▶ gate ──▶ promote
        │ code scorers · LLM judges    │      │ (grows from   │      │ version in the   │    (dataset +   (alias
        └──────────────────────────────┘      │  failures)    │      │ Prompt Registry) │     holdout)    move)
                                              └───────────────┘      └──────────────────┘
```

## The blog series

The harness follows these posts on ODSC's blog:

1. [Self-Improvement and Self-Evolving with MLflow](https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/): the vocabulary, and the MLflow components behind signal, store, and update.
2. [Self-Improvement & MLflow: Part 2](https://opendatascience.com/self-improvement-mlflow-part-2/): the first loop by hand, and where it stops scaling.
3. [Self-Improvement & MLflow: Automate via CI/CD](https://opendatascience.com/self-improvement-mlflow-part-3/): automating the loop, and where to stop.

## Four components

The repo holds four independent components. Each folder has its own README, install step and
tests, and none imports another.

| | what it is | start here |
|---|---|---|
| **[`connectors/openclaw-mlflow`](connectors/openclaw-mlflow/README.md)** | The OpenClaw → MLflow connector: MLflow's `@mlflow/mlflow-openclaw` tracing plugin, with the fixes it needs on OpenClaw 2026.10. Every OpenClaw run becomes an MLflow trace (`AGENT` → `LLM` / `TOOL` spans, tokens, session). Works with any OpenClaw profile. | `./install.sh --tracking-uri http://localhost:5000 --experiment openclaw` |
| **[`harness/`](harness/README.md)** | The MLflow self-improving harness, `selfimprove`: the blog series' framework (signal → store → update, with triage, approval and a gate) for an agent you bring (a Python function, an HTTP service, a CLI, OpenClaw or Hermes), configured from `.env`. Ships workflows, walkthrough notebooks, and the talk's four-loop demo in [`harness/session`](harness/session/README.md). | `make install doctor` |
| **[`sia/`](sia/README.md)** | The level-based self-improving agent (SIA): the tutorial's Acme Cloud support agent on OpenClaw, built up over six levels from tracing to an unattended loop. Set your provider and API key in `.env`, and every level runs with `make`. | `make install server`, then `make level0 level1 level2` |
| **[`skills/`](skills/README.md)** | Two agent skills. `self-improving-agent`: everything above, packaged for a coding agent. It applies the framework with the harness, its workflows and the connector, writes verifiers, handles gate rejections, and keeps to the rules (the proposer never approves, the holdout stays hidden). `self-improving-with-mlflow`: loaded by OpenClaw itself, so the agent can check its standing, pass on corrections, recall confirmed lessons and propose changes to itself, all through the harness's gate. | `cp -r skills/self-improving-agent ~/.claude/skills/`; `selfimprove self setup --workspace <dir>` |

How they fit together:

* The **connector** is the tracing layer for OpenClaw. The harness installs it into its own
  isolated OpenClaw profile (`SI_OPENCLAW_MLFLOW=1`). The talk demo relies on it, because
  whether the agent read a policy file is only visible in OpenClaw's own spans. SIA can use it
  too (see its README).
* The **skills** are how agents use the other three. `self-improving-agent` is for a coding agent:
  it knows the harness's commands and workflows, the connector's setup, and the verifier
  patterns from the talk's four loops. `self-improving-with-mlflow` is for OpenClaw at runtime: it
  runs the harness's `selfimprove self` commands about itself, and the harness keeps the gate.
* The **harness** and **SIA** are two builds of the same loop. SIA is the step-by-step
  tutorial with one fixed agent. The harness is the reusable version that follows the blog
  posts and works with any agent.

## Demos

| Demo | Length | What it shows |
|---|---|---|
| [`connectors/openclaw-mlflow/examples/live-demo.md`](connectors/openclaw-mlflow/examples/live-demo.md) | 5 min, live | ask a local OpenClaw a question, catch the made-up answer in its MLflow trace, fix `AGENTS.md` in MLflow (a new prompt version, then an alias move), ask again |
| [`harness/session/`](harness/session/README.md) | 4 loops, precomputed | the self-improvement loop run four times on OpenClaw, from a person doing every step to `cycle` promoting through the gate |

## Architecture

* [`docs/architecture.md`](docs/architecture.md): MLflow and the MLflow server, and how one OpenClaw
  answer (through the connector) and one Hermes answer (in process) become traces.
* [`harness/README.md`](harness/README.md#architecture): the harness, and what one `cycle` does.
* [`sia/README.md`](sia/README.md#architecture): the levels, the `sia` package and its runtimes.

## Shared by all four

```
README.md            this map
docs/                architecture.md: MLflow, the server, OpenClaw and Hermes
.agents/skills/      MLflow's agent skills for coding agents (trace analysis, evaluation, scorers);
                     skills-lock.json pins them
.github/workflows/   CI, one file per job, prefixed by component:
                       openclaw-mlflow.yml               connector: build + unit tests
                       selfimprove-offline.yml           harness: offline loop test
                       selfimprove-regression.yml        harness: dataset as @mlflow.test
                       selfimprove-loop.yml              harness: the scheduled loop
                       sia-offline-smoke.yml             SIA: offline smoke tests
                       sia-agent-regression.yml          SIA: the regression gate
```

Personal agents are someone's real setup, so every component keeps its OpenClaw (and Hermes)
state inside its own folder. None of them touches `~/.openclaw`, `~/.hermes`, or a gateway
you already run.

## Requirements

| Tool | Version | Needed for | Install |
|---|---|---|---|
| Python | 3.11 | harness, SIA, blog code | `uv python install 3.11` |
| [uv](https://docs.astral.sh/uv/) | latest | every Python venv (`make install`) | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `make`, `git` | any | running the levels and workflows | preinstalled on macOS (`xcode-select --install`) and most Linux |
| Node.js + npm | 22+ | OpenClaw and the connector | `brew install node@22`, or [nvm](https://github.com/nvm-sh/nvm): `nvm install 22` |
| OpenClaw | 2026.10+ | the OpenClaw runtime, the connector, the talk demo | see [Installing OpenClaw](#installing-openclaw) |
| MLflow | 3.17.2+ | everything | installed for you by `make install` (`mlflow[genai]`) |
| A model provider key | | live runs (not the offline modes or tests) | `ANTHROPIC_API_KEY`, or another provider, in each component's `.env` |

The offline backends (`SIA_BACKEND=offline`, the harness's offline workflow) and the unit tests run
without OpenClaw or an API key.

### Installing OpenClaw

OpenClaw ships as an npm package. With Node 22+ on your `PATH`:

```bash
node --version             # v22.x or later
npm install -g openclaw
openclaw --version         # 2026.10 or later
```

If `npm install -g` fails with a permissions error, use a Node from Homebrew or nvm rather than
`sudo`. To upgrade an older OpenClaw, run `npm install -g openclaw@latest`.

You don't need to run `openclaw onboard` or start a gateway. Each component creates its own
isolated OpenClaw profile (`.selfimprove/openclaw/` in the harness, `.openclaw-state/` in SIA) and
leaves `~/.openclaw` alone. If `openclaw` isn't on your `PATH`, point the connector at it with
`OPENCLAW_BIN=/path/to/openclaw`.

The connector and the live demo also need an MLflow **server**, because traces are exported over
HTTP. Leave one running in its own terminal (on a Mac, port 5000 is usually taken by AirPlay):

```bash
mlflow server --port 5070
```

## Platform support

| Platform | Status |
|---|---|
| macOS, Apple Silicon (arm64) | supported and recommended: the harness and SIA test suites pass |
| macOS, Intel (x86_64), or an Intel Python under Rosetta on Apple Silicon | supported: the harness and SIA install a compatible `cryptography` automatically, and their tests pass |
| Linux, x86_64 | supported: CI runs on `ubuntu-latest` |
| Linux, arm64 | not tested |
| Windows | not tested; use WSL2 |

Everything needs Python 3.11 through [uv](https://docs.astral.sh/uv/). The connector needs Node 22+;
it builds to a single JavaScript file, so it runs the same on either architecture.

### Intel (x86_64) Macs, and Intel Pythons on Apple Silicon

`cryptography` 49 and 50 publish an Intel macOS wheel that fails to load, with
`symbol not found in flat namespace` (`_OSSL_get_max_threads`, `_BIO_ADDR_free`). MLflow imports it as
soon as it opens a tracking store, so every command fails. Both `harness/pyproject.toml` and
`sia/pyproject.toml` therefore pin `cryptography<49` on Intel macOS only (48.0.1 works). Apple Silicon
and Linux are unaffected and get the latest.

**Check what you're running:**

```bash
uname -m                                                    # the machine: arm64 or x86_64
file "$(command -v uv)"                                     # the uv build: arm64 or x86_64
.venv/bin/python -c "import platform; print(platform.machine())"   # the venv's Python
```

A venv's Python follows the `uv` build, not the machine. An Intel `uv` on Apple Silicon creates Intel
venvs that run under Rosetta: they work, but more slowly.

**On an Intel Mac:** nothing extra. `make install` (or the quickstart's `uv venv` + `uv pip install`)
picks `cryptography` 48.0.1 for you.

**On Apple Silicon:** use a native `uv`, so your venvs are arm64:

```bash
brew install uv          # with arm64 Homebrew in /opt/homebrew
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then delete any existing `.venv` and run `make install` again in `harness/` and `sia/`.

**To build an Intel venv on purpose** (for example, to test Intel support from an Apple Silicon Mac),
ask uv for an x86_64 Python:

```bash
uv venv -p cpython-3.11-macos-x86_64-none .venv && uv pip install -p .venv -e ".[dev]"
```

**A venv made before the pin**, or one where another package forces a newer `cryptography`, still
fails. Fix it in place:

```bash
uv pip install -p .venv "cryptography<49"
```
