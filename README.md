# Self-Improving Agents with MLflow & OpenClaw

Companion repo for my talk at **ODSC AI West 2026**, *Build Your Own Self-Improving Agent with
OpenClaw and MLflow*, and for the [*Self-Improvement & MLflow* blog series](#the-blog-series).

## Setup

You need **macOS or Linux** (on Windows, use WSL2), plus three tools. You don't need to install Python,
OpenClaw or MLflow yourself: `make setup` handles them.

| Tool | Check | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (or `brew install uv`) |
| Node.js 22+ | `node --version` | `brew install node`, or [nvm](https://github.com/nvm-sh/nvm): `nvm install --lts` |
| `git`, `make` | `make --version` | preinstalled on Linux; on macOS, `xcode-select --install` |

Then clone the repo and run one command:

```bash
git clone https://github.com/jon-bown/self-improving-agents.git
cd self-improving-agents
make setup
```

`make setup` takes a couple of minutes the first time. It's safe to re-run, and it skips anything
already done. Everything it installs stays inside this folder:

| What | Where |
|---|---|
| One Python environment for `harness/`, `sia/` and `blog/`, from the lockfile `uv.lock`. uv picks a Python ≥ 3.10 that's native to your machine (arm64 on Apple Silicon) and downloads one if needed. | `.venv/` |
| A Jupyter kernel, **Python (self-improving-agents)**, which every notebook in the repo uses | `.venv/share/jupyter/` |
| OpenClaw, installed with npm into the venv: no global install, no `sudo` | `.venv/bin/openclaw` |
| The OpenClaw → MLflow connector, built and ready to install into an OpenClaw profile | `connectors/openclaw-mlflow/*.tgz` |
| Config files, created from the examples if missing | `harness/.env`, `sia/.env` |

It finishes by checking the result. Run `make check` at any time to repeat the check.

**Add an API key** for live runs: set `ANTHROPIC_API_KEY=` in `harness/.env` (the demo and the harness
read it there) and in `sia/.env`. To use another provider, see the comments in those files. Without a
key, the tests, the offline modes and most notebooks still run.

### After setup

**See OpenClaw traced in MLflow.** Use two terminals in the repo root:

```bash
make demo-server     # terminal 1: a separate MLflow server for the demo on http://127.0.0.1:5050; leave it running
make demo            # terminal 2: set up OpenClaw, install the connector, ask one question, list its trace
```

The first `make demo` creates an isolated OpenClaw profile in `~/odsc-5min`, installs the connector
into it, and points the connector at the demo server. It also registers the agent's `AGENTS.md` as a
prompt in MLflow, then asks *"Do you have an uptime SLA?"*. Open http://127.0.0.1:5050, go to experiment
**openclaw-demo** and then **Traces**, and you'll see the run: the agent span, its LLM and tool calls,
and the policy files it read. Ask more questions with `make demo-ask Q="How do I reset my password?"`.
The full five-minute talk demo, in which the agent fixes itself, is in
[`live-demo.md`](connectors/openclaw-mlflow/examples/live-demo.md). `make demo-reset` starts over
(stop the server first).

**Open the notebooks.** `make notebooks` starts JupyterLab on the whole repo. The notebooks pick the
**Python (self-improving-agents)** kernel automatically. In VS Code or Cursor, choose `.venv/bin/python`
as the interpreter or kernel instead; `.vscode/settings.json` already points there.

**Use the command line.** Run `source .venv/bin/activate` to put `selfimprove`, `mlflow` and `openclaw`
on your `PATH`. The Makefiles inside `harness/` and `sia/` find the venv without activating it:

```bash
cd harness && make doctor               # the harness: check your agent, then see harness/README.md
cd sia && make server                   # SIA: its MLflow server; then, in another terminal, make level0 level1 level2
make test                               # every offline test suite (harness, SIA, connector), at the root
```

### Options, updates and starting over

```bash
UV_PYTHON=3.12 make setup               # a specific Python version (any >= 3.10)
OPENCLAW_VERSION=2026.10.5 make setup   # a specific OpenClaw, instead of the latest
SKIP_OPENCLAW=1 make setup              # Python and notebooks only, no Node needed
git pull && make setup                  # update: re-syncs the venv to the new lockfile
make clean setup                        # start over: rebuilds .venv, OpenClaw and the connector
```

| Problem | Fix |
|---|---|
| `make check` reports an Intel Python on Apple Silicon, or `cryptography` fails with `symbol not found in flat namespace` | `make clean setup`. Setup asks uv for a native arm64 Python, but a venv built by an older version of this repo may be Intel. |
| `openclaw: command not found` outside `make` | `source .venv/bin/activate`, or use `.venv/bin/openclaw`. To use another OpenClaw, set `OPENCLAW_BIN=/path/to/openclaw`. |
| A notebook says its kernel runs the wrong MLflow | Switch to the **Python (self-improving-agents)** kernel (or `.venv/bin/python` in an editor). |
| `make demo-server` says the port is in use | `make demo-server DEMO_PORT=5051`, then `make demo DEMO_PORT=5051`. On a Mac, port 5000 is AirPlay Receiver, so the repo avoids it for the demo. |

## The loop

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

The repo holds four independent components. Each folder has its own README and tests, and none
imports another. They share one environment, which `make setup` builds.

| | what it is | start here |
|---|---|---|
| **[`connectors/openclaw-mlflow`](connectors/openclaw-mlflow/README.md)** | The OpenClaw → MLflow connector: MLflow's `@mlflow/mlflow-openclaw` tracing plugin, with the fixes it needs on OpenClaw 2026.10. Every OpenClaw run becomes an MLflow trace (`AGENT` → `LLM` / `TOOL` spans, tokens, session). Works with any OpenClaw profile. | `make demo` (built by `make setup`); for your own profile, `./install.sh --tracking-uri http://127.0.0.1:5070 --experiment openclaw` |
| **[`harness/`](harness/README.md)** | The MLflow self-improving harness, `selfimprove`: the blog series' framework (signal → store → update, with triage, approval and a gate) for an agent you bring (a Python function, an HTTP service, a CLI, OpenClaw or Hermes), configured from `.env`. Ships workflows, walkthrough notebooks, and the talk's four-loop demo in [`harness/session`](harness/session/README.md). | `cd harness && make doctor` |
| **[`sia/`](sia/README.md)** | The level-based self-improving agent (SIA): the tutorial's Acme Cloud support agent on OpenClaw, built up over six levels from tracing to an unattended loop. Set your provider and API key in `.env`, and every level runs with `make`. | `cd sia && make server`, then `make level0 level1 level2` |
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
Makefile             setup, check, notebooks, test, and the connector demo (make help)
pyproject.toml       the uv workspace: one .venv for harness/ and sia/; uv.lock pins it
scripts/setup.sh     what make setup and make check run
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

## Platform support

| Platform | Status |
|---|---|
| macOS, Apple Silicon (arm64) | supported and recommended: `make setup` builds a native arm64 venv |
| macOS, Intel (x86_64) | supported: the harness and SIA pin a `cryptography` that loads there (`cryptography<49`, Intel macOS only) |
| Linux, x86_64 | supported: CI runs on `ubuntu-latest` |
| Linux, arm64 | not tested |
| Windows | not tested; use WSL2 |

Any Python from 3.10 up works; uv picks one and `uv.lock` pins the packages, so every machine gets the
same environment. The connector builds to a single JavaScript file, so it runs the same on any
architecture with Node 22+.

To build an Intel venv on purpose on Apple Silicon (for example, to test Intel support), run
`make clean && UV_PYTHON=cpython-macos-x86_64 make setup`.
