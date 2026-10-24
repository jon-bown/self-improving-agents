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

## Three components

The repo holds three independent components. Each folder has its own README, `.env`, install
step, and tests, and none imports another.

| | what it is | start here |
|---|---|---|
| **[`connectors/openclaw-mlflow`](connectors/openclaw-mlflow/README.md)** | The OpenClaw → MLflow connector: MLflow's `@mlflow/mlflow-openclaw` tracing plugin, with the fixes it needs on OpenClaw 2026.10. Every OpenClaw run becomes an MLflow trace (`AGENT` → `LLM` / `TOOL` spans, tokens, cost, session). Works with any OpenClaw profile. | `./install.sh --tracking-uri http://localhost:5000 --experiment openclaw` |
| **[`harness/`](harness/README.md)** | The MLflow self-improving harness, `selfimprove`: the blog series' framework (signal → store → update, with triage, approval and a gate) for an agent you bring (a Python function, an HTTP service, a CLI, OpenClaw or Hermes), configured from `.env`. Ships workflows, walkthrough notebooks, a `self-improving-agent` skill, and the talk's four-loop demo in [`harness/session`](harness/session/README.md). | `make install doctor` |
| **[`sia/`](sia/README.md)** | The level-based self-improving agent (SIA): the tutorial's Acme Cloud support agent on OpenClaw, built up over six levels from tracing to an unattended loop. Set your provider and API key in `.env`, and every level runs with `make`. | `make install server`, then `make level0 level1 level2` |

How they fit together:

* The **connector** is the tracing layer for OpenClaw. The harness installs it into its own
  isolated OpenClaw profile (`SI_OPENCLAW_MLFLOW=1`). The talk demo relies on it, because
  whether the agent read a policy file is only visible in OpenClaw's own spans. SIA can use it
  too (see its README).
* The **harness** and **SIA** are two builds of the same loop. SIA is the step-by-step
  tutorial with one fixed agent. The harness is the reusable version that follows the blog
  posts and works with any agent.

## Shared by all three

```
README.md            this map
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
