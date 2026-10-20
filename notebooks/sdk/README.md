# MLflow on top of OpenClaw and Hermes, from Python

Two real agent runtimes, driven from Python, with MLflow layered on top. The layers: tracing, the Prompt Registry
as the source of the agents' instructions, evaluation, human and LLM feedback, and a gated improvement loop.

* **OpenClaw**: through its gateway, with the [`openclaw-sdk`](https://masteryodaa.github.io/openclaw-sdk/) Python
  client.
* **Hermes Agent**: in-process, through its `run_agent.AIAgent` class.

| Notebook | What it shows |
|---|---|
| [01 · OpenClaw from Python](01_openclaw_from_python.ipynb) | an isolated gateway, the SDK client, one turn, its event stream, workspace files over the gateway, per-turn usage and cost from `chat.history` |
| [02 · MLflow on top of OpenClaw](02_mlflow_on_openclaw.ipynb) | build the tracer step by step: root span → TOOL spans from SDK callbacks → LLM spans with tokens and cost; AGENTS.md from the Prompt Registry via `set_file`; prompt-to-trace links; code, human, and search |
| [03 · Hermes Agent from Python](03_hermes_from_python.ipynb) | plain `AIAgent`, then MLflow TOOL spans from its callbacks, then model-call spans; the same trace shape as OpenClaw; both runtimes side by side |
| [04 · Evaluate both agents](04_evaluate_both_agents.ipynb) | an MLflow evaluation dataset, code scorers that read TOOL spans, Claude Opus 5.5 judges, `mlflow.genai.evaluate` per runtime, a scorecard, human labels |
| [05 · Improve and gate](05_improve_and_gate.ipynb) | failure report from MLflow → Claude proposes AGENTS.md v(n+1) → evaluate on held-out questions on both runtimes → promote via the `@production` alias, or don't → rollback |

## Run them

```bash
make sdk-notebooks      # installs the hermes + openclaw-sdk + notebooks extras, opens JupyterLab on notebooks/sdk
make sdk-notebooks-ui   # MLflow UI for this series (notebooks/.mlflow/sdk.db), on http://127.0.0.1:5002
make sdk-gateway-stop   # stop the series' OpenClaw gateway (notebook 05 also stops it)
```

Needs:
* OpenClaw 2026.9 or later on your PATH (`npm i -g openclaw`).
* `ANTHROPIC_API_KEY` in `.env`. Claude Haiku 5.5 is the agent; Claude Opus 5.5 is the judge and the analyst.

Run the notebooks in order: each builds on what the previous one logged. Notebook 01 starts the series over: it
deletes `sdk.db` and puts the flawed v1 `AGENTS.md` back in both workspaces. A full pass costs a few dollars of API
usage. 04 and 05 do most of it, and each says its budget up front.

## What's isolated

| | where | never touches |
|---|---|---|
| MLflow | `notebooks/.mlflow/sdk.db` | your tracking server |
| OpenClaw gateway | `notebooks/.mlflow/sdk/gateway/`: state, config, token, device key, log; port **19556**, loopback, token auth, LAN discovery off | `~/.openclaw`, your gateway on 18789 |
| OpenClaw workspace | `notebooks/.mlflow/sdk/openclaw/` | |
| Hermes | home under `notebooks/.mlflow/hermes-sdk/homes/hermes/`; workspace `notebooks/.mlflow/sdk/hermes/`, made its own git root | `~/.hermes`; this repo (see below) |
| Claude credentials | `ANTHROPIC_API_KEY` only | Claude Code's saved login (Hermes would otherwise fall back to it) |

## Two compatibility shims, and why

* **openclaw-sdk 2.1.0 ↔ OpenClaw 2026.9.** The SDK speaks gateway protocol v3; current gateways speak v4.
  `sia.openclaw_client.SDKGateway` changes only the `connect` handshake. It sends:
  * protocol 4;
  * the gateway's challenge timestamp in the signature;
  * the `tool-events` capability (without it, tool calls aren't streamed);
  * an explicit token and device key, instead of `~/.openclaw`.

  The SDK's `on_tool_call` / `on_tool_result` hooks expect v3's tool phases, so the MLflow tracer reads tool events
  from `on_stream_event` instead.
* **Hermes 0.19.0 on Claude.** Hermes treats newer Console API keys (`sk-ant-usr-…`) as Claude subscription
  tokens, and it falls back to Claude Code's saved login. `sia.hermes_runtime` makes it use `ANTHROPIC_API_KEY`,
  as an API key, and nothing else.

* **Hermes' idea of "the project".** Hermes tells the model its *Workspace Root* is the enclosing git repository.
  For a workspace inside this repo, that's the repo, including `scenario/`, which holds the reviewers' answer key.
  Hermes' file tools have no hard sandbox. `hermes_runtime.fence()` makes each workspace its own git root, so the
  model only ever hears about the workspace.
* **Hermes in Jupyter.** `AIAgent` wraps `sys.stdout`/`sys.stderr` in a broken-pipe guard whose `write` is
  read-only, which breaks IPython's output capture for every later cell. Inside a kernel, `sia.hermes_runtime`
  skips that wrapper.

These are pinned in `pyproject.toml` (`openclaw-sdk==2.1.0`, `hermes-agent==0.19.0`) and covered by
`tests/unit/test_openclaw_client.py` and `tests/unit/test_hermes_runtime.py`.

## The code behind the notebooks

* `src/sia/openclaw_client.py`:
  * `Gateway`, the isolated gateway process;
  * `SDKGateway` / `connect()`, the openclaw-sdk client;
  * `MLflowTracer` and `add_llm_spans()`, MLflow on top;
  * `OpenClawAgent`, a synchronous handle with `ask()`, `get_agents_md()`, and `set_agents_md()`.
* `src/sia/hermes_runtime.py`:
  * `traced_agent_class()`, LLM spans around Hermes' model calls;
  * `run_hermes()`;
  * `HermesAgent`, the same interface as `OpenClawAgent`.
* `src/sia/signals.py`, `improvers.py`, `reviewers.py`, `scenario.py`: the scorers, the analyst, the simulated
  reviewers, and the Acme Cloud scenario shared with the rest of the tutorial.
