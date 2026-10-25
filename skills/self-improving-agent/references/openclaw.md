# OpenClaw: the agent, and the MLflow connector

OpenClaw's behavior is defined by plain files in its workspace (`AGENTS.md`, skills), which makes
it a good update target: the harness keeps `AGENTS.md` in the Prompt Registry and renders the
version under test into a fresh workspace before each run. The MLflow connector traces what
OpenClaw did inside each run.

## Contents

* OpenClaw with the harness
* The connector: what it records
* Installing the connector
* Linking harness traces to OpenClaw traces
* Sessions
* What it doesn't record
* Troubleshooting

## OpenClaw with the harness

```bash
SI_AGENT=openclaw
SI_AGENT_MODEL=anthropic:/claude-haiku-5-5   # MLflow URI; translated to OpenClaw's naming
SI_INSTRUCTIONS_FILE=agent/AGENTS.md
SI_WORKSPACE_DIR=knowledge                   # files the agent may read, copied next to AGENTS.md
SI_OPENCLAW_MLFLOW=1                         # install the connector on `selfimprove doctor`
MLFLOW_TRACKING_URI=http://127.0.0.1:5000    # the connector needs an MLflow server, not sqlite
```

Each call runs `openclaw agent --local --json` with a fresh session id, against an isolated
profile in `.selfimprove/openclaw/` (`OPENCLAW_STATE_DIR`). The user's `~/.openclaw` and any
gateway they run (port 18789) are never touched. The workspace is chosen per call with
`OPENCLAW_WORKSPACE_DIR`, which only works while the profile's config names no workspace
(`agents.defaults.workspace` wins over the variable), so the harness writes the profile without one.

Each OpenClaw call takes about 20 seconds, and the gate runs every dataset row for both production
and the candidate. Budget for it: GEPA, where every metric call is an OpenClaw run, is slow.

## The connector: what it records

`connectors/openclaw-mlflow` is MLflow's `@mlflow/mlflow-openclaw` plugin, fixed for OpenClaw
2026.10. Each OpenClaw run becomes one trace:

* an `openclaw_agent` AGENT root span with the conversation in chat format, which MLflow shows as
  messages;
* an `llm_call` LLM span;
* a TOOL span per tool call (`read`, ...) with its arguments and result: which file the agent read,
  and what it got back;
* sub-agent spans when the agent spawns one;
* token usage, including cache reads and writes, rolled up on the trace;
* model and provider, and `mlflow.trace.session` and `mlflow.trace.user` as trace metadata.

The TOOL spans are what make tool-span checks and grounded judges possible (`verifiers.md`).

## Installing the connector

One command builds it and installs it into whichever OpenClaw profile the CLI would use:

```bash
connectors/openclaw-mlflow/install.sh --tracking-uri http://localhost:5000 --experiment openclaw
```

* For another profile, set `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH` first. The harness does
  this for its isolated profile when `SI_OPENCLAW_MLFLOW=1`.
* It sets `plugins.entries.mlflow-openclaw.hooks.allowConversationAccess=true`. Without it,
  OpenClaw 2026.10 silently drops the hooks that create the root and LLM spans, and nothing is
  exported. It also adds the plugin to an existing `plugins.allow` list (never creates one).
* `openclaw mlflow status` shows the settings; `openclaw mlflow configure` changes them.
* Restart a running gateway afterwards; `--local` runs pick it up immediately.
* OpenClaw warns that it "can't verify where this plugin came from" on every start, because it was
  installed from a local archive. That's expected; the plugin still loads.

## Linking harness traces to OpenClaw traces

Each harness call produces two traces in the same experiment: the harness's own (tagged
`si_source`, with the prompt version and the judges' feedback) and OpenClaw's (the spans above).
The harness span records `openclaw.session_key`; OpenClaw's trace has the same value as
`mlflow.trace.session`:

```
metadata.`mlflow.trace.session` = 'agent:main:explicit:si-c9588f489907'
```

Connector traces have no `si_source` tag, so the loop never reads them as production traffic.

## Sessions

Runs with the same `--session-id` share a session key (`agent:main:explicit:<id>`) and continue
the conversation. In MLflow 3.17 the Sessions view is in the Traces tab, behind "Group by session"
(`/#/experiments/<id>/traces?groupBy=session`): one row per session, with the first input, the
last output and the total tokens. The harness uses a fresh session per call, so to show grouping,
run several turns with one session id.

## What it doesn't record

* **Cost.** Tokens arrive; cost doesn't, so MLflow's cost charts stay empty. Report tokens.
* **One span per model call.** OpenClaw reports a run's model calls as one `llm_call` span, even
  when the agent called tools in between.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| no traces at all | conversation access not granted (`openclaw mlflow status`); or `MLFLOW_TRACKING_URI` is `sqlite:///` (needs an HTTP server) |
| tool spans but no root or LLM span | the same conversation-access grant |
| `openclaw agent` runs take minutes | an old unbundled build of the plugin; rebuild with `install.sh` |
| "Another OpenClaw process is using your data" | two runs on one profile at once; the harness waits and retries, but don't run two jobs on one profile |
| the agent ignores the instructions under test | the profile names a workspace, which overrides `OPENCLAW_WORKSPACE_DIR`; remove it (the harness rewrites stale profiles) |
| `openclaw mlflow status` is slow | the CLI boots the whole plugin runtime; about two minutes on a laptop is normal |
