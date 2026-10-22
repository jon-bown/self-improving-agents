# MLflow tracing for OpenClaw

An OpenClaw plugin that logs every agent run to MLflow as a trace: a root `AGENT` span with
`LLM`, `TOOL` and sub-agent spans under it, plus token usage, cost, model and session.

<p align="center">
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/master/libs/typescript/integrations/openclaw/trace-screenshot.png" alt="An OpenClaw trace in the MLflow UI" width="700" />
</p>

This is the official `@mlflow/mlflow-openclaw` integration from MLflow's repository
(`libs/typescript/integrations/openclaw` at commit `568cd81`, 2026-10-21). It carries five fixes
that make it work on OpenClaw 2026.10 (listed [below](#what-is-fixed-here)). Plugin id, config
keys and CLI stay as they are upstream, so it can be swapped for the npm release once that has
the fixes.

## Quick start

You need Node 22+, `openclaw` on your `PATH`, and an MLflow **server**. The plugin exports over
HTTP, so a `sqlite:///` store won't work:

```bash
mlflow server --port 5000
```

Build the plugin, install it into your OpenClaw profile, and point it at an experiment (the
experiment is created if it doesn't exist):

```bash
./install.sh --tracking-uri http://localhost:5000 --experiment openclaw
```

Restart your gateway if one is running. Then run anything through OpenClaw and open the
experiment in the MLflow UI:

```bash
openclaw agent --local --agent main -m "Read AGENTS.md and summarize it"
```

`install.sh` targets whichever profile the `openclaw` CLI would use. To try it without touching
`~/.openclaw`, point it at a separate profile first:

```bash
OPENCLAW_STATE_DIR=/tmp/oc-test OPENCLAW_CONFIG_PATH=/tmp/oc-test/openclaw.json \
  ./install.sh --tracking-uri http://localhost:5000 --experiment openclaw-test
```

What it does, in order:

1. `npm ci` and `npm run build`, then `npm pack` into a `.tgz`. Add `--test` to also run the unit tests,
   or `--build-only` to stop at this step.
2. `openclaw plugins install --force --accept-capabilities <tgz>`. A local archive has no
   ClawHub or npm provenance, so OpenClaw asks for `--force`. For the same reason it logs
   "can't verify where this plugin came from" on every start. The warning is expected, and the
   plugin still loads.
3. Sets `plugins.entries.mlflow-openclaw` in the profile: `enabled`, `config.trackingUri`,
   `config.experimentId`, and `hooks.allowConversationAccess: true`. If the profile already
   has a `plugins.allow` list, it adds the plugin to it. All of this goes through one
   `openclaw config set --batch-json` call.

It takes about two minutes, most of it in `openclaw plugins install`.

### With the self-improving-agent harness

Set `SI_OPENCLAW_MLFLOW=1` (and an HTTP `MLFLOW_TRACKING_URI`) in `harness/.env`, then run
`selfimprove doctor`. It runs `install.sh` against the harness's isolated profile in
`.selfimprove/openclaw/`, never `~/.openclaw`. Each harness call then produces two traces in
the experiment:

* the harness's own trace (`si_source` tag, prompt version, the judges' feedback);
* OpenClaw's trace from this plugin, with the LLM and tool spans.

The harness span records `openclaw.session_key`. To find OpenClaw's trace for a harness trace,
search for ``metadata.`mlflow.trace.session` = '<that key>'``. Plugin traces have no `si_source` tag, so the loop never mistakes them
for production traffic.

## Configuration

| key (`plugins.entries.mlflow-openclaw.…`) | |
|---|---|
| `config.trackingUri` | MLflow server; falls back to `MLFLOW_TRACKING_URI` |
| `config.experimentId` | experiment **id** as a string; falls back to `MLFLOW_EXPERIMENT_ID` |
| `enabled` | `false` turns export off without uninstalling |
| `hooks.allowConversationAccess` | must be `true` on OpenClaw 2026.10+ (see fix 3) |

`openclaw mlflow configure` sets these interactively and `openclaw mlflow status` shows them.
If you set them by hand, quote the id as JSON: `openclaw config set
plugins.entries.mlflow-openclaw.config.experimentId '"1"'`. A bare `1` is parsed as a number
and fails validation.

## What is fixed here

MLflow main type-checks against a hand-written stub of the OpenClaw plugin API, not the real
`openclaw` package, so its CI doesn't catch these. Each was reproduced against OpenClaw 2026.10.5
with a live agent run.

1. **Renamed SDK module.** `openclaw/plugin-sdk/diagnostics-otel` is now
   `openclaw/plugin-sdk/diagnostic-runtime`. The old import fails, and the plugin never loads.
   (`src/service.ts`)
2. **Config API replaced.** `runtime.config.loadConfig()` and `writeConfigFile()` became
   `current()` and `replaceConfigFile()`. The plugin now reads through whichever API the host
   has, and clones `current()`, which is read-only. It can't save through the new
   `replaceConfigFile()`, though: that only works while the plugin's runtime is active, and the
   CLI retires it a few seconds into a command, long before anyone answers
   `openclaw mlflow configure`'s prompts ("runtime is no longer active"). `configure` now saves
   with `mutateConfigFile()` from `openclaw/plugin-sdk/config-mutation`, the SDK's documented
   write path, editing the config file as it is when it saves. (`src/configure.ts`)
3. **Conversation hooks are gated.** OpenClaw now drops `llm_input`, `llm_output` and
   `agent_end` subscriptions from non-bundled plugins unless
   `hooks.allowConversationAccess` is `true`. Those hooks create the root and LLM spans, so
   without the grant nothing is exported, and no error is shown. `openclaw mlflow configure`
   and `install.sh` now set it, and `status` reports it. (`src/configure.ts`)
4. **One SDK per process.** OpenClaw loads a plugin several times per process, each time from a
   fresh copy under `plugin-captures/` with its own `@mlflow/core`. The OpenTelemetry tracer
   provider can only be registered once. Spans went to the first copy's SDK, while the copy
   that handled `agent_end` flushed its own empty processor, and nothing reached MLflow. All
   copies now share the first SDK through a process-global symbol. (`src/service.ts`)
5. **Session and user as trace metadata.** The session key and user were written only as
   root-span attributes, so MLflow's Sessions view and `metadata.mlflow.trace.session` searches
   couldn't find them. The `userId` from OpenClaw's context was also dropped. Both are now
   written to the trace's metadata, using the shared SDK's trace manager (`updateCurrentTrace()`
   needs an active span context, which hook callbacks don't have). (`src/service.ts`)

Unit tests cover fixes 2 to 5 (`tests/index.test.ts`).

## Develop

```bash
npm ci
npm test          # jest, no OpenClaw needed
npm run build     # tsc -> dist/
```
