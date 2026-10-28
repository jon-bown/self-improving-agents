# MLflow tracing for OpenClaw

An OpenClaw plugin that logs every agent run to MLflow as a trace: a root `AGENT` span with
`LLM`, `TOOL` and sub-agent spans under it, plus token usage, model and session.

<p align="center">
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/master/libs/typescript/integrations/openclaw/trace-screenshot.png" alt="An OpenClaw trace in the MLflow UI" width="700" />
</p>

This is the official `@mlflow/mlflow-openclaw` integration from MLflow's repository
(`libs/typescript/integrations/openclaw` at commit `568cd81`, 2026-10-21). It carries seven fixes
that make it work on OpenClaw 2026.10 (listed [below](#what-is-fixed-here)). Plugin id, config
keys and CLI stay as they are upstream, so it can be swapped for the npm release once that has
the fixes.

## Contents

| | |
|---|---|
| `src/index.ts` | plugin entry: registers the tracing service, its hooks and the `openclaw mlflow` CLI |
| `src/service.ts` | turns OpenClaw's hook events into MLflow spans and exports them |
| `src/configure.ts` | `openclaw mlflow configure` and `openclaw mlflow status` |
| `openclaw.plugin.json` | plugin manifest: id `mlflow-openclaw` and its config schema |
| `install.sh` | builds the plugin and installs and configures it in one step |
| `tests/` | unit tests (jest), no OpenClaw needed |

## Requirements

* Node 22+ and `openclaw` on your `PATH` (`npm i -g openclaw`, 2026.10 or later).
* An MLflow **server**. The plugin exports traces over HTTP, so a `sqlite:///` store won't
  work. Start one and leave it running:

  ```bash
  mlflow server --port 5000
  ```

* An experiment for the traces, identified by its **id**. `install.sh --experiment NAME`
  creates it for you. To create one yourself, run this and note the id it prints:

  ```bash
  MLFLOW_TRACKING_URI=http://localhost:5000 mlflow experiments create -n openclaw
  ```

## Install

### One command

From this folder:

```bash
./install.sh --tracking-uri http://localhost:5000 --experiment openclaw
```

It builds the plugin, installs it into your OpenClaw profile, and points it at the experiment,
creating the experiment if needed. It takes about two minutes, most of it in
`openclaw plugins install`. Then restart your gateway (`openclaw gateway restart`) if one is
running.

`install.sh` targets whichever profile the `openclaw` CLI would use: `~/.openclaw` by default,
or the one set by `OPENCLAW_STATE_DIR` / `OPENCLAW_CONFIG_PATH`. Other options:
`--experiment-id ID` instead of a name, `--test` to also run the unit tests, and
`--build-only` to stop after building the archive.

### Step by step

The same steps by hand, with the gateway stopped while the plugin is installed. Run them from
this folder.

1. **Build the archive.** This writes `mlflow-mlflow-openclaw-<version>.tgz` here:

   ```bash
   ./install.sh --build-only
   ```

2. **Stop the gateway.** While it runs, plugin installs go through the gateway. Stopped, the
   CLI installs directly, and the gateway loads the new plugin when it starts. Your channels and
   the Control UI are offline until step 5.

   ```bash
   openclaw gateway stop
   ```

3. **Install the plugin.** Use the archive, not this folder: OpenClaw keeps its own copy under
   `~/.openclaw/extensions/`, so the install doesn't depend on this checkout. A local archive
   has no ClawHub or npm provenance, so OpenClaw needs `--force`, and `--accept-capabilities`
   records your consent to the hooks the plugin declares.

   ```bash
   openclaw plugins install --force --accept-capabilities ./mlflow-mlflow-openclaw-0.2.1-sia.1.tgz
   ```

   The version in the file name comes from `package.json`; step 1 prints the full path.

4. **Point it at MLflow.** The prompts ask for the tracking URI and the experiment id, starting
   from the current values (or `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID`). The
   command also enables the plugin, grants it conversation access (needed on OpenClaw 2026.10,
   see [fix 3](#what-is-fixed-here)), and adds it to `plugins.allow` if your profile has an
   allowlist.

   ```bash
   openclaw mlflow configure
   ```

5. **Start the gateway:**

   ```bash
   openclaw gateway start
   ```

   On a slow start it may print "still starting after 45s". That only means the CLI stopped
   waiting; `openclaw gateway status --deep` shows when it's listening.

6. **Check the install.** The plugin's source should be `global:mlflow-openclaw/...`, the
   installed copy:

   ```bash
   openclaw plugins list | grep -i mlflow
   ```

## Check that traces arrive

Show the settings the plugin will use:

```bash
openclaw mlflow status
```

```
  Enabled:        true
  Tracking URI:   http://localhost:5000
  Experiment ID:  1
  Conversation access: true
```

Then run one turn through the gateway:

```bash
openclaw agent --agent main -m "Say hello in five words"
```

Open the MLflow UI (http://localhost:5000), pick the experiment, and open **Traces**. Each run
is one trace: an `AGENT` root span, one `LLM` span for the run's model calls, and a `TOOL` span
per tool call, plus token usage, the model and the session. OpenClaw reports tokens but not
cost, so MLflow's cost charts stay empty. Runs with `--local` are traced too, without a gateway.

Two things you'll see that are normal:

* On every start OpenClaw warns that it "can't verify where this plugin came from". That's
  because it was installed from a local archive. The plugin still loads.
* One-off CLI commands such as `openclaw mlflow status` may end with "forced retirement after
  5000ms" lines for every loaded plugin. That's the CLI process shutting down, not the gateway,
  and it doesn't affect tracing.

## Next: start a self-improvement loop on these traces

[`examples/self_improvement_demo.py`](examples/self_improvement_demo.py) takes the traces this
plugin wrote and sets up the first pieces of a loop with plain MLflow. It registers your `AGENTS.md`
in the Prompt Registry with a `production` alias. It scores the traces with `read_knowledge`, a code
check on the `read` spans, and `grounded_in_docs`, a judge that sees the files the agent read. Then
it adds the failures to an evaluation dataset, each tagged with its trace:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python examples/self_improvement_demo.py \
  --experiment openclaw --agents-md /path/to/your/AGENTS.md
```

It needs `mlflow[genai]` and the judge's API key (`ANTHROPIC_API_KEY` by default; `--judge-model`
changes it). The checks assume the agent reads its policies from `knowledge/`; edit `read_knowledge`
for your own layout. For the whole loop (triage, improvement and a gate), use the
[harness](../../harness/README.md) with `SI_OPENCLAW_MLFLOW=1`.

[`examples/demo.sh`](examples/demo.sh) is a five-minute version for a live demo: `server`, `setup`,
then `ask "question"`, which always runs whatever `@production` points at. `score` judges the new
traces, `improve` has OpenClaw rewrite its own `AGENTS.md` from the failures (registered as a
candidate, [`examples/self_improve_loop.py`](examples/self_improve_loop.py)), and `gate` re-asks the
failing question plus a holdout one and moves `production` only if every check passes. Or edit
`AGENTS.md` in the MLflow UI (a new prompt version, then move the alias) and the next `ask` uses it.
`./demo.sh use 1` rolls back. It keeps its own OpenClaw profile and server in `~/odsc-5min`.
[`examples/live-demo.md`](examples/live-demo.md) is the script: every command, click and
edit, with the output to expect.

## Change the settings

Run `openclaw mlflow configure` again, then restart the gateway so it uses the new values:

```bash
openclaw mlflow configure
```

```bash
openclaw gateway restart
```

To script it instead, set the keys with `openclaw config set`:

```bash
openclaw config set --batch-json '[{"path":"plugins.entries.mlflow-openclaw.config.trackingUri","value":"http://127.0.0.1:5000"},{"path":"plugins.entries.mlflow-openclaw.config.experimentId","value":"3"}]'
```

To set a single key, quote the id as JSON: `openclaw config set
plugins.entries.mlflow-openclaw.config.experimentId '"3"'`. A bare `3` is parsed as a number
and fails validation.

To pause export without uninstalling, set `plugins.entries.mlflow-openclaw.enabled` to `false`
and restart the gateway.

## Update or remove

To install a new build, repeat [Step by step](#step-by-step) from step 1. In step 4 the
prompts start from your current values, so pressing Enter keeps them. To remove the plugin, stop the gateway first, for the same reason as above:

```bash
openclaw gateway stop
```

```bash
openclaw plugins uninstall --force mlflow-openclaw
```

```bash
openclaw gateway start
```

Uninstalling removes the plugin's files, its install record and its settings.

Only one MLflow tracing plugin should be installed. Another build of the same plugin (for
example `@byted-motor/mlflow-openclaw` from ClawHub) registers the same `mlflow-tracing`
service and `mlflow` command, and the second one to load fails. `openclaw plugins doctor` lists
any conflicts.

## Try it in a separate profile

Every command above accepts `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH`, so you can try the
plugin without touching `~/.openclaw` or your gateway:

```bash
export OPENCLAW_STATE_DIR=/tmp/oc-test OPENCLAW_CONFIG_PATH=/tmp/oc-test/openclaw.json
./install.sh --tracking-uri http://localhost:5000 --experiment openclaw-test
openclaw agent --local --agent main -m "Say hello in five words"
```

## With the self-improving-agent harness

Set `SI_OPENCLAW_MLFLOW=1` (and an HTTP `MLFLOW_TRACKING_URI`) in `harness/.env`, then run
`selfimprove doctor`. It runs `install.sh` against the harness's isolated profile in
`.selfimprove/openclaw/`, never `~/.openclaw`. Each harness call then produces two traces in
the experiment:

* the harness's own trace (`si_source` tag, prompt version, the judges' feedback);
* OpenClaw's trace from this plugin, with the LLM and tool spans.

The harness span records `openclaw.session_key`. To find OpenClaw's trace for a harness trace,
search for ``metadata.`mlflow.trace.session` = '<that key>'``. Plugin traces have no
`si_source` tag, so the loop never mistakes them for production traffic.

## Configuration reference

| key (`plugins.entries.mlflow-openclaw.…`) | |
|---|---|
| `config.trackingUri` | MLflow server; falls back to `MLFLOW_TRACKING_URI` |
| `config.experimentId` | experiment **id** as a string; falls back to `MLFLOW_EXPERIMENT_ID` |
| `enabled` | `false` turns export off without uninstalling |
| `hooks.allowConversationAccess` | must be `true` on OpenClaw 2026.10+ (see fix 3) |

`openclaw mlflow configure` sets all four; `openclaw mlflow status` shows them.

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

6. **Two minutes per run.** OpenClaw copies a plugin's whole folder each time it loads it,
   several times per process, and `@mlflow/core`'s dependencies are about 175 MB of
   `node_modules`. Every `openclaw agent` run took about 150 s instead of 20 s, and the gateway
   started slowly. `npm run build` now bundles the plugin into one `dist/index.js` (esbuild,
   `scripts/bundle.mjs`) and the package declares no runtime dependencies, so an install is
   one 20 MB file. (`package.json`)
7. **Traces lost on a fast exit.** The `agent_end` handler flushed the trace in the background
   and returned nothing, so a one-shot `openclaw agent` run could exit before the export
   finished. It now returns the flush, which OpenClaw awaits (up to 30 s). (`src/service.ts`)

Unit tests cover fixes 2 to 5 (`tests/index.test.ts`).

## Develop

```bash
npm ci
npm test          # jest, no OpenClaw needed
npm run build     # tsc -> dist/, then bundle dist/index.js
```
