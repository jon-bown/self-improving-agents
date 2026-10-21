# MLflow connectors for personal agents

These plugins send every run of a personal agent to a local MLflow server as a trace, with no code
changes to the agent:

| Agent | Plugin | One trace per | Spans |
|---|---|---|---|
| [OpenClaw](https://github.com/openclaw/openclaw) ≥ 2026.10.5 | `openclaw-mlflow/` (TypeScript) | agent run | `openclaw_agent` AGENT → `chat <model>` CHAT_MODEL per provider request, one TOOL per tool call |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) ≥ 0.19 | `hermes-mlflow/` (Python) | turn | `hermes_agent` AGENT → `chat <model>` CHAT_MODEL per request, TOOL per tool call, delegated subagents nested |

Each trace records:
- the conversation going in (OpenAI chat format, so MLflow's chat view and previews render it);
- the reply, tool arguments and results, and errors;
- token usage, which MLflow rolls up and prices;
- the session (`mlflow.trace.session`, so turns group into conversations) and the user.

**No dependencies.** Both plugins speak OTLP/HTTP JSON directly to MLflow's `/v1/traces`, so there is
no `npm install` and no `pip install`. They use only Node's `fetch` and Python's `urllib`.

## Setup

Start MLflow, then connect your agent:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

python connectors/connect.py openclaw     # or: make connect-openclaw
python connectors/connect.py hermes       # or: make connect-hermes
python connectors/connect.py status
```

`connect.py` checks that the server is up and gets or creates the experiment. It then wires the
plugin into the agent's **active** config:

- **OpenClaw:** a validated `openclaw config patch`. It adds this directory to `plugins.load.paths`,
  sets `plugins.entries.mlflow-otel` with the MLflow target, and turns on
  `hooks.allowConversationAccess`. OpenClaw requires that opt-in before a plugin may see prompts and
  replies.
- **Hermes:** symlinks the plugin into `$HERMES_HOME/plugins/mlflow-otel`, writes `MLFLOW_*` to
  `$HERMES_HOME/.env`, and runs `hermes plugins enable mlflow-otel`. It declines tool overrides,
  because a tracer replaces no tools.

Options:

| Option | Effect |
|---|---|
| `--tracking-uri` | Default `$MLFLOW_TRACKING_URI` or `http://127.0.0.1:5000` |
| `--experiment NAME` | Default `openclaw` or `hermes-agent` |
| `--no-content` | Keep structure, timing and tokens; drop all text |
| `--dry-run` | Print what would change |

To try it without touching your real profile, point the agent at a scratch one:
`OPENCLAW_STATE_DIR=… OPENCLAW_CONFIG_PATH=…` or `HERMES_HOME=…`.

In this repo, `python levels/level0_instrument/01_setup.py --mlflow-plugin` enables the OpenClaw plugin
in the demo's isolated profile. Every OpenClaw run's trace is then tagged
`sia.parent_trace_id = <the acme_support_agent trace>`, so the two traces can be joined.

## Configuration

Both plugins read the same environment variables. OpenClaw also accepts these keys in
`plugins.entries.mlflow-otel.config` (camelCase), and Hermes in `plugins.entries.mlflow-otel` in
`config.yaml` (snake_case).

| Setting | Default | |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | must be an http(s) server, not a file store |
| `MLFLOW_EXPERIMENT_ID` | | numeric; the OTLP endpoint requires an ID |
| `MLFLOW_EXPERIMENT_NAME` | `openclaw` / `hermes-agent` | used when no ID is set; created if missing |
| `MLFLOW_OTEL_CAPTURE_CONTENT` | `true` | `false` drops prompts, replies, tool args/results |
| `MLFLOW_OTEL_CAPTURE_SYSTEM_PROMPT` | `false` | OpenClaw only |
| `MLFLOW_OTEL_TRACE_TAGS` | | JSON object of extra trace tags, read per run |
| `MLFLOW_TRACKING_TOKEN` | | Hermes: sent as a Bearer token. OpenClaw: use `config.headers` |
| `MLFLOW_OTEL_DISABLED` | | OpenClaw: set to turn the plugin off |

Content is truncated at 20,000 characters per field.

## How the attributes map

MLflow's OTLP endpoint gives `mlflow.*` attributes precedence. They are used here because
they are exact:

| Attribute | Becomes |
|---|---|
| `mlflow.spanType` | the span type (AGENT, CHAT_MODEL, TOOL) |
| `mlflow.spanInputs` / `mlflow.spanOutputs` (JSON) | inputs and outputs; the root's become the trace's request and response previews |
| `mlflow.chat.tokenUsage` | token counts; summed into `mlflow.trace.tokenUsage`, and priced when the model is known |
| `mlflow.traceTag.<k>` on the root | trace tag `<k>`; `mlflow.traceTag.mlflow.traceName` names the trace |
| `session.id`, `user.id` | trace metadata `mlflow.trace.session` / `mlflow.trace.user` (an attribute literally named `mlflow.trace.session` is ignored) |

`gen_ai.*` semantic-convention attributes (`gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.provider.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`) are set too, so the spans make
sense in any OTel backend.

## Things we learned (and the plugins handle)

**OpenClaw 2026.10.5**
- The hooks used are `llm_input`, `llm_output`, `model_call_started`, `model_call_ended`,
  `before_tool_call`, `after_tool_call` and `agent_end`. Plugins can't subscribe to the richer
  diagnostic event stream: only `diagnostics-otel` gets it.
- A plugin is registered several times per process: one `full` registry, plus `discovery` registries.
  The agent's hooks can fire from a discovery registry, so hooks are registered in every mode and
  share one process-wide tracker.
- **`llm_output` fires after `agent_end`.** The run is held for up to 250 ms so the reply and token
  usage make it in.
- `openclaw agent --local` calls `process.exit()` about 500 ms after the run, which cuts off an async
  `fetch`. Anything still unsent is posted from the `exit` handler by a short synchronous child
  process. A long-lived Gateway never takes that path.
- User prompts carry a `[Tue 2026-10-20 16:03 EDT]` envelope. It's stripped from inputs.
- MLflow's own `@mlflow/mlflow-openclaw@0.2.0-rc.0` targets older OpenClaw. It imports a removed SDK
  path, uses the removed `subagent_spawning` hook, and ships TypeScript only.

**Hermes 0.19**
- Hooks are synchronous and may run on tool worker threads. State is keyed by `turn_id`,
  `api_request_id` and `tool_call_id` under a lock, and parents are set explicitly.
- `on_session_end` fires at the end of every turn, so that is where a trace is sent (synchronously,
  5 s timeout). Turns still open at `on_session_finalize` or process exit are flushed.
- A return value from `pre_llm_call` is injected into the prompt, and one from `pre_tool_call` can
  block the tool. Every hook here returns `None`.
- Provider calls run on bare threads without context propagation, so `mlflow.openai.autolog()` would
  produce orphan traces. The hooks give a coherent tree for every provider.

**MLflow 3.16**
- Don't write spans into a trace that the MLflow Python client is still building. Mixing the two
  writers corrupts the trace. Link traces with tags instead (`MLFLOW_OTEL_TRACE_TAGS`).

## Development

```bash
cd connectors/openclaw-mlflow && npm test         # Node ≥ 23.6 runs the .ts sources directly
.venv/bin/python -m pytest tests/unit/test_hermes_connector.py tests/unit/test_connect.py
```

OpenClaw loads the plugin from source (`.ts`) when it's on `plugins.load.paths`. Publishing it as a
package needs compiled JS: `npm run build`, then add `"runtimeExtensions": ["./dist/index.js"]` to
`package.json`.
