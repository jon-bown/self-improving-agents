# Getting OpenClaw traces into MLflow, then running it through the harness

## Why nothing shows up

You probably have more than one of these problems. The first one alone explains an empty UI.

1. **`sqlite:///mlflow.db` can't receive OpenClaw traces.** The plugin is TypeScript running inside OpenClaw, and it exports traces over HTTP to an MLflow **server**. A `sqlite:///` URI is a file path for Python clients, so the plugin has nowhere to send anything, and you get no error. You need `mlflow server` running and an `http://` tracking URI.
2. **Conversation access isn't granted.** OpenClaw 2026.10 silently drops the `llm_input`, `llm_output` and `agent_end` hooks for non-bundled plugins unless `plugins.entries.mlflow-openclaw.hooks.allowConversationAccess` is `true`. Those hooks create the root and LLM spans, so without the grant nothing gets exported, again with no error.
3. **No experiment id.** The plugin needs `experimentId`, either from its config or from `MLFLOW_EXPERIMENT_ID`. A tracking URI alone isn't enough.
4. **The npm/ClawHub build doesn't work on 2026.10.5.** If you installed `@mlflow/mlflow-openclaw` from npm, or `@byted-motor/mlflow-openclaw` from ClawHub, it imports `openclaw/plugin-sdk/diagnostics-otel`. That module was renamed to `diagnostic-runtime`, so the plugin never loads. This repo carries a fixed build in `connectors/openclaw-mlflow/`, with seven fixes for 2026.10, including the conversation-access grant. Install only one MLflow plugin, because two builds register the same service and the second one fails to load.

## Step 1: make plain `openclaw agent --local` produce traces

Start a server. You can keep your existing database by using it as the backend store:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

If you installed the npm or ClawHub plugin, remove it first (`openclaw plugins uninstall --force mlflow-openclaw`, and stop the gateway before doing that if one is running). Then install the fixed connector from the repo. It builds the plugin, installs it, creates the experiment, sets the URI and experiment id, and grants conversation access:

```bash
cd connectors/openclaw-mlflow
./install.sh --tracking-uri http://localhost:5000 --experiment openclaw
```

You need Node 22+ for this. It takes about two minutes. Check the result:

```bash
openclaw mlflow status
#   Enabled:        true
#   Tracking URI:   http://localhost:5000
#   Experiment ID:  <id>
#   Conversation access: true
openclaw agent --local --agent main -m "Say hello in five words"
```

Then open http://localhost:5000, choose the `openclaw` experiment and go to **Traces**. Each run is one trace with these parts:
- an `openclaw_agent` root span
- an `llm_call` span
- a `read` (or other) **TOOL span per tool call**, with its arguments and result

Some things that are normal:
- On every start OpenClaw warns that it "can't verify where this plugin came from", because the plugin was installed from a local archive.
- `openclaw mlflow status` can take a couple of minutes.
- `--local` runs pick up the plugin right away. A running gateway needs `openclaw gateway restart`.
- Cost stays empty because OpenClaw only reports tokens.
- All of a run's model calls appear as one `llm_call` span.

If you get tool spans but no root or LLM span, conversation access is still off. Run `openclaw mlflow configure` to fix it.

To try this without touching your real `~/.openclaw`, export `OPENCLAW_STATE_DIR=/tmp/oc-test OPENCLAW_CONFIG_PATH=/tmp/oc-test/openclaw.json` before running `install.sh`.

## Step 2: run your agent through the self-improvement harness

The harness has built-in OpenClaw support (`SI_AGENT=openclaw`). Every call runs `openclaw agent --local --json` with a fresh session, against an **isolated profile** in `harness/.selfimprove/openclaw/`. Your `~/.openclaw` and your gateway are never touched. With `SI_OPENCLAW_MLFLOW=1`, `selfimprove doctor` runs the same `install.sh` against that isolated profile, so the harness profile gets the tracing plugin too, separately from yours.

```bash
cd harness
uv venv -p 3.11 .venv && uv pip install -p .venv -e .
cp .env.example .env
```

Set these in `harness/.env`:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000     # the server from step 1 (the .env.example default is sqlite: change it)
SI_EXPERIMENT=my-openclaw-agent

SI_AGENT=openclaw
SI_AGENT_MODEL=anthropic:/claude-haiku-5-5    # MLflow-style URI; translated to OpenClaw's naming
SI_OPENCLAW_MLFLOW=1                          # install the connector into the harness's profile
SI_AGENT_TIMEOUT=600

SI_PROMPT_NAME=my-openclaw-agents-md
SI_INSTRUCTIONS_FILE=path/to/your/AGENTS.md   # the update target: becomes v1 in the Prompt Registry
SI_WORKSPACE_DIR=path/to/your/docs            # the docs the agent should read; copied next to AGENTS.md each run
SI_CRITERIA_FILE=my-criteria.yaml
SI_GOLDEN_FILE=my-golden.yaml                 # a few hand-written questions + expected answers

ANTHROPIC_API_KEY=...
SI_JUDGE_MODEL=anthropic:/claude-sonnet-5-5
```

The loop changes only `AGENTS.md`, your instructions. The docs folder stays fixed. The harness renders the version under test into a fresh workspace for every call.

### Checking that it actually reads the docs

Tool calls are visible in two places.

- **In the UI.** Each harness call produces two traces in the experiment. One is the harness's own trace, tagged `si_source`, with the prompt version and the judges' feedback. The other is OpenClaw's trace, with the `read` TOOL spans that show which file was read and what came back. The harness span records `openclaw.session_key`. To find the matching OpenClaw trace, search for:
  ```
  metadata.`mlflow.trace.session` = '<that key>'
  ```
- **As a check that runs on every trace and in the gate.** Put a tool-span check in your criteria file. The harness finds the connector's trace by session key on its own:

```yaml
quality:
  instructions: |
    Question: {{ inputs }}
    Answer: {{ outputs }}
    Would you send this answer as-is? Return true if it is correct, grounded and to the point.

checks:
  - name: read_docs
    requires_tool: read              # a `read` TOOL span must exist...
    tool_input_match: 'docs/'        # ...on a file in your docs folder (match your SI_WORKSPACE_DIR name)
    rationale: "Answers must come from the docs, not from memory."

grounded:
  - name: grounded_in_docs
    tool: read
    tool_input_match: 'docs/'
    instructions: >-
      Return true only if every factual claim in {{ outputs }} is supported by the files in {{ inputs }}.

reference: [key_facts, correctness]
```

`requires_tool` fails a missing, failed or empty lookup without calling a model. `grounded` judges the answer against what the agent actually read. For a complete worked example on a real OpenClaw agent, see `harness/session/`: `session.env`, `criteria/loop1.yaml` to `loop4.yaml`, and `RUNBOOK.md`.

### First run

```bash
caffeinate -i .venv/bin/selfimprove doctor --smoke "a typical question"   # installs the connector, makes one traced call
.venv/bin/selfimprove init                                                # AGENTS.md -> v1 'production'; golden cases -> dataset
.venv/bin/selfimprove ask "..."        # or: replay questions.txt
.venv/bin/selfimprove score            # run the checks; open every trace read_docs fails
.venv/bin/selfimprove label --why      # your pass/fail + reason: the ground truth
```

After that, run the loop by hand first:

```bash
selfimprove triage
selfimprove approve
selfimprove improve --improver analyst
selfimprove diff production latest
selfimprove gate
selfimprove promote N --reason ...
```

Read the diff yourself. Keep the gate's verdict and the promotion as your decisions. Don't automate (`cycle`, `SI_AUTO_PROMOTE`) until several manual rounds have gone by without you overriding the gate.

### Things to know

- **Speed.** Each OpenClaw call takes about 20 seconds. The gate runs every dataset row for both production and the candidate, so budget for it, and keep the laptop awake (`caffeinate -i`). GEPA is slow with OpenClaw because every metric call is an agent run.
- **One run per profile.** Don't run two harness jobs on the same profile at once ("Another OpenClaw process is using your data").
- **Wrong instructions.** If the agent ignores the instructions under test, the profile's config names a workspace (`agents.defaults.workspace`), which overrides the per-call workspace. The harness writes its profile without one. Don't add one.
- **Your own traces.** You can also feed the OpenClaw traces from your own day-to-day use into the loop as production signal with `SI_PRODUCTION_FILTER`. The gate still calls the agent through the harness, because a candidate's behaviour has to be produced, not observed.
