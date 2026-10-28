# OpenClaw → MLflow traces, then the self-improvement harness

## Why nothing shows up

The main cause is `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. The OpenClaw plugin runs inside
the Node process and sends spans over **HTTP** to an MLflow server's REST API. It can't write to
a SQLite file, so it fails without saying anything. You need a running `mlflow server`.

Three other things also stop export without any error. Check them while you're in there:

| Cause | Symptom | Fix |
|---|---|---|
| Tracking URI is `sqlite:///…` | no traces, no error | run `mlflow server`, use `http://localhost:5000` |
| No **experiment id** set | plugin loads but never registers its hooks (it needs both URI and id) | `openclaw mlflow configure`, or `MLFLOW_EXPERIMENT_ID` |
| `hooks.allowConversationAccess` not `true` | OpenClaw 2026.10+ drops the `llm_input`/`llm_output`/`agent_end` hooks that build the root and LLM spans, with no error | `openclaw mlflow configure` sets it |
| Upstream npm/ClawHub build of the plugin | on 2026.10.x it imports a renamed SDK module (`plugin-sdk/diagnostics-otel`) and never loads; it can also exit before flushing on a one-shot `--local` run | use the patched build in `connectors/openclaw-mlflow` of this repo |

If you installed `@mlflow/mlflow-openclaw` (or `@byted-motor/mlflow-openclaw`) from npm or ClawHub,
uninstall it first. Two copies register the same service, and whichever loads second fails.

## Step 1: get traces from your everyday OpenClaw

```bash
# 1. An MLflow server, reusing the SQLite file you already have. Use an absolute path:
#    a relative sqlite:/// path resolves against whatever directory you start from.
mlflow server --backend-store-uri sqlite:///$HOME/path/to/mlflow.db --port 5000
# leave it running; the UI is http://localhost:5000

# 2. In another terminal, replace whatever plugin you installed with the patched one
openclaw gateway stop                                   # only if you run a gateway
openclaw plugins uninstall --force mlflow-openclaw      # only if one is installed

cd "self-improving-agent/connectors/openclaw-mlflow"
./install.sh --tracking-uri http://localhost:5000 --experiment openclaw
#   builds, installs into your profile, creates the experiment, sets the id and
#   allowConversationAccess. Takes about 2 minutes.
openclaw gateway start                                  # if you stopped it

# 3. Check
openclaw mlflow status
#   Enabled: true / Tracking URI: http://localhost:5000 / Experiment ID: <id> / Conversation access: true
openclaw agent --local --agent main -m "Say hello in five words"
```

In the UI, open experiment **openclaw** and go to **Traces**. Each run is one trace: an
`openclaw_agent` root span (AGENT), an `llm_call` span for every model call, and a TOOL span for
every tool call, named after the tool (`read`, …). Token usage and cost are attached.

Two warnings are expected. "Can't verify where this plugin came from" appears because the plugin
was installed from a local archive. "Forced retirement after 5000ms" appears when a one-off CLI
command shuts down. Neither one affects tracing.

`install.sh` installs into whatever profile `openclaw` uses, which is `~/.openclaw` by default. To
try it on a throwaway profile first, export `OPENCLAW_STATE_DIR=/tmp/oc-test` and
`OPENCLAW_CONFIG_PATH=/tmp/oc-test/openclaw.json` before running it.

## Step 2: run your agent through the harness, with tool calls visible

The harness has a built-in OpenClaw adapter (`SI_AGENT=openclaw`). For each question it runs
`openclaw agent --local --json` in a new session. Your instructions go in as `AGENTS.md` and your
docs folder is copied next to it. The harness uses its **own isolated profile** in
`harness/.selfimprove/openclaw/` (read-only `read` tool, gateway port 19555). Your `~/.openclaw`
and your gateway are never used. With `SI_OPENCLAW_MLFLOW=1`, `selfimprove doctor` installs the
same patched plugin into that isolated profile, so every harness call shows OpenClaw's own
LLM and tool spans.

### Install and configure

```bash
cd self-improving-agent/harness
uv venv -p 3.11 .venv && uv pip install -p .venv -e .
cp .env.example .env
```

Edit `harness/.env`:

```bash
# The plugin needs HTTP, so the harness must use the server too (not its sqlite default)
MLFLOW_TRACKING_URI=http://localhost:5000
SI_EXPERIMENT=openclaw-selfimprove

SI_AGENT=openclaw
SI_AGENT_MODEL=anthropic:/claude-haiku-5-5     # the model OpenClaw runs, "<provider>:/<model>"
SI_OPENCLAW_MLFLOW=1                            # OpenClaw's own LLM/TOOL spans per call
ANTHROPIC_API_KEY=...                           # keys come from .env only

# What gets improved: your agent's instructions (they become AGENTS.md)
SI_PROMPT_NAME=my-openclaw-instructions
SI_INSTRUCTIONS_FILE=/path/to/your/AGENTS.md
# The docs the agent should read. Copied to <workspace>/<folder name>/, e.g. docs/
SI_WORKSPACE_DIR=/path/to/your/docs

SI_CRITERIA_FILE=my_criteria.yaml
SI_GOLDEN_FILE=my_golden.yaml                   # a few hand-written questions + expected facts
SI_JUDGE_MODEL=anthropic:/claude-opus-5-5
```

Make sure your `AGENTS.md` tells the agent where the docs are, for example: "answer from the
files in `docs/`, read the relevant one before answering, and say which one you used."

### Have the verifier check that it actually reads the docs

A span check costs nothing and doesn't depend on a judge. Put one in `my_criteria.yaml` (start
from `examples/criteria.yaml`, but take out its `lookup_policy` checks: that tool only exists in
the example agent):

```yaml
quality:
  instructions: |
    Question: {{ inputs }}
    Answer: {{ outputs }}
    Would you send this answer as-is? True if it is correct, current and to the point.

checks:
  - name: read_the_docs
    requires_tool: read              # a TOOL span named "read" must exist and succeed
    tool_input_match: 'docs/'        # ...and its input must point into your docs folder
    rationale: "Answers must come from the docs, not memory."
    must_pass: true                  # a candidate that skips the docs never ships
  - name: concise
    max_words: 150

guidelines:
  - name: grounded
    guidelines: "Every factual claim must come from the documentation the agent read. FAIL on guesses."
```

How the harness finds the tool spans: its own trace for a call holds an `openclaw.agent` span
with `openclaw.session_key`, `openclaw.tool_calls` and `openclaw.tools` attributes. The plugin
writes a second trace with the same session key in its `mlflow.trace.session` metadata.
`requires_tool` looks in both. A `read` that returns an error or `not_found` counts as a failed
call, so reading the wrong path still fails the check.

### Run it

```bash
.venv/bin/selfimprove doctor --smoke "a typical question for your agent"
#   should show: openclaw ... valid / mlflow plugin: installed in the isolated profile
.venv/bin/selfimprove init                     # your AGENTS.md -> v1 @production
.venv/bin/selfimprove ask "a real question"    # then open it in the UI
```

In the UI, under experiment `openclaw-selfimprove`, every call produces two traces:

* **the harness trace**: tags `si_source` and `si_prompt_version`, scorer feedback, and the
  `openclaw.tools` attribute (for example `read`);
* **OpenClaw's trace**: `openclaw_agent` → `llm_call` / `read` spans. Open the `read` spans to see
  exactly which files it opened. To find the matching trace, search
  ``metadata.`mlflow.trace.session` = '<openclaw.session_key>'``.

Plugin traces have no `si_source` tag, so the loop never counts them as production traffic.

## Step 3: the loop (start manual, one level at a time)

```bash
selfimprove replay questions.txt          # realistic traffic (one question per line)
selfimprove label                         # you: p / f / s, with a reason. You are the ground truth
selfimprove score                         # checks + judges, including read_the_docs
selfimprove align                         # does the quality judge agree with your labels?
selfimprove triage && selfimprove approve # failures -> themes -> tests in the dataset
selfimprove improve --from-file new_AGENTS.md   # level 1: you write the candidate
selfimprove gate                          # candidate vs production, on CI + hidden holdout
selfimprove promote N                     # your decision; `selfimprove rollback` undoes it
selfimprove report                        # traces in, themes out, tests made, tests you edited
```

Once you've done that a few times, let the trace analyst write candidates
(`improve --improver analyst`) and keep reading the diffs. Move to GEPA only when the judge
agrees with your labels at least 75% of the time and the dataset has about 20 or more training
rows. Leave `SI_AUTO_PROMOTE=false` until the gate has made several calls you agreed with. The
harness enforces some rules no matter what: whoever writes a candidate doesn't approve it, the
holdout is never shown to an improver, and a judge's verdict is never used as an expected answer.

## Notes

* Your everyday OpenClaw (Step 1) and the harness's isolated profile (Step 2) are separate
  installs. They can share one MLflow server; use different experiments to keep them apart.
* The harness only improves the instruction text (`AGENTS.md`). Changes to skills, tools or
  config come back as proposals for you to accept.
* `install.sh` and the harness both refuse a `sqlite:///` URI for the plugin, for the reason
  above.
