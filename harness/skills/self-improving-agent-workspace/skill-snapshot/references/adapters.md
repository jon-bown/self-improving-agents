# Bringing an agent

The harness needs exactly one thing from an agent: answer a question using the instructions
text it is handed. Everything else (tracing, versions, evaluation) happens around that call.

## The contract

```python
def answer(question: str, instructions: str) -> str: ...
def answer(question: str, instructions: str, workspace: Path) -> str: ...  # also gets the folder
```

`instructions` is the version under test, not necessarily production. `workspace` is a folder
holding that text (named like `SI_INSTRUCTIONS_FILE`, e.g. `AGENTS.md`) plus a copy of
`SI_WORKSPACE_DIR`, for agents that read their instructions from disk.

## Choosing SI_AGENT

| Agent | SI_AGENT | Settings |
|---|---|---|
| a Python function or SDK (OpenAI Agents, LangGraph, Claude Agent SDK, Hermes `AIAgent`, `openclaw-sdk`) | `python` | `SI_AGENT_TARGET=pkg.module:answer` |
| a service | `http` | `SI_AGENT_URL`; POST `{"input", "instructions"}`, reply `{"output"}` |
| a CLI | `command` | `SI_AGENT_COMMAND=tool --system-file {instructions_file} --message {input}` |
| no agent yet | `litellm` | `SI_AGENT_MODEL=anthropic:/claude-haiku-5-5`; reference files from `SI_WORKSPACE_DIR` become `read_file` tools |
| OpenClaw | `openclaw` | `SI_AGENT_MODEL`; instructions become `AGENTS.md`; isolated profile in `.selfimprove/openclaw/` |

The `command` adapter splits the template like a shell, then fills each argument, so a question
is always a single argument and is never interpreted by a shell.

## Keeping the user's own setup untouched

A personal agent is someone's real configuration and billing. When wiring one up:

* run it against an **isolated state directory** (OpenClaw: `OPENCLAW_STATE_DIR`; Hermes:
  `HERMES_HOME`) under `.selfimprove/`, never `~/.openclaw` or `~/.hermes`;
* never stop or reuse a gateway, server or kernel the user already runs, and use a different port
  if the agent needs one;
* read API keys from `.env` only; do not let an agent fall back to a saved subscription login;
* fence file-based agents to their workspace (Hermes treats the enclosing git repository as the
  project: `git init` the workspace so it cannot see answer keys elsewhere in the repo).

## Agents that already trace themselves

If the agent logs its own MLflow traces (autolog, its own spans, an OpenClaw or Hermes MLflow
plugin), the harness can read those as production signal: set `SI_PRODUCTION_FILTER` to a trace
filter that selects them, for example `tags.environment = 'production'`. Scoring, labeling,
triage, alignment and the analyst all work on those traces. The gate and optimizers still call
the agent through the adapter, because a candidate's behavior has to be produced, not observed.

## Examples

`examples/support_agent.py` is a complete `python` adapter: a LiteLLM tool-use agent with a
`lookup_policy` TOOL span and an offline stand-in for rehearsals. `examples/hermes_agent.py` shows
the same contract for Hermes' `AIAgent`, with its isolation shims. OpenClaw is built in
(`SI_AGENT=openclaw`); to drive it through its gateway with the `openclaw-sdk` client instead,
wrap that client in an `answer()` function the same way.
