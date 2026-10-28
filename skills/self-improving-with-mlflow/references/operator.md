# Running an agent with this skill (for its owner)

The skill is the agent's side of the `selfimprove` loop. Everything the agent writes waits for
you or the gate. This page covers your side: deploying it, and the commands it never runs.

## Deploy

```bash
cd harness && make install            # the harness, once
selfimprove init                      # register AGENTS.md as v1, if you haven't
selfimprove self setup --workspace ~/acme-live
source .selfimprove/openclaw-live/env.sh
openclaw agent --local --agent main --session-id demo --message "How long do refunds take?"
```

`self setup` writes an isolated OpenClaw profile (default `harness/.selfimprove/openclaw-live/`,
gateway port 19556). It never touches `~/.openclaw` or a gateway on 18789. The profile has:

* `tools.allow: ["read", "exec"]`: no `write`, so the agent can't edit its own files;
* `tools.exec.mode: "allowlist"`, with one entry: `<profile>/bin/selfimprove`, a wrapper that runs
  `self status|note|recall|propose` and refuses everything else (`gate`, `promote`, `queue`,
  `self lessons|sync|setup`);
* `skills.workshop.autonomous.mode: "off"`: OpenClaw's own learning applies changes with no
  verifier, so every change goes through the gate instead;
* the skill copied into `<workspace>/skills/self-improving-with-mlflow/` (without its evals);
* with `SI_OPENCLAW_MLFLOW=1`, the MLflow connector, so every conversation is traced.

Set `SI_PRODUCTION_FILTER` so the connector's traces count as production traffic, and
`SI_SELF_WORKSPACE` to the workspace (the wrapper already sets it for the agent).

## Your commands

| when | command |
|---|---|
| the agent noted a correction | `selfimprove queue list`, then `queue accept <id>` (a HUMAN `quality` label on that trace) or `queue reject <id> --note "..."`. `queue edit <id> --expected "..."` fixes the right answer first. |
| confirmed failures have built up | the usual `triage` and `approve`, then `selfimprove self lessons` (add `--gate` to gate it now) |
| a candidate is waiting (lessons, or the agent's own proposal) | `selfimprove gate`, then `selfimprove promote <v>` if you approve |
| production changed | `selfimprove self sync` writes `AGENTS.md` and `LESSONS.md` into the live workspace; run it after `promote` or on a schedule |
| the agent proposed a skill change | read `.selfimprove/proposals/<id>.diff`, then `queue accept <id>` writes it into `<workspace>/skills/<name>/SKILL.md` |

## How the lessons pack works

`self lessons` reads the dataset's human-confirmed failures (`source` `human_fail` or `queue`),
**train split only**, groups them by theme, and writes one lesson per theme (at most
`SI_LESSONS_MAX`, default 8). With model credentials, an LLM writes each lesson as a pointer to
the source to check. Without them, it uses the failure's recorded reason.

The pack is a delimited block inside the registered instructions, so:

* the gate scores instructions and lessons together, on the same train split and holdout;
* `promote` ships both and `rollback` takes both back, with one alias move;
* the workspace gets the block as `LESSONS.md`, and `AGENTS.md` keeps a one-line pointer;
* improvers (analyst, GEPA, MetaPrompt, a coding agent) and the agent's own proposals keep
  production's block as it is: lessons only change through `self lessons`.

`self recall` reads the pack at `production`. It never reads traces or dataset rows, so what the
agent recalls is exactly what the gate measured. A lesson whose evidence later moves into the
holdout is dropped at read time too.

## Settings

| variable | default | what it does |
|---|---|---|
| `SI_SELF_NOTE_MAX_PER_SESSION` | 5 | notes per conversation per day |
| `SI_SELF_NOTE_WINDOW_MIN` | 30 | how far back `--trace previous` looks |
| `SI_SELF_PROPOSE_MIN_EVIDENCE` | 2 | confirmed failures a proposal must cite |
| `SI_SELF_PROPOSE_GATE` | off | `1` runs the gate as soon as the agent proposes (still no promotion unless `SI_AUTO_PROMOTE=true`) |
| `SI_LESSONS_MAX` | 8 | lessons in the pack |
| `SI_SELF_WORKSPACE` | unset | the live workspace (`self sync`, skill proposals) |
| `SI_MAX_PROMPT_GROWTH` | 2.0 | also the early length check on proposals |

## Live failure rate on deployed traffic

Connector traces don't carry the prompt version the way harness traces do, so `self sync` logs
which version it wrote and when (`.selfimprove/self_sync.jsonl`). `self status` credits deployed
traffic to whichever version was live when each trace started.
