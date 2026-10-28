# Live demo: catch it in MLflow, let it fix itself

A five-minute live demo of self-improvement on a local OpenClaw agent, with the MLflow connector
tracing every run.

1. You ask the agent a question in the terminal, and it makes something up.
2. Its trace in MLflow shows the policy files it read, and that none of them say what it said. A
   judge in MLflow fails the answer and says why.
3. OpenClaw reads those failures and rewrites its own instructions. The rewrite goes into the
   MLflow Prompt Registry as a candidate, not into production.
4. A gate re-asks the failing question plus one the agent never saw, scores both, and promotes the
   candidate only if every check passes.
5. You ask again, and it's fixed.

You can also make the fix by hand in the MLflow UI instead of steps 3 and 4.

The agent is OpenClaw answering as support for a fictional product, Acme Cloud. Its `AGENTS.md`
lives in the MLflow Prompt Registry, and `./demo.sh ask` always runs whatever version
`@production` points at. So moving the alias is the deploy, and moving it back is the rollback.

Everything runs through [`demo.sh`](demo.sh), from this folder. It keeps its own OpenClaw
profile and MLflow server in `~/odsc-5min` (port 5050), so it never touches `~/.openclaw`, a
gateway you run, or any other MLflow server.

## You need

* `openclaw` on your `PATH` (2026.10 or later) and Node 22+
* the harness venv: `cd harness && make install`, from the repo root
* `ANTHROPIC_API_KEY` in `harness/.env`; `demo.sh` reads it from there and never prints it
* port 5050 free (port 5000 on a Mac is the AirPlay Receiver, not MLflow)

## Before the talk (about 2 minutes)

From `connectors/openclaw-mlflow/examples`:

**Terminal 1:** start the MLflow server, and leave it running.

```bash
./demo.sh server
```

**Terminal 2:** set everything up (about 50 seconds).

```bash
./demo.sh setup
```

`setup` does four things:

* makes the OpenClaw profile and its workspace;
* copies in three Acme policy files (`billing`, `account_access`, `product_howto`);
* installs the connector;
* registers [`agents/v1.md`](agents/v1.md) as version 1 of the prompt `openclaw-agents-md`, with
  `@production` pointing at it.

It ends by printing the two pages to open:

```
ready. In MLflow (experiment openclaw-demo):
  traces:  http://127.0.0.1:5050/#/experiments/1/traces
  prompt:  http://127.0.0.1:5050/#/experiments/1/prompts  (openclaw-agents-md)
```

Open both in a browser and close the MLflow Assistant panel on the right.

**Rehearse, then reset**, so the stage run starts clean:

1. Stop the server with Ctrl-C.
2. Run `./demo.sh reset`, which deletes `~/odsc-5min`.
3. Start again with `./demo.sh server` and `./demo.sh setup`.

## On stage (about 5 minutes)

### 1. Ask (about 30 s)

```bash
./demo.sh ask "Do you have an uptime SLA?"
```

The output starts with the version that's running, then the answer:

```
running openclaw-agents-md v1 (@production): v1: what the agent runs today

Yes. Acme Cloud has a **99.9% monthly uptime SLA**, and if we miss it you receive **service credits**.
```

None of the policy files mention an SLA. With this v1, every rehearsal run answered like this.

### 2. Find it in MLflow (about 1 min)

1. **Traces** → the newest trace. Open its `read` spans: the policy files the agent opened, with
   their contents. None of them mention an SLA.
2. **Prompts & versions → Prompts** → `openclaw-agents-md`. v1 is `@production`. Point at its
   last bullet, which causes the answer:

   > If the files don't cover a question, answer from these quick facts: uptime SLA 99.9% monthly
   > with service credits, API limit 1,000 requests per minute, data hosted in the US and EU.

   Say: "Someone wrote those quick facts from memory. They aren't in any policy, and the agent
   repeats them."

### 3. Let MLflow catch it (about 15 s)

```bash
./demo.sh score
```

`score` runs two checks on every trace that doesn't have a verdict yet:

* `read_knowledge`: a code check that the agent read a policy file
* `grounded_in_docs`: a judge that sees the files the agent read, and fails any claim that isn't in
  them

The verdicts are attached to the traces in MLflow. In the terminal:

```
   Do you have an uptime SLA?            read_knowledge=pass  grounded_in_docs=FAIL
      why: The policy files (billing, product how-to, account access) say nothing about an uptime SLA. ...
```

### 4. OpenClaw fixes itself (about 30 s)

```bash
./demo.sh improve
```

`improve` hands OpenClaw the failing traces (question, answer, and the judge's reason) together
with the `AGENTS.md` in production, and asks for a corrected file. It registers the reply as a new
prompt version, a **candidate**, and prints the diff:

```
OpenClaw is rewriting v1 from 1 failing trace(s) ...
--- v1 (@production)
+++ v2 (candidate)
@@ -11,5 +11,4 @@
   - `knowledge/product_howto.md`: exports, teammates and seats, API keys, integrations, offline mode
-- Customers want a straight answer. Never say you don't know, that something isn't documented,
-  or that they should ask someone else. If the files don't cover a question, answer from these
-  quick facts: uptime SLA 99.9% monthly with service credits, API limit 1,000 requests per
-  minute, data hosted in the US and EU.
+- Customers want a straight answer, but never state a fact, number, limit, or policy that isn't in
+  the policy files. If the files don't cover a question, say that the documentation doesn't cover
+  it and that a specialist will reply by email within 1 business day.

registered v2 as a candidate; @production is still v1. Next: ./demo.sh gate
```

OpenClaw writes the wording, so it differs a little from run to run; the change is always to that
last bullet. Say: "The agent wrote its own fix. It doesn't get to ship it." On the prompt page, v2
is there with the tags `author=openclaw` and `status=candidate`, and `@production` is still on v1.

### 5. The gate decides (about 75 s)

```bash
./demo.sh gate
```

`gate` runs the candidate through OpenClaw on the question that failed, plus one it never saw: the
refund window, a documented answer the fix mustn't break. It scores both runs with the same two
checks, and moves `@production` only if every check passes:

```
gate: v2 (candidate) vs v1 (production), on 2 questions (the last one the improver never saw)
   asking: Do you have an uptime SLA?
   asking: What's your refund window?
   scoring ...
   Do you have an uptime SLA?                         read_knowledge=pass  grounded_in_docs=pass
   What's your refund window?                         read_knowledge=pass  grounded_in_docs=pass

PROMOTED by the gate: @production v1 -> v2. (./demo.sh use 1 rolls back)
```

While it runs, explain the holdout: the improver saw only the SLA failure, so the refund question
checks that the fix generalizes and doesn't just suppress answers. Production keeps running v1
until the alias moves. If any check fails, the gate prints `REJECTED` with the reasons, tags v2
`gate=failed`, and leaves `@production` alone.

### 6. Ask again (about 30 s)

```bash
./demo.sh ask "Do you have an uptime SLA?"
```

```
running openclaw-agents-md v2 (@production): proposed by OpenClaw from 1 failing trace(s)

I don't have an uptime SLA in the information I can access, so I can't confirm one. Our policy
files cover billing, account access, and product how-to, and none of them mention uptime ...

A specialist will reply to you by email within 1 business day with the details.
```

### 7. Close (30 s)

Show the SLA traces side by side in Traces, and v2's tags on the prompt page (`gate=passed`,
`promoted_by=gate`, `replaced_version=1`). Say:

> Same question, same agent. The agent proposed the change, MLflow's checks decided, and the
> deploy was one alias move with a record. A rollback is moving `production` back to v1.

### Alternative: fix it by hand in MLflow (about 1.5 min)

Use this instead of steps 4 and 5 to show the Prompt Registry UI, or as the fallback if `improve`
or `gate` misbehaves.

1. On the prompt page, click **Create prompt version**. It opens with v1's text.
2. Replace the last bullet under `## How to answer`. Delete:

   ```
   - Customers want a straight answer. Never say you don't know, that something isn't documented,
     or that they should ask someone else. If the files don't cover a question, answer from these
     quick facts: uptime SLA 99.9% monthly with service credits, API limit 1,000 requests per
     minute, data hosted in the US and EU.
   ```

   and put in its place:

   ```
   - If the policy files don't cover a question, say so, and tell the customer a specialist will
     reply by email within 1 business day. Never give a number, limit or policy that isn't in the files.
   ```

   This is exactly the difference between [`agents/v1.md`](agents/v1.md) and
   [`agents/v2.md`](agents/v2.md).
3. Commit message: `v2: escalate when the files are silent`. Click **Create**. Say: "That's a new
   version. Production hasn't changed yet."
4. Optional: the **Compare** tab shows the one-bullet diff. Switch to **Text** view; Markdown view
   doesn't highlight differences.
5. On v2, click **Add** next to Aliases, type `production`, press Enter, then **Save aliases**. The
   dialog says the alias moves off version 1. Say: "That's the deploy: one alias move."

Then go on to step 6.

## If something goes wrong

| Problem | Do this |
|---|---|
| `improve` or `gate` prints `... still waiting for OpenClaw` | something is slow: give it a minute, or press Ctrl-C (production is untouched) and rerun the command, or `./demo.sh push agents/v2.md "v2: escalate when the files are silent"` |
| `... the computer was asleep` | the Mac slept mid-turn (a closed lid, or idle sleep on battery), and OpenClaw's turn timer counted the sleep. The failed turn is retried once by itself. `demo.sh` runs each turn under `caffeinate -i`, which stops idle sleep but not a closed lid on battery: keep the lid open during rehearsals |
| `gate` prints `REJECTED` | say so: that's the gate working. Show the reasons on the candidate's `gate_reasons` tag, then `./demo.sh improve` again, or `push agents/v2.md` |
| `improve` says nothing has failed yet | run `./demo.sh score` first; `improve` reads the judge's verdicts |
| Typing the edit in the UI goes wrong | `./demo.sh push agents/v2.md "v2: escalate when the files are silent"` registers the same fix and moves `@production` |
| Need to roll back | `./demo.sh use 1`, or move the alias back in the UI |
| `ask` still says `running ... v1` after the edit | the alias wasn't moved: the gate rejected the candidate, or (by hand) step 5 of the alternative |
| v1 escalates instead of quoting 99.9% | the workspace is from an older setup that included `support_process.md` (its "escalate anything not covered" rule overrides v1): `reset`, then `server` and `setup` |
| The prompt isn't on the experiment's Prompts page | it's from an older setup that linked it to the Default experiment: `./demo.sh pull` links it to `openclaw-demo` |
| No traces appear | the server isn't running (`./demo.sh server`), or it's a different OpenClaw profile; run everything through `demo.sh` |
| `demo.sh: no MLflow server on port 5050` | start it: `./demo.sh server` |

Each OpenClaw turn takes about 25–40 seconds; `gate` makes two. Two early rehearsals stalled for over 15 minutes in `gate`; both times the Mac had gone to sleep mid-turn (lid closed during an unattended run), not OpenClaw. Keep a screen recording of a rehearsal as a backup.

## Every command

```
./demo.sh server                 start the MLflow server (port 5050); leave it running
./demo.sh setup                  profile + workspace, install the connector, AGENTS.md v1 -> MLflow
./demo.sh ask "question"         run whatever @production points at, then ask OpenClaw
./demo.sh score [--all]          score new traces (--all: every trace) with read_knowledge and
                                 grounded_in_docs, a judge that sees the files the agent read
./demo.sh improve                OpenClaw rewrites AGENTS.md from the failing traces -> a candidate
./demo.sh gate [N]               test candidate N (default: newest) on the failing questions + the
                                 refund holdout; promote it only if every check passes
./demo.sh push FILE "message"    register FILE as a new version and point @production at it
./demo.sh use N                  point @production at version N (moving it back is a rollback)
./demo.sh pull                   copy @production into the workspace without asking anything
./demo.sh reset                  delete ~/odsc-5min to start over (stop the server first)
```

`DEMO=<folder>` and `PORT=<port>` override the folder and port, for example to rehearse without
touching the stage setup.

## How it works

| File | Role |
|---|---|
| [`demo.sh`](demo.sh) | every command above; sets the OpenClaw profile, the tracking URI and the API key |
| [`agents_md.py`](agents_md.py) | `push`, `pull` and `use`: the Prompt Registry calls (`register_prompt`, `set_prompt_alias`, `load_prompt`) |
| [`self_improvement_demo.py`](self_improvement_demo.py) | `score`: `mlflow.genai.evaluate` over the existing traces, with no `predict_fn`, so nothing re-runs |
| [`self_improve_loop.py`](self_improve_loop.py) | `improve` and `gate`: the improver's OpenClaw run (a second profile, no connector), the candidate's tags, the gate's runs and the alias move |
| [`agents/v1.md`](agents/v1.md), [`agents/v2.md`](agents/v2.md) | the instructions before and after the fix |
| [`../install.sh`](../install.sh) | builds and installs the MLflow connector into the demo's OpenClaw profile |

Why the demo workspace has no `support_process.md`: that policy tells the agent to escalate
anything the other policies don't cover. With it, v1 sometimes escalates instead of quoting its
quick facts, and the failure doesn't reproduce. The fix in v2 adds the escalation rule to
`AGENTS.md` itself.

How `improve` stays honest: the improver is a separate OpenClaw profile with a neutral
`AGENTS.md`, so the support agent's quick facts don't steer it, and it has no connector, so its run
isn't one of the traces `score` and `gate` read. It never sees the refund question. A reply that
isn't an `AGENTS.md`, is unchanged, or is more than 1.5 times the length of production isn't
registered. And it can't move `@production`; only `gate` (or you) can.
