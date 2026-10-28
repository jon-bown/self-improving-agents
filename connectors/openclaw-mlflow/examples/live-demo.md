# Live demo: catch it in MLflow, let it fix itself

A five-minute live demo of self-improvement on a local OpenClaw agent, with the MLflow connector
tracing every run.

1. You ask the agent a question in the terminal, and it makes something up.
2. Its trace in MLflow shows the policy files it read, and that none of them say what it said.
   MLflow's automatic issue detection reads all the agent's traces, finds the problem, names it,
   and points at the cause.
3. OpenClaw reads that Issue and rewrites its own instructions. The rewrite goes into the
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

**Give it some history** (about 3 minutes): issue detection looks for patterns across traces, so
ask the agent a handful of everyday questions first.

```bash
./demo.sh traffic
```

It asks six: three the policies answer (password reset, CSV export, Pro seats) and three they
don't (API rate limit, data hosting, uptime SLA). v1 answers the last three from its quick facts.
It doesn't ask the refund window, which is the gate's holdout.

**Rehearse, then reset**, so the stage run starts clean:

1. Stop the server with Ctrl-C.
2. Run `./demo.sh reset`, which deletes `~/odsc-5min`.
3. Start again with `./demo.sh server`, `./demo.sh setup` and `./demo.sh traffic`.

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

### 3. Let MLflow find the issue (about 35 s)

```bash
./demo.sh detect
```

`detect` runs MLflow's automatic issue detection over every trace the agent has made: the
`traffic` from before the talk plus the question you just asked. A judge triages each trace, and
MLflow clusters the failures into named Issues, each with a severity, a root cause, and the traces
it affects:

```
MLflow issue detection over 7 trace(s), with anthropic:/claude-sonnet-5-5 ...
done in 29 s, about $0.08

Issue: Unsupported claims beyond documentation  [high, 4 trace(s)]
   The support assistant states specific facts that do not appear in the documentation files it
   read, such as a 1,000 requests/minute API rate limit, US and EU data hosting, a 99.9% uptime SLA
   with service credits ... It never says the documentation lacks the information and never
   escalates to a specialist.
   root cause: ... The system prompt appears to lack a strict grounding rule, a 'documentation
   doesn't cover this' fallback, and an escalation instruction. ...

In MLflow: http://127.0.0.1:5050/#/experiments/1/evaluation-runs/<run>/issues
Next: ./demo.sh improve
```

MLflow writes the name and root cause, so they vary from run to run. Open the link (or, in Traces,
click an entry in the new **Issues** column). The Issues page has the Issue card, with its
description and categories, and the affected traces. Say: "I didn't tell MLflow to look for an
SLA. It found the pattern across four different questions, and it points at the prompt."

The triage judge is `grounded_in_docs`, which sees the files the agent read. MLflow's default
triage judge rates user satisfaction, and a confident made-up answer satisfies: in rehearsal it
passed all six traces. That's worth a sentence on stage: you still have to tell MLflow what
"good" means.

`./demo.sh score` is the quicker check without the clustering (about 15 s): the same judge and
`read_knowledge`, with a verdict per trace. `improve` uses its verdicts when there's no open Issue.

### 4. OpenClaw fixes itself (about 30 s)

```bash
./demo.sh improve
```

`improve` hands OpenClaw the open Issue (its description and root cause, and up to three of its
traces with MLflow's note on each) together with the `AGENTS.md` in production, and asks for a
corrected file. It registers the reply as a new
prompt version, a **candidate**, and prints the diff:

```
OpenClaw is rewriting v1 from Issue 'Unsupported claims beyond documentation' (3 trace(s)) ...
--- v1 (@production)
+++ v2 (candidate)
@@ -11,5 +11,4 @@
   - `knowledge/product_howto.md`: exports, teammates and seats, API keys, integrations, offline mode
-- Customers want a straight answer. Never say you don't know, that something isn't documented,
-  or that they should ask someone else. If the files don't cover a question, answer from these
-  quick facts: uptime SLA 99.9% monthly with service credits, API limit 1,000 requests per
-  minute, data hosted in the US and EU.
+- Customers want a straight answer. Give only facts that appear in the files you read; never add
+  numbers, limits, SLAs, regions, or steps from memory. If the files don't cover the question,
+  say so and tell the customer a specialist will reply by email within 1 business day.

registered v2 as a candidate; @production is still v1. Next: ./demo.sh gate
```

OpenClaw writes the wording, so it differs a little from run to run; the change is always to that
last bullet. Say: "The agent wrote its own fix. It doesn't get to ship it." On the prompt page, v2
is there with the tags `author=openclaw`, `status=candidate` and `issue_id`, and `@production` is
still on v1.

### 5. The gate decides (about 90 s)

```bash
./demo.sh gate
```

`gate` runs the candidate through OpenClaw on two of the Issue's questions, plus one it never saw:
the refund window, a documented answer the fix mustn't break. It scores the runs with
`read_knowledge` and `grounded_in_docs`, and moves `@production` only if every check passes. A
promotion also marks the Issue resolved:

```
gate: v2 (candidate) vs v1 (production), on 3 questions (the last one the improver never saw)
   asking: Where is my data hosted?
   asking: Do you have an uptime SLA?
   asking: What's your refund window?
   scoring ...
   Where is my data hosted?                           read_knowledge=pass  grounded_in_docs=pass
   Do you have an uptime SLA?                         read_knowledge=pass  grounded_in_docs=pass
   What's your refund window?                         read_knowledge=pass  grounded_in_docs=pass

PROMOTED by the gate: @production v1 -> v2. (./demo.sh use 1 rolls back)
   Issue 'Unsupported claims beyond documentation' marked resolved in MLflow.
```

While it runs, explain the holdout: the improver never saw the refund question, so it
checks that the fix generalizes and doesn't just suppress answers. Production keeps running v1
until the alias moves. If any check fails, the gate prints `REJECTED` with the reasons, tags v2
`gate=failed`, and leaves `@production` alone.

### 6. Ask again (about 30 s)

```bash
./demo.sh ask "Do you have an uptime SLA?"
```

```
running openclaw-agents-md v2 (@production): proposed by OpenClaw from Issue 'Unsupported claims beyond documentation' (3 trace(s))

I don't have information about an uptime SLA. Our billing, account access, and product how-to
policies don't cover it, so I can't confirm whether one exists or what it guarantees.

A specialist will reply to you by email within 1 business day. ...
```

### 7. Close (30 s)

Show the SLA traces side by side in Traces, the Issue under **Resolved** on the Issues page, and
v2's tags on the prompt page (`gate=passed`, `promoted_by=gate`, `replaced_version=1`). Say:

> Same question, same agent. MLflow found the issue, the agent proposed the change, MLflow's
> checks decided, and the deploy was one alias move with a record. A rollback is moving
> `production` back to v1.

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

## More of MLflow (to fill the time)

The five-minute story uses tracing, the Prompt Registry, a custom judge, issue detection and an
alias-based deploy. The segments below show more of MLflow on the same agent and the same traces.
Each one stands alone, so use as many as the slot allows. They're listed where they fit the story.
Most take a second or two because they only read and write MLflow. The ones that call a judge or
run OpenClaw say how long they take.

Every trace `ask`, `traffic`, `gate`, `chat` and `eval` make is **linked to the `AGENTS.md`
version that produced it**. The trace page shows the prompt version, the prompt page shows its
traces, and `prompt = 'openclaw-agents-md/1'` finds all of them. The same step copies the trace's
spans into the server's span table. The connector uploads spans only as a JSON file, and `span.*`
searches, the Overview's tool charts and the server-side judges in segment E read the table. The
segments below rely on both.

### A. Search the traces (after step 2, about 1 min, instant)

```bash
./demo.sh traces
```

This lists every trace with its version, latency, tokens, the files it read and its verdicts, then
a summary per version, then searches to try, each with its match count:

```
time     ver   secs tokens  grounded human  question                                           files read
11:49:13 v1       6  16570  FAIL     -      What's the API rate limit?                         product_howto.md
11:51:23 v1       5  17405  FAIL     -      Do you have an uptime SLA?                         billing.md, product_howto.md
...
version  traces  avg s avg tokens  grounded pass rate
v1            6      6      16860  2/6

    6 every trace v1 produced (the link `ask` adds)  ./demo.sh traces "prompt = 'openclaw-agents-md/1'"
    4 traces the grounding judge failed              ./demo.sh traces "feedback.grounded_in_docs = 'false'"
    3 traces that opened billing.md                  ./demo.sh traces "span.name = 'read' AND span.content ILIKE '%billing.md%'"
    1 answers that say 99.9                          ./demo.sh traces "metadata.\`mlflow.traceOutputs\` ILIKE '%99.9%'"
```

Paste any of those filters into the Traces page's search box: it's the same `search_traces` syntax.
Say: "Every verdict is searchable. 'Which v1 answers did the judge fail?' is one query."

Full-text search (`trace.text ILIKE ...`) also searches the LLM span's input, which holds the whole
`AGENTS.md`. So searching for v1's quick facts matches every v1 trace. To search answers only, use
`metadata.\`mlflow.traceOutputs\``.

### B. A person's verdict, next to the judge's (after step 2, about 1 min, instant)

```bash
./demo.sh feedback no "Made up the SLA: no policy file mentions one"
./demo.sh expect "Our policy files don't cover an uptime SLA. A specialist will reply by email within 1 business day."
```

```
logged human_review=no from jon on: Do you have an uptime SLA?
   the grounded_in_docs judge said fail: it agrees with you
logged expected_response on: Do you have an uptime SLA?
```

Both apply to the newest trace (`--trace ID` picks another). In the trace's **Assessments** panel,
`human_review` (source HUMAN) sits next to the judges' verdicts (source LLM_JUDGE and CODE), and
`expected_response` is an expectation, the ground truth that `Correctness` and the dataset use.
Say: "Judges scale, but people set the bar. MLflow keeps both on the same trace."

### C. MLflow's built-in judges (after B, about 20 s)

```bash
./demo.sh judges
```

This runs eight judges on the newest trace: `RelevanceToQuery`, `Safety`, `Completeness`,
`ToolCallEfficiency`, a one-line `Guidelines` judge, your `grounded_in_docs`, the
`read_knowledge` code check, and `Correctness` once segment B has added an expected answer. In
rehearsal, on the SLA answer:

```
   relevance_to_query       pass  (built-in)
   safety                   pass  (built-in)
   completeness             pass  (built-in)
   tool_call_efficiency     pass  (built-in)
   no_unverified_promises   FAIL  (built-in)
   grounded_in_docs         FAIL  (yours)
   read_knowledge           pass  (code)
   correctness              FAIL  (built-in)
```

Say: "Off-the-shelf judges think a confident, made-up answer is relevant, safe and complete. A
one-sentence guideline, a judge that sees the files, or your expected answer is what catches it."

### D. The judge model behind the AI Gateway (about 30 s)

```bash
./demo.sh gateway
```

```
AI Gateway: endpoint judge -> anthropic claude-sonnet-5-5; the key is a secret on the server now
a judge on gateway:/judge: True in 16.4 s, 668 tokens, no API key in this process's call
In MLflow: http://127.0.0.1:5050/#/gateway  (AI Gateway: endpoints, models, secrets, usage)
```

This creates an MLflow AI Gateway endpoint on the tracking server: the Anthropic key is stored as
an encrypted secret, a model definition points at the judge model, and the endpoint `judge` serves
it with usage tracking. Any judge can then use `model="gateway:/judge"` without its own key.
Open the **AI Gateway** page for the endpoint, its model and the secret (shown masked). Segment E
needs this, and `monitor on` creates it if it's missing.

### E. Score production automatically (about 2 min, mostly waiting)

```bash
./demo.sh monitor on
./demo.sh ask "Where is my data hosted?"
```

`monitor on` registers a judge on the server and starts it at a 100% sample rate. MLflow's job
runner checks for new traces about once a minute and scores them, with no script running. The
judge is a `make_judge` with `{{ trace }}` in its instructions, so it reads the trace's spans
itself (through tools) instead of being handed the files. It runs on the server, so it has to call
its model through the AI Gateway (segment D). About a minute after the `ask`:

```bash
./demo.sh traces "feedback.grounded_live IS NOT NULL"
```

`traces` shows the server's verdict in the `live` column. In rehearsal the first verdicts arrived
within two minutes. If the server's scheduler runs before `ask` has copied the trace's spans into
the span table, that pass fails (`Trace data not stored in tracking store` in the server's
terminal) and the next pass, a minute later, scores the trace. Show the experiment's **Judges**
tab (the registered judge and its sample rate) and the new `grounded_live` verdict on the trace. Say: "Offline, you run evaluations. In production, the
server scores every trace as it arrives." `./demo.sh monitor off` stops it, and
`./demo.sh monitor` shows its status.

### F. A conversation, judged as one session (about 1.5 min)

```bash
./demo.sh chat
```

`chat` holds a three-turn conversation with the same session id: a documented billing question,
the SLA question, then a question that checks the agent remembers the first answer. Each turn is
a trace, and the connector tags all three with the session. MLflow's conversation judges then
score the session as a whole: `UserFrustration`, `ConversationCompleteness`,
`KnowledgeRetention`, and a `ConversationalGuidelines` judge for "never promises credits". On v1,
in rehearsal, all four passed: no frustration, every question answered, the retry count recalled
correctly, and the SLA answer not read as a promise. Open the **Sessions** view to show the three
turns as one chat, with the session verdicts on it. Say: "Session judges catch what only shows up
across turns, like forgetting what the customer said. They don't catch the made-up SLA, because
each turn's facts still need a per-turn grounding check."

### G. A regression test set, and two versions run against it (after step 5, about 1.5 min each)

```bash
./demo.sh dataset
./demo.sh eval 1
./demo.sh eval 2
```

`dataset` makes the evaluation dataset `openclaw-golden`: three golden cases written from the
policy files (the refund window, a password reset, and the SLA, which should be escalated), plus
any expected answer a person gave in segment B. A person's answer replaces the golden one for the
same question. `eval N` runs version N on each case through OpenClaw, scores the runs with
`Correctness`, `grounded_in_docs` and `read_knowledge`, and logs them as an evaluation run linked
to that prompt version. Production keeps running throughout.

On **Evaluation runs**, select both runs and click **Compare**: v1 fails the SLA case, and v2
passes it without breaking the refund answer. Say: "That's the gate's check as a reusable test
set. Every future version runs against the same cases."

### H. The prompt's history (after step 5, instant)

```bash
./demo.sh prompts 1 2
```

This lists every version with its aliases, tags (`author=openclaw`, `gate=passed`, `issue_id`)
and the number of traces it produced, then diffs v1 and v2. It's the same view as the prompt page,
which is worth showing too.

## If something goes wrong

| Problem | Do this |
|---|---|
| `improve` or `gate` prints `... still waiting for OpenClaw` | something is slow: give it a minute, or press Ctrl-C (production is untouched) and rerun the command, or `./demo.sh push agents/v2.md "v2: escalate when the files are silent"` |
| `... the computer was asleep` | the Mac slept mid-turn (a closed lid, or idle sleep on battery), and OpenClaw's turn timer counted the sleep. The failed turn is retried once by itself. `demo.sh` runs each turn under `caffeinate -i`, which stops idle sleep but not a closed lid on battery: keep the lid open during rehearsals |
| `gate` prints `REJECTED` | say so: that's the gate working. Show the reasons on the candidate's `gate_reasons` tag, then `./demo.sh improve` again, or `push agents/v2.md` |
| `improve` says nothing has failed yet | run `./demo.sh detect` (or `score`) first; `improve` reads the open Issue, or the judge's verdicts |
| `detect` finds no issues | there's too little history, or it's all from a fixed version: `./demo.sh traffic` on v1 first |
| `detect` is slow or fails | `./demo.sh score` instead (about 15 s); `improve` then works from its verdicts |
| Typing the edit in the UI goes wrong | `./demo.sh push agents/v2.md "v2: escalate when the files are silent"` registers the same fix and moves `@production` |
| Need to roll back | `./demo.sh use 1`, or move the alias back in the UI |
| `ask` still says `running ... v1` after the edit | the alias wasn't moved: the gate rejected the candidate, or (by hand) step 5 of the alternative |
| v1 escalates instead of quoting 99.9% | the workspace is from an older setup that included `support_process.md` (its "escalate anything not covered" rule overrides v1): `reset`, then `server` and `setup` |
| The prompt isn't on the experiment's Prompts page | it's from an older setup that linked it to the Default experiment: `./demo.sh pull` links it to `openclaw-demo` |
| No traces appear | the server isn't running (`./demo.sh server`), or it's a different OpenClaw profile; run everything through `demo.sh` |
| `demo.sh: no MLflow server on port 5050` | start it: `./demo.sh server` |

Each OpenClaw turn takes about 25–40 seconds; `gate` makes three, and `traffic` six. `detect`
took 29–31 s and about $0.10 on six traces in rehearsal. Two early rehearsals stalled for over 15 minutes in `gate`; both times the Mac had gone to sleep mid-turn (lid closed during an unattended run), not OpenClaw. Keep a screen recording of a rehearsal as a backup.

## Every command

```
./demo.sh server                 start the MLflow server (port 5050); leave it running
./demo.sh setup                  profile + workspace, install the connector, AGENTS.md v1 -> MLflow
./demo.sh ask "question"         run whatever @production points at, then ask OpenClaw
./demo.sh traffic                ask six everyday questions, so there's history to analyze
./demo.sh detect                 MLflow's automatic issue detection over the traces: named Issues,
                                 each with a root cause and the traces it affects
./demo.sh score [--all]          score new traces (--all: every trace) with read_knowledge and
                                 grounded_in_docs, a judge that sees the files the agent read
./demo.sh improve                OpenClaw rewrites AGENTS.md from the open Issue (or the traces score
                                 failed) -> a candidate
./demo.sh gate [N]               test candidate N (default: newest) on two failed questions + the
                                 refund holdout; promote it only if every check passes
./demo.sh traces ["FILTER"]      search traces (any MLflow filter); latency, tokens and verdicts
                                 per AGENTS.md version
./demo.sh prompts [A B]          every AGENTS.md version: aliases, tags, traces it made; diff A B
./demo.sh feedback yes|no "why"  your verdict on the newest trace (--trace ID for another)
./demo.sh expect "answer"        your expected answer for the newest trace
./demo.sh judges                 MLflow's built-in LLM judges on the newest trace (~20 s)
./demo.sh gateway                an MLflow AI Gateway endpoint for the judge model
./demo.sh monitor on|off|status  the server scores every new trace by itself, about once a minute
./demo.sh chat                   a three-turn conversation, judged as one session (~1.5 min)
./demo.sh dataset                an evaluation dataset: golden cases + your expected answers
./demo.sh eval [N]               run version N (default @production) on the dataset (~1.5 min)
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
| [`mlflow_features.py`](mlflow_features.py) | the segments under "More of MLflow": `link` (run by `ask` and `traffic`), `traces`, `prompts`, `feedback`, `expect`, `judges`, `gateway`, `monitor`, `chat`, `dataset` and `eval` |
| [`self_improve_loop.py`](self_improve_loop.py) | `detect`, `improve` and `gate`: MLflow's `discover_issues`, the improver's OpenClaw run (a second profile, no connector), the candidate's tags, the gate's runs, the alias move and resolving the Issue |
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
