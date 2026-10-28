# Runbook: the four loops from the command line

The same four loops as the notebooks, as `selfimprove` commands. Each loop: **find** a failure in
OpenClaw's answers, turn it into **signal**, add it to the **store** as tests, make the
**update**, and let the **gate** decide. From loop to loop, more of that moves from you to the
harness.

| Loop | Failure | Signal | Store | Update | Promotes |
|---|---|---|---|---|---|
| 1 | Skips the lookup | you label; `read_knowledge` checks the connector's TOOL spans | you add the failures as tests | you write v2 | you |
| 2 | Improvises when the docs are silent | + `grounded_in_docs` judges the answer against what it read | triage + approve | the analyst | you |
| 3 | Unsafe account help, after the support team's edit | + must-pass safety check; the judge is aligned to your labels | triage + approve | an optimizer | you |
| 4 | Bad escalation, after the growth team's edit | + `no_commercial_promises`; scheduled scoring; a 50% label sample | `cycle` | `cycle` (analyst) | the gate; `monitor` can roll back |

Running everything live takes about two hours (each OpenClaw call is about 20 s, and the gate
runs every dataset row twice). For the talk, `session/store/` already holds a full run: start
the server and open the notebooks, or the MLflow UI at http://127.0.0.1:19560.

## Setup

From `harness/`, with `ANTHROPIC_API_KEY` in `harness/.env` and OpenClaw installed
(`npm i -g openclaw`):

```bash
make session-server
```

Leave it running and use a second terminal for the rest. To start from an empty store instead of
the precomputed one, stop the server, run `.venv/bin/python -m session.precompute --fresh`, and
start it again.

```bash
export SI_ENV_FILE=session/session.env
export SI=.venv/bin/selfimprove
```

`doctor` writes the isolated OpenClaw profile in `session/.state/openclaw/` (never `~/.openclaw`)
and installs the MLflow connector into it, so every OpenClaw run also logs its own trace with
the LLM and TOOL spans. `init` registers `session/agent/AGENTS.md` as v1 in the Prompt Registry
and adds the eight golden holdout cases to the dataset.

```bash
$SI doctor --smoke "How long do I have to request a refund?"
$SI init
```

Paths in `SI_*` variables are relative to `session/`; file arguments to `replay` and
`--from-file` are relative to `harness/`.

## Loop 1 · Manual: the agent skips the lookup

v1 answers from stale "quick facts" in AGENTS.md instead of reading `knowledge/`.

**Find.** Send the loop's traffic to production and score it:

```bash
export SI_CRITERIA_FILE=criteria/loop1.yaml
$SI replay session/traffic/loop1.txt
$SI score
```

**Signal.** You are the reviewer. Label each trace pass or fail, with the correct answer for each
failure. You can do it in the terminal, in the MLflow UI, or let the simulated reviewers do it:

```bash
$SI label
```

```bash
.venv/bin/python -m session.reviewer
```

In the UI, open a failing trace's **linked OpenClaw trace** (search
``metadata.`mlflow.trace.session` = '<openclaw.session_key>'``): it has an `llm_call` and no
`read` span. `read_knowledge` is that check in code.

**Store.** You add the labeled failures as tests (`store.record_from_trace`, as in notebook 01):

```bash
.venv/bin/python -c "
from session.precompute import configure; configure(1)
from selfimprove import store, traces
fails = [t for t in traces.production(traces.FAILED + ' AND ' + traces.HAS_EXPECTATION)]
print(store.merge([store.record_from_trace(t, source='human_fail', author='you', theme='skips_lookup') for t in fails]), 'tests added')
store.save_checkpoint('triage')"
```

**Update.** You write v2: read the files by name, drop the quick facts.

```bash
$SI improve --from-file session/fixes/v2_manual.md -m "v2: read the policy files by name, drop the stale quick facts"
$SI diff production latest
```

**Gate.** Production and v2 run on the same rows (train and holdout) with the same scorers:

```bash
$SI gate
```

If it passes, you promote it. Then send the same traffic again and compare:

```bash
$SI promote 2 --reason "loop 1: gate passed"
$SI replay session/traffic/loop1.txt
.venv/bin/python -m session.reviewer
$SI report
```

## Loop 2 · Assisted: the agent improvises when the docs are silent

v2 reads the files now, but when they don't cover a question it hedges, sends the customer
elsewhere, or pads the answer, instead of escalating to a specialist. The new verifier, `grounded_in_docs`, judges the answer against the files the agent
actually read, taken from the connector's `read` spans.

```bash
export SI_CRITERIA_FILE=criteria/loop2.yaml
$SI replay session/traffic/loop2.txt
$SI score
.venv/bin/python -m session.reviewer
```

**Store, by the harness.** Triage reads the new failures, groups them into themes with an analyst,
and proposes one test per failure (capped per theme). Approve checks each one (does it reproduce?
is it a near duplicate? would a reviewer agent sign off?) and merges it or queues it for you:

```bash
$SI triage --force
$SI approve
$SI queue list
```

**Update, by the analyst.** It reads the failure report and the current AGENTS.md and writes a
candidate. It never promotes:

```bash
$SI improve --improver analyst
$SI diff production latest
$SI gate
```

If the gate rejects the candidate, review the human queue: the approver's flagged tests wait
there. Accept the ones that are right (they join the dataset), then let the analyst try again:

```bash
$SI queue accept <id> --note "checked against knowledge/"
$SI improve --improver analyst
$SI gate
```

You review the proposal that passes and promote it (`$SI promote <version>`).

## Loop 3 · Optimizer: unsafe account help

First, the support team adds "faster account recovery" to AGENTS.md and ships it without the
gate (`changes/support_team_fast_recovery.md`, appended to production and promoted):

```bash
.venv/bin/python -c "from session.precompute import configure, outside_change; configure(3); outside_change('changes/support_team_fast_recovery.md', 'support-team', 'Faster account recovery: verify customers in chat')"
```

Now the agent asks for passwords and says it has reset 2FA. `no_credential_requests` is
**must-pass**: a candidate that fails it on any row never ships. The `quality` judge sees only the
question and the answer, not the policy, so first check how often it agrees with the reviewers
and align it to them:

```bash
export SI_CRITERIA_FILE=criteria/loop3.yaml
$SI replay session/traffic/loop3.txt
$SI score
.venv/bin/python -m session.reviewer
$SI align --force
```

**Store, then the optimizer.** MetaPrompt rewrites AGENTS.md against every scorer on the train
split. The holdout stays hidden from it; only the gate sees it:

```bash
$SI triage --force
$SI approve
$SI improve --improver metaprompt
$SI gate
```

If the gate rejects it, read why. Optimizers grow prompts, and `SI_MAX_PROMPT_GROWTH` (2.0 by
default) stops a candidate that more than doubles production. If the growth is real content (in
the precomputed run, a new security section on a short prompt), a person can raise the limit for
this agent and re-run the gate on the same version (`$SI gate <version>`). Otherwise, try GEPA
(slower: every metric call is an OpenClaw run). You promote whichever candidate passes.

```bash
$SI improve --improver gepa --budget 40
$SI gate
```

## Loop 4 · Autonomous: bad escalation on new traffic

The growth team adds a retention playbook (discounts, roadmap dates) and ships it without the
gate:

```bash
.venv/bin/python -c "from session.precompute import configure, outside_change; configure(4); outside_change('changes/growth_team_retention.md', 'growth-team', 'Retention playbook: discounts and roadmap dates')"
```

Then new kinds of questions arrive: discounts, roadmap dates, contracts. Reviewers label only half of
them. `cycle` runs score, monitor, align, triage, approve, improve and gate in one MLflow run, and
with `SI_AUTO_PROMOTE=true` the gate promotes:

```bash
export SI_CRITERIA_FILE=criteria/loop4.yaml SI_AUTO_PROMOTE=true
$SI replay session/traffic/loop4.txt
.venv/bin/python -m session.reviewer --sample 0.5
$SI cycle
```

After a gate promotion, `monitor` compares live traffic on the new version with the failure rate
it replaced, and rolls back if it got worse by more than `SI_ROLLBACK_MARGIN`:

```bash
$SI replay session/traffic/loop4.txt
$SI score
$SI monitor
$SI report
```

The schedule for this lives in `.github/workflows/selfimprove-loop.yml` (nightly jobs, a weekly
`cycle`). What stays with a person: the golden cases, labels on a sample of traffic, the human
queue, the must-pass rules and the margin.
