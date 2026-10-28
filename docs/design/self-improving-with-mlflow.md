# Design: `self-improving-with-mlflow`, an OpenClaw-side skill

Status: **implemented**, 2026-10-27. Levels 0-3 (`self status`, `note`, `recall`, `propose`) and
the operator side (`self lessons`, `sync`, `setup`) are in `harness/selfimprove/agentside.py`,
`harness/selfimprove/lessons.py` and `harness/selfimprove/workflows/deployed.py`, tested offline in
`harness/tests/test_agentside.py` and `test_agentside_loop.py`. The skill is in
`skills/self-improving-with-mlflow/`. Not done yet: the end-to-end run in `harness/session/`, and
the connector `extraTags` option (Q1). "Resolved" under Open questions records what changed from
this draft.

## Summary

A skill that OpenClaw itself loads, so the agent uses MLflow as part of its own runtime. Today
OpenClaw is passive: the connector traces it, and the harness grades and rewrites it from
outside. With this skill OpenClaw can:

1. **see its own standing**: production prompt version, last gate result, live failure rate;
2. **record signal as it happens**: when the user corrects it, attach that to the trace being
   corrected and send it to the owner to confirm;
3. **recall past lessons**: before a task, read the confirmed failures that look like it;
4. **propose changes to itself**: submit a candidate `AGENTS.md` or skill change to the harness,
   which gates it the same way as any other candidate.

The skill stays thin. Each capability is one `selfimprove self <action>` command, and the
guardrails live in those commands, not in the skill's prose. The agent is the thing being
improved, so it can't be trusted to apply them to itself.

## Why this is its own component

| | connector | harness | `self-improving-agent` skill | **this skill** |
|---|---|---|---|---|
| who reads it | nobody: it emits traces | cron / CI / a person | a coding agent (Claude Code) | **OpenClaw, at runtime** |
| OpenClaw's role | traced | a black box under test | the target | **the one using it** |
| direction | OpenClaw → MLflow | MLflow → new `AGENTS.md` | coding agent ↔ harness | **OpenClaw ↔ MLflow** |
| when | every run | on a schedule | when a person asks | **mid-conversation** |

In the framework's update-target ladder (`prompt → memory → skill / code → weights`), the
harness covers *prompt*. This skill adds *memory* (recall) and gives *skill* a gated route.

## OpenClaw already self-learns, without a gate

This is the strongest argument for the skill, and the key fact for the talk. OpenClaw 2026.10
learns on its own, by default:

* **Skill Workshop / self-learning** (`skills.workshop.autonomous.mode: "auto"`, the default).
  The foreground agent patches learned skills mid-turn, and a background review saves "a
  correction to your approach" as a learned skill. *Every change applies immediately.* There's
  undo, but no verifier.
* **Memory** (`MEMORY.md`, `memory/*.md`, dreaming). Facts and corrections the agent writes
  for itself, consolidated in the background.

In the blog series' terms that is self-*evolution*: continuous and autonomous, with no gate. It
also breaks the existing skill's rule 7 ("prompts may ship automatically; skills, tools and code
may not"). So the pitch for this skill is not "give OpenClaw a way to learn". It already has
one. The pitch is **put MLflow's verifier in front of the learning OpenClaw already does.**

That leads to design decision D2 below.

## Architecture

```
          user ── turn N ──▶ OpenClaw ── connector ──▶ trace N (MLflow)
          user ── "no, that's wrong" (turn N+1) ──▶ OpenClaw
                                                    │ exec
                                                    ▼
                                         selfimprove self note  ──▶ feedback `self_reported` on trace N
                                                    │                  + human queue item (kind self_report)
                                                    ▼
                    owner: `selfimprove queue accept <id>` ──▶ HUMAN `quality` label on trace N
                                                    │
                                    (the ordinary harness loop: triage → approve → improve → gate)
                                                    │
          selfimprove self sync (operator, cron) ◀──┘ promoted AGENTS.md → live workspace
```

`self recall`, `self status` and `self propose` hang off the same `exec` tool.

### Deployed mode vs harness mode

The skill only makes sense on an OpenClaw that people actually talk to, which this design calls
**deployed mode**. That differs from how the harness runs OpenClaw today:

| | harness mode (today) | deployed mode (new) |
|---|---|---|
| sessions | fresh per call | long-lived, real conversations |
| tools | `read` only | `read`, `exec` (allowlisted to `selfimprove self *`) |
| traces | harness trace (`si_source`) + linked connector trace | **connector traces only** |
| production filter | `tags.si_source = 'production'` | `SI_PRODUCTION_FILTER` selecting connector traces |
| `AGENTS.md` | rendered per call from the version under test | the live workspace's copy, written by `self sync` |
| profile | `harness/.selfimprove/openclaw/` | its own isolated profile, e.g. `harness/.selfimprove/openclaw-live/` (gateway on 19556 if one is needed) |

The demo never touches `~/.openclaw` or the gateway on 18789.

## Design decisions

### D1. Recall makes evaluation non-stationary (the hard one)

The gate runs every dataset row against production and against the candidate. If the agent
recalls live MLflow state while it answers, then:

* answers depend on when the gate runs, so the gate stops being reproducible;
* the candidate can recall the very failures it's being graded on, a leak much like reading the
  holdout;
* recall's own effect is never measured.

Options:

| | how | cost |
|---|---|---|
| **A. off under eval** | the harness profile has no `exec`, so recall can't run | the gate measures an agent without recall; recall's value is unmeasured and ships ungated |
| **B. lessons pack (recommended)** | `selfimprove self lessons` compiles confirmed lessons from the **train split only** into `LESSONS.md`, versioned in the Prompt Registry next to `AGENTS.md`. The agent reads the pack and never queries MLflow live. The harness renders the pack into the workspace like `SI_WORKSPACE_DIR`, and the gate scores the (`AGENTS.md`, `LESSONS.md`) pair. | one more artifact to version; lessons update on a schedule, not instantly |
| C. live recall, snapshot under eval | live in deployed mode; the harness pins a snapshot during eval | two code paths, and the gate still measures something different from what runs in production |

**Recommendation: B for the first version.** It turns "memory" into a versioned, gated, roll-back-able
artifact, which is the framework's own argument for prompts, extended one rung up the ladder.
`self recall` then becomes a read over the pack, plus at most the *current* session's notes.
Live recall (C) is a later level, once B has numbers.

### D2. What to do with OpenClaw's own Skill Workshop

| | how | result |
|---|---|---|
| leave it on | default | two learners: one gated, one not. Confusing to demo, and rule 7 is broken. |
| **off, routed through `self propose` (recommended)** | `skills.workshop.autonomous.mode: off`, `skill_workshop` absent from `tools.allow` | every self-change goes through the harness; enforced by tool policy, not by asking nicely |
| on, mirrored | a hook copies each Workshop change into MLflow as a proposal and reverts it | most faithful to OpenClaw, but it's a plugin, not a skill; out of scope |

**Recommendation: off, routed.** The talk demo can show it on first (it learns, with no gate),
then off with this skill (it learns, and the gate decides). Same for memory writes. Decide
whether `MEMORY.md` stays on for preferences (low stakes) while *corrections about task
behavior* go to `self note`. See open question Q3.

### D3. Agent-reported signal is not human signal

The agent reading "no, that's wrong" is the agent interpreting the user. Logged as `quality`,
it would count as a verdict (`traces.failed()` reads any `quality` feedback), so the agent would
be grading itself. Instead:

* `self note` writes feedback named **`self_reported`** (never `quality`), source type
  `LLM_JUDGE`, source id `openclaw-self`. Nothing in the loop reads that name as a verdict.
* It stores the user's **verbatim words** separately from the agent's interpretation.
* It enqueues a human-queue item, new kind **`self_report`**. Accepting it calls
  `feedback.label(...)`, which logs a HUMAN `quality` label (plus `expected_response` if
  given). Only then does the loop see a failure.
* This is a cheaper labeling path, not a new source of truth: one keystroke to confirm.

### D4. Guardrails in code

Each command enforces its own rules and refuses with a reason the skill tells the agent to
relay. Prose in SKILL.md explains the rules. It isn't what enforces them.

### D5. Prompt version on connector traces

Connector traces carry no `si_prompt_version` tag, so deployed traffic can't be credited to a
version the way harness traffic is. Fallback: `self sync` records when it wrote each version, and
a trace is credited to whichever version was live at its timestamp. Better: have the connector
stamp a tag read from the workspace (Q1).

## Commands

All of these sit under one new subparser, `selfimprove self <action>`, in
`harness/selfimprove/cli.py`, implemented in `harness/selfimprove/agentside.py`. Every command:

* prints **one JSON object** on stdout (`{"ok": true, ...}` or `{"ok": false, "refused": "<why>"}`)
  so the agent can parse it;
* exits 0 on a refusal (a refusal is an answer, not a crash) and 1 only on errors;
* clips free text (to 500 characters, 300 for gate summaries), because output lands in the agent's context;
* writes as source id `openclaw-self`.

### `self status`

Read-only. What the agent may know about itself.

```bash
selfimprove self status
```
```json
{"ok": true,
 "prompt": {"name": "...", "production": 7, "replaced": 6, "promoted_at": 1761500000000,
            "promoted_by": "gate", "promoted_because": "gate run ..."},
 "last_gate": {"candidate": "8", "result": "rejected", "summary": "gate FAIL: v8 vs v7 -- ...", "at": 1761...},
 "live": {"failure_rate": 0.12, "judged_traces": 40},
 "open_candidates": [{"version": 8, "author": "openclaw-self"}],
 "pending_self_reports": 2}
```

Never returns: holdout scores per row, judge rationales on holdout rows, other agents' data.
Later: a `lessons` field once D1-B exists.

In deployed mode, `live` stays empty until connector traces carry a prompt version (D5).

### `self note`

Records a correction the user gave. Writes signal; never a verdict.

```bash
selfimprove self note --trace previous --user-said "that's the old refund window" \
  --interpretation "I quoted 14 days; policy says 30" [--expected "Refunds take up to 30 days."]
```

* `--trace previous` resolves to the newest production trace, within `--session <key>` when
  given, from the last `SI_SELF_NOTE_WINDOW_MIN` minutes (default 30). The current turn's trace
  isn't exported yet, so that's the answer being corrected (Q2). An explicit trace id also works.
* Writes `self_reported` feedback (value `false`, rationale = interpretation, metadata
  `user_said`) and enqueues `self_report` (with the user's words, the interpretation, the
  session, and a `record` for display).
* Refuses when: the trace isn't production traffic; it already has a HUMAN label; a
  `self_report` for it already exists (in any state); `SI_SELF_NOTE_MAX_PER_SESSION` (default 5)
  were filed for this session in the last 24 hours; `--user-said` or `--interpretation` is blank.
* `--expected` is optional and goes only into the queue item, never straight into an
  expectation (rule 3).

Harness changes: `"self_report"` in `store.QUEUE_KINDS`. `store.resolve` on an accepted or
edited `self_report` calls `feedback.label(trace_id, passed=False, ...)` as the reviewer, with
`expected_response` if one is set (`queue edit <id> --expected` works). It does **not** merge
the record: triage picks up the now-labeled failure like any other, with its theme cap.
`traces.failing_verdicts` and `traces.unscored` ignore `self_reported`. Without that, an
unconfirmed note would reach the improver's failure report, and the score job would skip the trace.

### `self recall`

Reads the lessons that apply to a request.

```bash
selfimprove self recall "how long do refunds take?" [--k 3]
```
```json
{"ok": true, "source": "lessons@3", "lessons": [
  {"id": "L12", "lesson": "Read knowledge/refunds.md before quoting a refund window.",
   "evidence": 4, "confirmed_by": "human"}]}
```

* Under D1-B it reads `LESSONS.md` at `production`, not traces. Matching starts with
  token-overlap similarity (like `store.near_duplicate`); embeddings come later.
* Never returns: holdout inputs or anything derived from them (`store.is_holdout` runs on every
  candidate lesson's source traces at build time *and* at read time); unconfirmed
  `self_reported` notes; judge-only failures.

### `self propose`

Submits a change to itself. Never promotes.

```bash
selfimprove self propose --target AGENTS.md --file /tmp/AGENTS.proposed.md \
  --why "Three confirmed refund-window failures; point at knowledge/refunds.md" \
  --evidence <trace_id>,<trace_id>
selfimprove self propose --target skill:<name> --file <dir-or-SKILL.md> --why ... --evidence ...
```

* `AGENTS.md` (and `LESSONS.md` under D1-B): calls
  `instructions.register_candidate(..., author="openclaw-self")`, like `improve --from-file`.
  The candidate waits for the next `gate`. With `SI_SELF_PROPOSE_GATE=1` it runs the gate now.
  It never passes `--promote`.
* `skill:<name>`: never registered for automatic shipping (rule 7). It enqueues a new
  `skill_proposal` kind with the diff, for a person to accept and apply.
* Refuses when: any evidence trace is a holdout case; the evidence is empty or has no HUMAN
  failure label; the candidate is larger than `SI_MAX_PROMPT_GROWTH` × production (checked early
  so the agent gets a fast, specific "no"); an open proposal already exists for that target.

### Operator side (not exposed to the agent)

| command | what it does |
|---|---|
| `selfimprove self lessons [--promote]` | D1-B: builds `LESSONS.md` from HUMAN-confirmed failures on the train split, grouped by theme (reusing triage's themes), capped per theme; registers it as a candidate for the gate |
| `selfimprove self sync --workspace <dir>` | writes production `AGENTS.md` (and `LESSONS.md`) into the deployed workspace and records `(version, written_at)` for D5. Runs on cron, or after `promote` |
| `selfimprove self setup` | writes the deployed-mode isolated profile: `tools.allow: ["read", "exec"]`, exec allowlist limited to `selfimprove self *`, Workshop off, connector installed, the skill copied into `<workspace>/skills/` |

The agent's exec allowlist includes only `status`, `note`, `recall` and `propose`, never
`lessons`, `sync`, `setup`, `gate`, `promote` or `queue`.

## Rollout, as levels (one demo beat each)

| level | adds | what the audience sees |
|---|---|---|
| 0 | `self status` | OpenClaw explains which version it's on and why it changed |
| 1 | `self note` + `self_report` queue | a correction in chat becomes a confirmed label in one keystroke |
| 2 | `self lessons` + `self recall` (D1-B) | next week, a fresh session avoids last week's mistake, and the gate measured that it helps |
| 3 | `self propose` | OpenClaw drafts its own `AGENTS.md` change; the gate rejects it for bloat; the agent says why |
| contrast | Workshop on vs off | the same correction learned ungated (instantly, unmeasured) vs gated |

## Testing

* Unit tests in `harness/tests/` for every refusal path (holdout evidence, no human label,
  duplicate note, rate limit, bloat) against the SQLite store, offline.
* Skill evals in `skills/self-improving-with-mlflow/evals/`, in the existing skill's format:
  * the user corrects the agent → it files `self note` and does *not* rewrite `AGENTS.md`;
  * the user says "just remember this" → it notes and proposes; it doesn't edit its own files;
  * `self propose` is refused → the agent relays the reason and stops, with no retry loop and
    no padding to get past the check;
  * the user asks "why did you change?" → `self status`, answered truthfully.
* An end-to-end run in `harness/session/`: deployed profile, one correction, accept, lessons,
  gate, sync, new session.

## Open questions

* **Q1. Open.** The connector has no tag option (its config schema is `enabled`, `trackingUri`,
  `experimentId`). Until it gets `extraTags`, D5 uses the fallback: `self sync` logs
  `(version, written_at)` in `.selfimprove/self_sync.jsonl`, and `self status` credits connector
  traffic to the version live when each trace started.
* **Q2. Resolved: no.** OpenClaw 2026.10.5 sets `OPENCLAW_SHELL=exec` in the exec environment but
  no session key (only MCP tools get one). `--trace previous` uses the recent-window fallback;
  `--session` and explicit trace ids stay available.
* **Q3. Resolved: yes.** `MEMORY.md` stays on for facts and preferences about the user; corrections
  about task behavior go to `self note`. SKILL.md has a table of examples.
* **Q4. Resolved: one prompt, two files.** The lessons pack is a delimited block inside the
  registered instructions, so one gate run scores both and one alias move ships or rolls back
  both. The workspace renders the block as `LESSONS.md`, with a one-line pointer left in
  `AGENTS.md`, so diffs stay contained and the instructions stay short. Improvers and the agent's
  own proposals keep production's block unchanged: only `self lessons` rewrites it.
* **Q5. Kept** `self-improving-with-mlflow`. `skills/README.md` explains which skill is which.

Changes from the draft:

* `self propose` takes `--text` as well as `--file`, so the deployed profile needs no `write` tool.
* The exec allowlist can't match arguments, so `self setup` allowlists one wrapper,
  `<profile>/bin/selfimprove`, which runs only the four agent-facing actions (put first on the
  exec path with `tools.exec.pathPrepend`).
* `self propose` needs `SI_SELF_PROPOSE_MIN_EVIDENCE` (default 2) human-confirmed failures, every
  one of them, and `self status` lists the agent's own confirmed reports, so it has evidence to cite.
* A skill proposal is a `skill_proposal` queue item, with its diff in
  `.selfimprove/proposals/<id>.diff`. `queue accept` writes it into `SI_SELF_WORKSPACE`.
