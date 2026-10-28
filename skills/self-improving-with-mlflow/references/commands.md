# `selfimprove self` commands (agent-facing)

These four are the only commands the deployed agent can run. Each prints one JSON object on
stdout and exits 0, including on a refusal: `{"ok": false, "refused": "..."}` is a normal answer,
to relay. Free text in the output is clipped (500 characters, 300 for gate summaries).

| command | use it when | writes |
|---|---|---|
| `self status` | asked which version you are, or why you changed | nothing |
| `self recall "<request>" [--k 3]` | before a task you may have got wrong before | nothing |
| `self note --trace previous --user-said ... --interpretation ... [--expected ...] [--session <key>]` | the user corrected your last answer | a `self_reported` note on that trace and a queue item for your owner |
| `self propose --target AGENTS.md\|skill:<name> (--text ... \| --file <path>) --why ... --evidence <id>,<id>` | two or more confirmed failures share a cause your instructions could fix | a candidate for the gate (AGENTS.md) or a proposal for a person (skill) |

## Output

`self status`

```json
{"ok": true,
 "prompt": {"name": "acme-support", "production": 7, "replaced": 6, "promoted_at": 1761500000000,
            "promoted_by": "gate", "promoted_because": "gate run 3f2a..."},
 "last_gate": {"candidate": "8", "result": "rejected", "summary": "gate FAIL: v8 vs v7 -- bloat: ...", "at": 1761...},
 "live": {"failure_rate": 0.12, "judged_traces": 40},
 "open_candidates": [{"version": 8, "author": "openclaw-self"}],
 "pending_self_reports": 2,
 "confirmed_reports": [{"trace_id": "tr-...", "interpretation": "I quoted the old 30-day window"}],
 "lessons": 3,
 "synced": {"version": 7, "at": 1761500100000}}
```

`self recall "how long do refunds take?"`

```json
{"ok": true, "source": "lessons@7", "lessons": [
  {"id": "L3fa2c", "lesson": "Read knowledge/billing.md before quoting a refund window.",
   "evidence": 4, "confirmed_by": "human"}]}
```

`self note ...`

```json
{"ok": true, "trace_id": "tr-...", "queue_item": "a1b2c3d4",
 "next": "a person confirms or rejects it (`selfimprove queue accept|reject a1b2c3d4`)"}
```

`self propose --target AGENTS.md ...`

```json
{"ok": true, "target": "AGENTS.md", "candidate_version": 9,
 "next": "the gate decides (`selfimprove gate`); you can't promote it"}
```

With `SI_SELF_PROPOSE_GATE=1` it also carries `"gate": {"result": "rejected", "summary": "..."}`.
A skill proposal returns `"target": "skill:<name>"` and a `queue_item` instead of a version.

## How `--trace previous` finds the answer

The turn you're in is still running, so its trace isn't exported yet: the newest production trace
in the last `SI_SELF_NOTE_WINDOW_MIN` minutes (default 30) is the answer being corrected. Pass
`--session <key>` when you know the conversation's `mlflow.trace.session` key; OpenClaw doesn't
put it in the exec environment, so with several people talking to you at once, prefer a trace id.

## Refusals you may see

| `refused` says | meaning | what to do |
|---|---|---|
| `give the user's exact words ...` | `--user-said` or `--interpretation` was blank | fill both in |
| `no production answer found ...` | nothing to attach the note to (too old, or not live traffic) | tell the user; their owner can label it |
| `already labeled` | a person already graded that answer | nothing; say so |
| `already reported` | a note for that answer is waiting | nothing; say it's pending |
| `note limit for this session` | too many notes in one conversation today | tell the user; their owner can review the conversation |
| `evidence needs at least N confirmed failures` | one correction is not a pattern | note it instead; don't propose |
| `evidence has no confirmed failure` | nobody has marked that trace wrong yet | wait for confirmation |
| `evidence includes held-out cases` | one of your evidence traces is evaluation data | drop the proposal; don't ask which |
| `too long` | the candidate grew past the length limit | shorten it only if the extra text was padding; otherwise stop |
| `lessons are rebuilt by your owner` | your proposal contains the lessons block | propose AGENTS.md as you see it in your workspace |
| `open proposal exists` | a proposal of yours for that target is already waiting | wait for the gate or the person |
| `no change` | the proposal matches what's live | nothing to do |
| `not available to you` | you ran something other than the four commands above | don't; say it's your owner's command |
