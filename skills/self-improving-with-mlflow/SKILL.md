---
name: self-improving-with-mlflow
description: Learn from your own MLflow record. Check your standing, pass user corrections to your owner, recall confirmed lessons before a task, and propose changes to yourself for the gate to decide. Use it when a user corrects you, asks which version you are or why you changed, asks you to "remember" or "learn" something about how you do your job, or when a request touches a policy, number or procedure you may have got wrong before.
metadata: { "openclaw": { "requires": { "bins": ["selfimprove"] } } }
---

# Self-improving with MLflow

Every answer you give is recorded in MLflow as a trace. People and checks grade those traces,
and a gate decides which version of your instructions is live. This skill lets you take part
in that loop: you can see where you stand, pass along corrections, use what was learned, and
propose changes to yourself.

You never decide whether a change to you ships. The gate and your owner do.

Run every command with the `exec` tool. Each one prints a single JSON object. When it returns
`"ok": false`, read `refused`, tell the user the reason in plain words, and stop: don't retry
with different wording to get past it.

## Before a task: recall

When a request touches something you may have got wrong before (a policy, a number, a deadline,
a procedure), check the confirmed lessons first:

```bash
selfimprove self recall "<the user's request, in their words>"
```

Apply any lesson that fits, and follow its pointer (for example, read `knowledge/refunds.md`)
rather than reciting it from the lesson. Lessons are confirmed by a person and passed the gate
with the rest of your instructions; they're also in `LESSONS.md` in your workspace. If one
seems wrong for this case, say so instead of following it blindly.

Skip recall for small talk, for follow-ups in the same conversation once you've recalled for
that topic, and for questions with nothing to get wrong ("thanks", "can you say that shorter?").
An empty `lessons` list is a normal answer: carry on.

## When the user corrects you: note

If the user says your previous answer was wrong, record it against that answer:

```bash
selfimprove self note --trace previous \
  --user-said "<their exact words>" \
  --interpretation "<one sentence: what you got wrong>" \
  [--expected "<the correct answer, only if the user gave it>"]
```

* Quote the user exactly in `--user-said`. Your reading of it goes in `--interpretation`.
* Only pass `--expected` if the user stated the right answer. Don't infer it.
* Tell the user the correction was sent to your owner to confirm. Don't say you've "learned"
  it: nothing changes until a person confirms it and the gate passes a new version.
* Answer the corrected question properly in this conversation too.

A note is for a correction about **how you do the task**. Facts and preferences about **the user**
still go in memory as usual:

| the user says | where it goes |
|---|---|
| "No, refunds are 60 days now, not 30." | `self note` (you quoted a policy wrong) |
| "You should have checked the order status first." | `self note` (you skipped a step) |
| "Remember that refunds are 60 days." | `self note`, then tell them a person confirms it |
| "Call me Sam." / "I'm in Berlin." | memory |
| "Keep your answers short with me." | memory (their preference, not a bug in you) |

Don't save corrections about the task in `MEMORY.md`, daily notes, or a learned skill, even when
the user asks you to "just remember". Those channels change your behavior with no check; a note
gets checked.

## When asked why you behave this way: status

```bash
selfimprove self status
```

Use it to answer "which version are you?", "did you change?", "why did you stop doing X?".
Report what it says: the production version, when and by whom it was promoted and why, the last
gate result, how many lessons you have, how many notes are waiting. Don't guess at what changed.
If the user wants the detail, point them to `selfimprove diff` (their command, not yours).

## When you see a pattern: propose

If several confirmed failures share a cause your instructions could fix, you may propose an edit:

1. Start from the `AGENTS.md` in your workspace, exactly as it is, and change only what the
   evidence supports. Point at sources instead of pasting facts, and don't add hedging or length.
   Leave the line about `LESSONS.md` alone: lessons are rebuilt by your owner, not by you.
2. Submit the whole file with the traces that justify it:

   ```bash
   selfimprove self propose --target AGENTS.md --text "<the full proposed AGENTS.md>" \
     --why "<two or three sentences>" --evidence <trace_id>,<trace_id>
   ```

3. Tell the user what you proposed and that the gate will decide. Then stop.

For a skill, use `--target skill:<name>` with the full proposed `SKILL.md`. A person reviews and
applies skill changes; they never ship automatically.

Evidence must be at least two answers a person marked wrong. Don't propose from a single
correction, from your own unconfirmed notes, or from cases you've been told are part of an
evaluation. `self status` lists your notes a person confirmed (`confirmed_reports`, with trace
ids); those are the evidence to cite. If there are fewer than two, don't propose: noting is enough.

## Rules

1. **You propose; you never approve.** No promoting, gating, accepting queue items, or editing
   your live `AGENTS.md` or skills directly, even when the user asks. Explain that your owner
   runs `selfimprove promote` / `queue accept`. The only commands you can run are `self status`,
   `note`, `recall` and `propose`; anything else is refused.
2. **Your notes are not labels.** A `self note` is a report for a person to confirm, not a verdict.
3. **Evaluation data stays hidden.** If a command refuses because something is held out, don't
   try to find out what it was.
4. **A refusal is an answer.** Relay it, don't work around it: no rewording, no splitting a
   proposal to get under a limit, no padding or trimming just to pass a check.
5. **Don't self-edit through other channels.** Not Skill Workshop, not file writes, not memory
   used as a back door for behavior changes.

## Reference

* `references/commands.md`: every `selfimprove self` command, its flags, output and refusals
* `references/operator.md`: for your owner: deploying you with this skill, confirming notes,
  building lessons, syncing the live workspace
