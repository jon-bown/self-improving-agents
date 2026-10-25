# Running the loop nightly in GitHub Actions, then letting it promote

You're at level 2 now: the analyst writes candidates and you approve them. You want level 4: a scheduled `cycle`, and the gate approving (`SI_AUTO_PROMOTE=true`). Do it in two moves. **First, schedule the loop with auto-promote off** and let it run unattended for a while. **Then turn auto-promote on** once the record shows the gate decides the way you would. The loop is the same at every level. The only change is who approves.

Most of the plumbing is already in the repo: `.github/workflows/selfimprove-loop.yml` is the scheduled loop and `.github/workflows/selfimprove-regression.yml` is the PR gate. Both are off until you opt in.

---

## Part 1: What to set up for the nightly run

### 1. The shared MLflow server is the single source of truth

- **The backend must be a database, not file storage.** It must be reachable from GitHub-hosted runners (or use a self-hosted runner inside your network), and it needs auth.
- **The support bot in production must log its traces to the same server and experiment** (`SI_EXPERIMENT`). The loop only learns from what it can see there. If the bot traces itself, set `SI_PRODUCTION_FILTER` (for example `tags.environment = 'production'`) so `score`, `triage`, `align` and `monitor` can find its traces. Otherwise its traces need `tags.si_source = 'production'`.
- **The bot must load its instructions by alias:** `prompts:/<SI_PROMPT_NAME>@production`. If it reads a file or a pinned version, a promotion moves an alias that nobody reads. Rollback then does nothing either. Check that a changed alias actually reaches the running bot, and how quickly it does if the bot caches prompts.
- **Move your history over.** Your manual runs may have used the default `sqlite:///.selfimprove/mlflow.db`. If so, your prompt versions, dataset (with its holdout split), labels and judge-agreement history are local. Move them to the shared server before the first scheduled run, or re-run `selfimprove init` there and re-import. Don't let CI start from an empty store while your real history sits on a laptop.

### 2. Repository secrets and variables

The workflow already reads these:

| Kind | Name | Value |
|---|---|---|
| variable | `SELFIMPROVE_ENABLED` | `true` (the master switch) |
| secret | `MLFLOW_TRACKING_URI` | the shared server |
| secret | `ANTHROPIC_API_KEY` (and/or `OPENAI_API_KEY`) | for the judge, the analyst, the approver, and the bot if it calls a model |
| variable | `SI_EXPERIMENT` | your support bot's experiment |
| variable | `SI_IMPROVER` | `analyst` (what you use today) |
| variable | `SI_AUTO_PROMOTE` | **`false` for now** |
| optional | `SI_FAST_JUDGE_MODEL`, `TYPESAFE_API_KEY` | cheap first-pass judge (Jev), if traffic gets large |

### 3. Gaps in the workflow you'll need to close

- **Your bot's settings aren't in CI yet.** The workflow runs `cp .env.example .env`, so every setting it doesn't pass explicitly falls back to the example agent. That covers `SI_AGENT`, `SI_AGENT_TARGET`/`SI_AGENT_URL`, `SI_PROMPT_NAME`, `SI_INSTRUCTIONS_FILE`, `SI_CRITERIA_FILE`, `SI_GOLDEN_FILE`, `SI_AGENT_NAME`, `SI_INPUT_KEY`, `SI_PRODUCTION_FILTER` and the thresholds. Environment variables take precedence over `.env`. So add your bot's values to the workflow's `env:` block (as repository variables), or commit a non-secret config that is copied instead of `.env.example`.
- **Fail fast if the tracking URI is missing.** If the secret is absent, `MLFLOW_TRACKING_URI` is set to an empty string and the harness won't fall back to `.env`. MLflow would then write to a throwaway local store on the runner, and the run would look green. Add a first step such as `test -n "$MLFLOW_TRACKING_URI" || exit 1`.
- **The runner must be able to call the bot.** The gate and the improvers run candidates; they don't just read traces.
  - With `SI_AGENT=python`, install the bot's package and dependencies in the job (the workflow only installs the harness).
  - With `SI_AGENT=http`, point it at a staging instance that accepts an `instructions` field, not at the production endpoint.
  - Either way, evaluation traffic is tagged `si_source=eval`, so it never comes back as signal.
- **Give the human queue a real home.** Contradictions, flagged tests, judge-only failures and halts land in `.selfimprove/human_queue.jsonl`. In CI that file is only kept in the Actions cache and uploaded as an artifact, and caches get evicted. Several people share this queue, so point `SI_STATE_DIR` at shared storage and name someone to work it (`selfimprove queue list / accept / edit / reject`).

### 4. The schedule (already in the workflow)

- **Nightly (06:17 UTC):** `score → monitor → align → triage → approve`. This keeps signal flowing, checks the judge against your labels, turns new failures into proposed tests, and has the separate approver merge or flag them. It never writes a candidate.
- **Weekly (Monday 07:43 UTC):** `selfimprove cycle`. It writes a candidate only if there are at least `SI_MIN_TRACES` judged traces and the live failure rate is above `SI_FAIL_THRESHOLD`. It halts and queues a `halt` item if the judge has drifted and re-alignment doesn't fix it. Otherwise it gates the candidate.
- **Every run:** `selfimprove report` goes into the job summary. `concurrency: selfimprove-loop` prevents overlapping runs.
- **Per PR:** turn on `selfimprove-regression.yml` too, so a code change can't quietly break what the dataset protects.

With auto-promote off, a passing gate ends in `awaiting_approval`, and you ship with `selfimprove promote N` as you do today. The difference is that the candidate, the scorecards and `candidate.diff` are already logged on the gate run in MLflow.

---

## Part 2: How to know it's safe to turn on auto-promote

The rule from the skill is: **turn it on only after the loop has run several times without a person overriding the gate.** The harness doesn't compute an override rate for you, so keep a tally while auto-promote is off. Count one override each time you:

- refuse a candidate the gate passed (after reading the diff),
- promote something the gate rejected,
- roll back a version you promoted.

Each gate run carries `si.gate_passed` and `si.gate_reasons` tags. Prompt versions carry `promoted_by`, `status`, `replaced_version` and `rolled_back_because`. Together they are your record.

### Checklist (all should be true)

1. **The gate and you agree, repeatedly.** Several weekly gate decisions in a row with no overrides. At least one should be a rejection you agreed with, which shows the gate actually stops bad candidates. A gate that has only ever said yes hasn't been tested.
2. **The judge is trustworthy, and stays that way.** The nightly `align` keeps judge-vs-human agreement at or above `SI_AGREEMENT_THRESHOLD` (0.75). You can see it in `selfimprove report` under "judge agreement". No recent `halted_untrusted_judge` cycles. Agreement was measured on at least 10 labels with at least 30% in each class. You keep labeling real traces after the switch, because without fresh labels nothing notices the judge drifting.
3. **The holdout means something.**
   - It is big enough. With `SI_HOLDOUT_FRACTION=0.2` and about 20 train rows, the holdout is about 5 cases, so one case is a 20-point swing.
   - Raise `SI_PROMOTION_MARGIN` above its default `0.0`. At 0.0, a candidate can pass by improving on train while staying flat on the holdout. Require a real holdout gain, at least one case's worth.
   - Nobody has looked at the holdout cases. Keep it that way.
4. **Must-pass scorers cover what must never break.** For a support bot that's things like correct policy facts, never promising refunds you don't offer, and escalating when it should. Mark them `must_pass: true` in your criteria file so no average gain can trade them away. Leave `SI_MAX_PROMPT_GROWTH` on so length bloat is rejected automatically.
5. **The baseline numbers are calm** (`selfimprove report`).
   - Few tests a person edited before merging. Many edits mean the criteria or the judge are still wrong: fix the verifier, don't automate further.
   - The human queue isn't growing, and contradictions are resolved.
   - Each theme produces at most a handful of tests.
6. **Rollback works and is fast enough.**
   - **Only gate promotions roll back automatically.** `monitor` acts only on versions tagged `promoted_by=gate`; it never undoes a person's promotion. It compares the new version's live failure rate (on must-pass scorers) with the rate recorded at promotion, and rolls back when the rise exceeds `SI_ROLLBACK_MARGIN` (0.10).
   - **It needs at least `SI_MIN_TRACES` judged traces on the new version.** So nightly `score` must produce must-pass verdicts on production traffic.
   - **It runs only nightly**, so a bad prompt could be live for a day or more. If that's too long for support, add a more frequent `monitor` schedule.
   - **Run a manual `selfimprove rollback` once** before you rely on it, and confirm the bot picks up the change.
7. **Only prompts auto-ship.** Skill, tool and code changes for the bot stay human-accepted proposals. Auto-promote moves the instructions alias and nothing else.

### Turning it on, and what you still own

- **Switch it on:** set the repository variable `SI_AUTO_PROMOTE=true`. Only the weekly `cycle` (and a manually dispatched `gate`) can then promote; the nightly jobs never do. Setting it back to `false` is your off switch.
- **Keep reading the diffs.** After every automatic promotion, read `candidate.diff` on the gate run. Watch for judge-pleasing hedging, text copied from failing cases or judge rationales, facts pasted in where a pointer to the source belongs, instructions about grading, and length growth with no new behavior. The gate catches some of this; reading catches the rest.
- **Keep owning the definition of "good":** the thresholds and margins, the criteria and must-pass list, golden cases and expectations, the human queue, an audit sample of what the approver merged, and a review of every rollback.
- **Turn it back off** after any override, unexplained rollback or drop in judge agreement. Resume once you understand why.

The short version: schedule it now with `SI_AUTO_PROMOTE=false`. Turn it on when you have a run of gate decisions you'd have made yourself, a judge that tracks your labels, a holdout that can actually separate candidates, and a rollback you've watched work.
