# Scheduling the loop and earning auto-promote

Right now you're at **level 2**: the analyst writes candidates and you approve them. Getting to level 4 (scheduled, gate promotes) takes two separate steps, and they should happen at different times:

1. **Schedule it with promotion still off.** CI runs the loop and the gate gives a verdict, but you still move the alias.
2. **Turn on `SI_AUTO_PROMOTE=true`** only once the record from step 1 shows the gate and you agree.

The loop is the same at every level. The only things that change are who writes the candidate and who approves it.

---

## Part 1: Running it nightly in GitHub Actions

The repo already has the workflows at the root, under `.github/workflows/`:

| Workflow | What it does | When |
|---|---|---|
| `selfimprove-loop.yml` | **Nightly** (06:17 UTC): `score`, `monitor`, `align`, `triage`, `approve`. **Weekly** (Mondays 07:43 UTC): the full `cycle`, which adds `improve` and `gate`. Also `workflow_dispatch` for any single job | cron |
| `selfimprove-regression.yml` | runs every dataset row as a test case against the agent and judges | PRs touching `harness/`, or by hand |
| `selfimprove-offline.yml` | the harness's own tests, offline, no keys | every change to `harness/` |

Every job is an MLflow run tagged `si.job`, so you can see what happened overnight in MLflow, not in a CI log. Each run also appends `selfimprove report` to the job summary.

### Setup checklist

**Repository settings**

- [ ] Variable `SELFIMPROVE_ENABLED=true`. The loop and regression jobs do nothing until this is set.
- [ ] Secret `MLFLOW_TRACKING_URI` pointing at your shared server. It needs a **database backend**, and it has to be **the same server production logs to**: the jobs read production traces from it and write the dataset and Prompt Registry back to it.
- [ ] Provider secrets: whichever of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TYPESAFE_API_KEY` your agent, judge and improver use.
- [ ] Variables: `SI_EXPERIMENT` (it defaults to `my-agent`, so set it to your support bot's experiment) and `SI_IMPROVER=analyst` (the default, and what you already use). Leave `SI_AUTO_PROMOTE` unset for now. It defaults to `false`.

**Agent config in the repo**

- [ ] The workflow runs `cp .env.example .env`, so your local `.env` never reaches CI. Either commit your support bot's settings into `harness/.env.example` or set them as repo variables: `SI_AGENT`, its target, `SI_PROMPT_NAME`, `SI_INSTRUCTIONS_FILE`, `SI_CRITERIA_FILE`, `SI_GOLDEN_FILE`, `SI_JUDGE_MODEL`, and the thresholds you've tuned.
- [ ] **The runner has to be able to call your bot.** `score` only reads traces, but `gate` runs the candidate *and* production against the train and holdout rows. With `SI_AGENT=python`, the bot's code and dependencies must install in the checkout. With `SI_AGENT=http`, the URL must be reachable from a GitHub-hosted runner (or use a self-hosted one). Point it at a non-production instance that accepts the `instructions` field.
- [ ] **Production traces are selectable.** The harness picks production traffic by `tags.si_source = 'production'`, which only traces it created itself carry. If your bot logs traces on its own, set `SI_PRODUCTION_FILTER` (for example `tags.environment = 'production'`). Otherwise score, triage and monitor will see nothing.
- [ ] **Your state is on the shared server.** If your hand runs used a local MLflow, the shared server still needs the prompt with its `production` alias, the dataset (with its holdout split) and your labels. Don't run `init` against a server that already has them.

**The human queue**

- [ ] The queue is a file (`.selfimprove/human_queue.jsonl`). The workflow keeps it in the Actions cache and uploads it as an artifact, so an item you `queue accept` on your laptop never gets back to CI. If you'll work the queue locally, or more than one person will, point `SI_STATE_DIR` at shared storage. Then update the cache and artifact paths in the workflow, which are hardcoded to `harness/.selfimprove/...`.

**Nightly vs. weekly**

- [ ] You asked for nightly. By default, only the signal jobs run nightly and `improve` plus `gate` run weekly. I'd keep it that way at first. A cycle only proposes a change when the live failure rate is above `SI_FAIL_THRESHOLD` (0.15) on at least `SI_MIN_TRACES` (8) judged traces, so a nightly cycle on light traffic mostly does nothing, and when it does fire it stacks candidates faster than you can review them. If you do want nightly cycles, change the second cron in `selfimprove-loop.yml`.

### Guardrails `cycle` already applies

- **No signal, no change:** `improve` runs only above `SI_FAIL_THRESHOLD` with at least `SI_MIN_TRACES` judged traces.
- **Verifier health:** if the judge's agreement with people drops below `SI_AGREEMENT_THRESHOLD` (0.75) and re-alignment fails, the cycle halts and queues the problem for a person.
- **Triage and approval:**
  - Triage waits for `SI_TRIAGE_THRESHOLD` (5) new failures.
  - At most `SI_TESTS_PER_THEME_CAP` (3) tests per theme.
  - Contradictions go to the human queue.
  - Judge-only failures never become tests.
- **The gate's rules:**
  - The candidate must beat production on train.
  - Must-pass checks have to hit 100%.
  - The candidate must gain on the hidden holdout by at least `SI_PROMOTION_MARGIN`.
  - The bloat guard rejects anything longer than `SI_MAX_PROMPT_GROWTH` (2.0) times production.
- `concurrency: selfimprove-loop` means two loop jobs never run at once.

### What your week looks like once it's scheduled (still level 2)

1. Monday's cycle writes a candidate, and the gate ends it as `rejected` or `awaiting_approval`.
2. You run `selfimprove diff production latest` and check it for the red flags (below).
3. If you agree, run `selfimprove promote N --reason "..."`. If you don't, leave it and write down why.
4. Work the queue with `queue accept|reject|edit <id> --note "..."`.
5. Read the report in the job summary.

---

## Part 2: How to know it's safe to turn on auto-promote

The rule from the skill: **turn on `SI_AUTO_PROMOTE=true` only after levels 1 to 3 have run several times without a person overriding the gate.** The scheduled level-2 phase above is your trial. Each gate run is a prediction, and your promote/decline decision is the label. Track how often they disagree.

### What counts as an override (each one resets the count)

- The gate passed a candidate and you declined it. Typical reasons: the diff had padding or hedging, it pasted facts in instead of pointing to the source, or it talks to the grader.
- The gate rejected a candidate and you thought it should ship.
- You had to fix a check or the criteria to get a correct answer through. That means the verifier was wrong, even though the fix was the right call.
- You rolled back a promotion because live traffic got worse.

### The readiness checklist

All of these should hold, and you should still be checking them weeks later, not just once:

| Check | Where to see it | Ready when |
|---|---|---|
| Gate and you agree | `si.job = gate` runs (outcome `awaiting_approval` / `rejected`) next to your `promote --reason` records | several consecutive cycles in a row (I'd want at least 4 weekly ones) with zero overrides |
| The judge agrees with people | `judge agreement` in `selfimprove report` | at or above 0.75 and stable across nightly `align` runs, measured on held-out human labels |
| The criteria match reviewers | `selfimprove report`: "accepted as-is" vs. "edited by a person" | edits near zero. Many edits mean the criteria or judge are off: fix the verifier, don't automate |
| The dataset can carry it | dataset summary in the report | at least ~20 train rows plus a holdout big enough that a "gain" isn't one row flipping |
| Critical failures are must-pass | your criteria file | everything a support bot must never do is `must_pass: true`, so an average gain can't buy it. Examples: refund or policy promises it can't keep, leaking account data, skipping the policy lookup (a tool-span check) |
| Rollback can actually fire | live traffic per version | at least `SI_MIN_TRACES` (8) judged production traces on a new version between promotion and the next nightly `monitor`. Below that, monitor can never roll back |
| Rollback works | a dry run | you've run `selfimprove rollback` once on purpose and seen production go back to the previous version |

### One detail that matters

`monitor` **only rolls back automatically on versions the gate promoted** (`promoted_by=gate`, with the failure rate it replaced recorded). Versions you promote by hand get `"not auto-promoted, so no automatic rollback"`. So until you turn auto-promote on, you're the rollback.

Once it's on, `monitor` runs nightly before triage. It rolls back when the live failure rate climbs more than `SI_ROLLBACK_MARGIN` (0.10) above the rate the version replaced, and it reviews every rollback.

### Turning it on

1. Set the repo variable `SI_AUTO_PROMOTE=true`.
2. Keep reading the diff of every auto-promotion after the fact, at least for the first several. The bloat guard catches some judge-pleasing text, and you reading the diff catches the rest.
3. Set up a way to hear about promotions and rollbacks. The workflow only writes the job summary and MLflow runs, so add a notification step or watch the `si.job = gate` / `monitor` runs.
4. If an override happens after you've turned it on, set it back to `false` and fix the verifier before trying again.

---

## What stays with you, even at level 4

- **Thresholds and criteria:** `SI_PROMOTION_MARGIN`, `SI_MAX_PROMPT_GROWTH`, `SI_ROLLBACK_MARGIN` and the criteria file define what counts as good.
- **Ground truth:** golden cases and expectations, plus labels on a sample of traffic (`selfimprove label`), so judge agreement keeps getting measured.
- **The human queue:** contradictions, flagged tests, and judge-only failures that need a label.
- **Audits and rollbacks:** an audit sample of what the automatic `approve` step merged, and a review of every rollback.
- **Anything that isn't the prompt.** Only the prompt ships automatically. A change to the bot's skills, tools or code is a proposal a person accepts.

Two lines not to cross: loosening a threshold or editing a check just so a candidate passes, and letting the improver make that call. Those are your decisions, recorded with who and why: change the setting, re-gate the same version, and note it on that gate run.

### Red flags to look for in diffs (now and after auto-promote)

- New hedging that judges reward ("I believe", "please verify").
- Text copied from failing cases or the judge's rationales.
- Facts pasted in instead of a pointer to the source.
- Instructions about the evaluation itself ("always mention...").
- A big length increase with no new behavior to match.

References: `harness/skills/self-improving-agent/references/workflows.md` (scheduling, auto-promote, rollback) and `references/guardrails.md` (why each rule exists).
