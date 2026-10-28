# v7 rejected for bloat, v9 rejected on must-pass: what to do next

**Short answer:** don't raise the limit just yet. Raising it is allowed, but only after you've read v7's diff and confirmed the extra length is real behavior. Before any of that, deal with production: as you describe it, the live version asks customers for 2FA codes. That's the urgent problem, and neither candidate has to pass for you to fix it.

I haven't looked at your MLflow store, so the steps below tell you what to check rather than assuming the answers.

## 0. First, stop the leak in production (today)

The version your support team shipped is live, and it asks for credentials. Put production back on the last version from before their edit. That version passed your gate, and moving production to it is your call as a person:

```bash
selfimprove diff <pre-edit-version> production    # confirm which version is the clean one
selfimprove promote <pre-edit-version> --reason "revert ungated AGENTS.md edit: agent asked customers for 2FA codes"
```

Use `promote <N>` with an explicit version instead of `rollback`. `rollback` restores whatever the current version's `replaced_version` tag or the `previous` alias points at. A version shipped outside the loop may not have those set to the version you expect.

You lose whatever useful content the support team added, but only until a candidate that keeps it passes the gate. If the edit isn't already in the registry as its own version, register it with its author and commit message so the history shows where the regression came from.

## 1. v7 (MetaPrompt): bloat at 2.2x

The bloat guard (`SI_MAX_PROMPT_GROWTH`, default 2.0) asks whether the extra length is new behavior or padding. A security section added to a short prompt is the textbook case where a person may legitimately raise the limit. Check three things before you do:

1. **Is bloat the only reason it failed?** The gate records every reason, not just the first. Look at the `si.gate_reasons` tag on v7's gate run, or `gate_table.txt` in that run. If a must-pass check, the holdout margin or "worse on train" also failed, raising the limit won't help.
2. **Read the diff:** `selfimprove diff production 7`. You're looking for:
   - Does it **remove** the support team's "ask for the code" instruction, or does it keep that and add a contradicting "never ask for codes" section on top? Layered contradictions are a common reason a candidate gets long, and they're fragile.
   - Hedging and caveats that please the judge ("please verify", "I believe"). Text copied from failing cases or judge rationales. Policy pasted in when a pointer to `knowledge/` would do. Any text about how answers are graded. If any of these turn up, it's padding: reject v7 and leave the limit alone.
3. **Could a short version do the same job?** A rule that support never asks for passwords, 2FA, backup or one-time codes and never changes an account from chat, plus a pointer to the policy file, takes a few lines. Usually the better fix is to write that yourself as a level-1 candidate and gate it. That candidate fits under 2.0, so no guardrail changes:

   ```bash
   selfimprove improve --from-file AGENTS.trimmed.md
   selfimprove diff production latest
   selfimprove gate
   ```

**If the diff is real content and you still want v7 as written,** raise the limit for this agent only, as a recorded human decision:

- Set `SI_MAX_PROMPT_GROWTH` in this agent's `.env` to the smallest value that fits, e.g. `2.5`. Don't remove the guard and don't pick a huge number.
- Re-gate **the same version**: `selfimprove gate 7`. Don't re-run the improver.
- Record who decided and why on that gate run, e.g. "raised to 2.5: v7 adds a real account-security section to a short prompt; reviewed diff, no padding". Then decide whether the new limit stays or goes back to 2.0 for future candidates.

Note: the gate measures growth and gain against **current production**. Once you've reverted production (step 0), v7 is compared with the clean version. Its length ratio changes, and it has to beat that version on the holdout by `SI_PROMOTION_MARGIN`. If it can't, it isn't an improvement over simply reverting, and that's a correct rejection.

## 2. v9 (GEPA): must-pass `no_credential_requests` at 92%

A must-pass check has to be at 100% on every row. 92% usually means one row failed. **Open that row before you blame either side.** The gate run holds the candidate's scorecard and per-row rationales.

- **The agent really asked for a code or offered to disable 2FA:** v9 is out. Don't promote it. Run the improver again or use the trimmed candidate above.
- **The check flagged a correct refusal:** the regex has lookbehinds for "not ", "never " and "n't ", but phrasings like "please don't **ever** send us your 2FA code" or "to verify your identity, enter the 6-digit code in the app" can still match. In that case the check is wrong, not the agent. Fix the pattern in the criteria file. Run `selfimprove score` over existing traces and read every trace it fails, so you know you haven't opened a real hole. Then re-gate the **same** version with `selfimprove gate 9`, and record who changed the check and why.

What not to do: drop `must_pass`, loosen the check so a candidate gets through, or edit stored labels. All three hide the exact failure you shipped last week.

## 3. Who decides

The limit change and any change to the check are guardrail decisions, so they're yours (or your team lead's), recorded with a reason. Don't hand them to MetaPrompt, GEPA, or a coding agent acting as the improver. Whoever wrote a candidate doesn't get to loosen the gate it failed.

## Suggested order

1. Revert production to the pre-edit version (`promote <N> --reason ...`).
2. Open v9's failing row: check wrong → fix it, record why, `gate 9`. Agent wrong → drop v9.
3. Read v7's diff and its full gate reasons: padding or layered contradictions → write a short candidate yourself. Real content → raise the limit for this agent with a reason, then `gate 7`.
4. Promote whichever candidate passes (`gate N --promote`, or `promote N --reason ...`), and keep `monitor` running on the new version.
