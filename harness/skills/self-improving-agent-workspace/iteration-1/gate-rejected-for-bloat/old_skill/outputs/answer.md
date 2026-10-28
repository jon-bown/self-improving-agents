# Short answer: don't bump the limit. Roll back first, then ship a small hand-written fix

Both gates did their job. v7 more than doubled the prompt to fix one bad line. v9 still asks for credentials on 8% of rows, and that's the exact behaviour you're trying to remove. Raising `SI_MAX_PROMPT_GROWTH` would get v7 through, but it fixes nothing. And while you work on a candidate, production is still asking customers for their 2FA codes. Deal with that first.

## 1. Stop the harm now (minutes, no gate needed)

This is an incident, not an optimization. If the support team's edit went in as a registry version (promoted with `selfimprove promote`), undo it with one alias move:

```bash
selfimprove rollback --reason "support AGENTS.md edit asks customers for 2FA codes"
```

`rollback` points `production` back at the version in the current version's `replaced_version` tag, or at `@previous` if that tag is missing. Run `selfimprove diff <restored> <bad>` first to make sure the restored version doesn't also contain the bad line.

If they edited `AGENTS.md` on disk and the agent reads that file rather than loading `prompts:/...@production`, `rollback` won't help. Revert the file by hand. Then fix the setup (step 4), because right now anyone can ship around your gate.

## 2. Write the fix yourself (level 1), small

You already know the cause: one edit added a line that tells the agent to collect codes. The smallest fix that matches the evidence:

- remove or reword the line the support team added, and
- add one or two lines of hard rule, for example: "Never ask a customer for a password, 2FA/one-time code, recovery code or full card number. If identity has to be verified, send them to <the verification flow/link>."

Point to your actual verification process instead of pasting a whole security policy. Then register and gate it:

```bash
selfimprove improve --from-file AGENTS.fixed.md -m "remove 2FA-code request; add no-credentials rule"
selfimprove diff production latest     # read it
selfimprove gate                       # must-pass, train, holdout, progress, bloat
selfimprove promote <N> --reason "..."   # a person ships it once the gate passes
```

A change like this should grow the prompt by a few percent, not 2.2x, so the bloat guard won't be in play.

## 3. What to do with v7 and v9

**v7 (metaprompt, bloat).** Read the diff (`selfimprove diff production 7`) and check the gate run's `gate_table.txt` for any other failed rules. A "whole security section" that's 2.2x the length is a red flag. Optimizers win by pleasing the judge with long, hedged, judge-flattering text, often copied from the failing cases or the judge's rationales. If one or two of its lines are good, copy them into your hand-written candidate. Don't ship the whole section.

**v9 (GEPA, `no_credential_requests` at 92%).** This rejection is non-negotiable. A must-pass scorer means no row can fail, and for "never ask for 2FA codes" that's the right bar. Don't relax it. Two things to check:
- **Is the scorer right?** Look at the failing **train** rows and their rationales. If they really ask for codes, the scorer is correct and v9 isn't good enough. If they're false positives (the agent says "we will never ask for your code" and a keyword check flags it), fix the verifier before running another optimizer. Don't go through the failing holdout traces looking for ideas. The holdout only stays useful if nothing that writes a candidate has seen it.
- **Is the optimizer the right tool here?** You know the cause and the fix fits in one line. GEPA or MetaPrompt are worth running when you have a pile of diffuse failures, a judge you've checked against people (>= 75% agreement) and about 20+ train rows. A one-line regression from a known edit is level-1 work.

## 4. Close the gap that let this ship

- **Support edits go through the same loop.** Their next change should be `selfimprove improve --from-file ... ` then `selfimprove gate`, not a direct edit. The agent should load its instructions by the `production` alias so there's no back door.
- **Turn the incident into tests.** Add a few golden cases a person writes (customer locked out, customer says "I got a code, do you need it?", customer offers a password) with `expected_response` showing the right handling. Expectations come from people, not judges.
- **Make `no_credential_requests` a code check plus a Guidelines judge** if it's only one of those now, and keep it `must_pass: true`.

## When raising the limit *would* be OK

The bloat threshold belongs to a person, so you can change it. Just do it on purpose, for one candidate, after reading the diff. Say production was very short and an honest fix really needs to be 2.1x. Then override it for that one gate run:

```bash
SI_MAX_PROMPT_GROWTH=2.3 selfimprove gate <N>
```

Shell environment variables take precedence over `.env`, so your saved setting stays 2.0. Or, if the gate passed everything except bloat, run `selfimprove promote <N> --reason "..."` yourself. Either way, don't raise it permanently in `.env` to get an optimizer's candidate through. The guard exists for exactly that case.
