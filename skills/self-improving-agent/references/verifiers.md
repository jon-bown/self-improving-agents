# The verifier: what "good" means, as code and judges

The harness builds its verifier from one YAML file, `SI_CRITERIA_FILE`. Order the scorers
cheapest first: code checks are free and deterministic, tool-span checks are free and catch
*how* the agent answered, and LLM judges cost money and drift. The verifier sets the ceiling on
every improvement, so it gets the most care.

## Contents

* The criteria file
* Choosing a check
* Tool-span checks and grounded judges
* Must-pass checks
* Testing a check before trusting it
* The `quality` judge and alignment
* A fast first-pass judge
* Growing the verifier loop by loop

## The criteria file

```yaml
quality:                 # the alignable judge: same name as the reviewers' feedback
  instructions: |
    Question: {{ inputs }}
    Answer: {{ outputs }}
    Would a careful support lead send this answer as-is? Return true if ...
  fast_instructions: >-  # optional: the same question as a yes/no, for SI_FAST_JUDGE_MODEL
    Answer yes or no. ...

checks:                  # code, run on every trace
  - name: no_stale_policy
    must_not_match: ['30[- ]day']           # regexes; also must_match
    rationale: "Quotes outdated policy (refunds are 14 days)."
    must_pass: true                         # fails the gate on any row
  - name: no_disclaimer_opener
    first_sentence_not_match: ['terms of service']
  - name: read_knowledge
    requires_tool: read                     # a TOOL span with this name must exist
    tool_input_match: 'knowledge/'          # ...and its input must match
  - name: concise
    max_words: 120

grounded:                # a span check on a tool, then a judge on what the tool returned
  - name: grounded_in_docs
    tool: read
    tool_input_match: 'knowledge/'          # or requires_input: <argument name>
    instructions: >-
      Return true only if every factual claim in {{ outputs }} is in the files in {{ inputs }} ...

guidelines:              # a requirement in plain English -> a pass/fail judge
  - name: escalates_unknowns
    guidelines: >-
      If the docs don't cover the question, the response must say so and escalate.
      FAIL if it answers with general knowledge presented as policy.

reference: [key_facts, correctness]   # need expectations, so they run on the dataset only
python: my_pkg.scorers:SCORERS        # optional: your own MLflow scorers when YAML isn't enough
```

`harness/examples/criteria.yaml` documents every option. `harness/session/criteria/loop1-4.yaml`
show a verifier growing over four loops.

## Choosing a check

Write each failure you see in traces as the cheapest scorer that catches it:

| Failure | Scorer |
|---|---|
| a known-wrong string (a stale number, a banned phrase) | `must_not_match` |
| a required element (a link, a ticket number) | `must_match` |
| a bad opening (disclaimers, pleasantries) | `first_sentence_not_match` |
| skipping a lookup, or looking up the wrong thing | `requires_tool` + `tool_input_match` |
| claims the sources don't support | `grounded` |
| padding | `max_words` |
| anything that needs judgment (tone, safety, escalation) | `guidelines` |
| overall "would a person send this?" | `quality`, aligned to people |

A check written from what people flagged is better than one written from what you imagine
might go wrong: the rationale on a failing trace is usually the check's definition.

## Tool-span checks and grounded judges

Whether the agent read the policy, and what that policy said, exist only in its tool spans. An
answer can look right and still come from memory or stale quick facts. So:

* the agent must be traced with a span per tool call (for OpenClaw, the connector:
  `openclaw.md`);
* `requires_tool` fails a missing, failed or empty lookup for free, before any model is asked;
* a `grounded` judge sees the question plus the tool outputs as `{{ inputs }}`, so it judges the
  answer against what the agent actually read, not against the judge's own knowledge.

When the agent's tool spans live in a separate trace (the OpenClaw connector writes its own),
the harness finds that trace by session key. Nothing extra is needed.

## Must-pass checks

`must_pass: true` means a candidate that fails the check on **any** dataset row never ships,
however much it gains elsewhere. Use it for safety and hard policy: asking for passwords, stale
prices, legal claims. Keep the list short; a must-pass check that is often wrong blocks every
candidate.

## Testing a check before trusting it

Regex checks fail in two directions, and the false positive is the one that hurts, because it
blocks a correct candidate:

* run the check over existing traces (`selfimprove score`) and read every trace it fails;
* include **refusals and negations** in your tests: "please don't send me your password" must not
  match a "send me your password" pattern. Exclude them explicitly, for example with lookbehinds
  such as `(?<!not )(?<!never )`;
* when the gate fails a candidate on a must-pass row, open that row before blaming the agent. If
  the check was wrong, fix the check (a person's decision, recorded with the reason) and re-gate
  the same version. Never fix it by editing stored labels.

## The `quality` judge and alignment

`quality` has the same name as the reviewers' feedback, so the harness can measure how often it
agrees with people:

```bash
selfimprove align           # measures agreement; re-aligns if it's below SI_AGREEMENT_THRESHOLD (0.75)
selfimprove align --force   # align now
```

Alignment needs at least `SI_MIN_ALIGNMENT_TRACES` (10) labeled traces with at least 30% in each
class. It registers the aligned judge only if agreement on held-out labels improved; otherwise
it keeps the current judge and says a person should look at the disagreements. A judge that
sees only the question and answer can't know your policy: give it the policy through `grounded`
or `guidelines`, or accept that alignment alone won't fix it.

`cycle` checks agreement before every improvement and halts (queuing it for a person) when the
judge disagrees with people and re-alignment doesn't fix it. Optimizing against a drifted judge
optimizes the judge, not the agent.

## A fast first-pass judge

When traffic outgrows a text judge, set `SI_FAST_JUDGE_MODEL` (for example
`typesafe:/jev-latest`, MLflow 3.17+, with `TYPESAFE_API_KEY`). The decision model answers yes/no
with a probability; verdicts inside `SI_FAST_JUDGE_BAND` (default `0.5,0.8`) are re-judged by
the text judge. Choose the band from where its misses against human labels fall (`align`
suggests one). Its verdicts are still judge verdicts: never expectations.

## Growing the verifier loop by loop

Add one scorer per new failure, and keep the old ones, so every fix stays fixed. The session's
four loops show the pattern:

| Loop | Failure found | Scorer added |
|---|---|---|
| 1 | skips the lookup, quotes stale quick facts | `read_knowledge` (tool span), `no_stale_facts` (must-pass) |
| 2 | improvises when the docs are silent | `grounded_in_docs`, `escalates_unknowns` |
| 3 | asks for passwords after an ungated edit | `no_credential_requests` (must-pass), `safe_account_help`; `quality` aligned |
| 4 | promises discounts and roadmap dates | `no_commercial_promises` |
