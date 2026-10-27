"""Write session/notebooks/0N_*.ipynb from this file; edit here, not in the notebooks.

    cd harness && .venv/bin/python session/build_notebooks.py [output folder]

Cells marked "Under the hood" say which MLflow calls produced each step's evidence, and read it
back with the raw MLflow API. They only read from the store, like every other cell.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def under(s):
    return md("**Under the hood:** " + s.strip("\n"))


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "selfimprove (.venv)", "language": "python", "name": "selfimprove"}
    nb.metadata["language_info"] = {"name": "python"}
    for i, cell in enumerate(cells):
        cell.id = f"{name[:2]}-{i:02d}"  # stable ids, so a rebuild only diffs what changed
    nb.cells = cells
    nbf.write(nb, OUT / name)


LADDER = """
| Loop | Failure | Signal | Store | Update | Promotes |
|---|---|---|---|---|---|
| 1 | Skips the lookup | you | you | you | you |
| 2 | Makes things up | you + a grounding judge | triage + approve | the analyst | you |
| 3 | Unsafe account help, from another team's edit | you + an aligned judge, must-pass check | triage + approve | an optimizer | you |
| 4 | Bad escalation, from another team's edit | a 50% label sample + scheduled scoring | `cycle` | `cycle` | the gate |
"""

SETUP = """
import _session as s

experiment_id = s.setup({n})
r = s.results({n})
"""

# --------------------------------------------------------------------------------------------
notebook("01_manual.ipynb", [
    md("""
# Loop 1 · Manual: the agent skips the lookup

The Acme Cloud support agent runs on **OpenClaw**. Its behavior is one file, `AGENTS.md`, which the
harness keeps in the MLflow Prompt Registry and renders into OpenClaw's workspace for every call.
The policies it should answer from sit next to it in `knowledge/`. The **MLflow connector**
(`connectors/openclaw-mlflow`) logs every OpenClaw run as its own trace, with an `LLM` span for
its model calls and a `TOOL` span per tool call.

Four loops, each finding one failure and fixing it with measurable evidence. v1 ships with the
first two; the other two arrive later, when other teams edit AGENTS.md and ship it without the
gate. Each loop hands more of the work to the harness:
""" + LADDER + """
In this loop a person does every step. All results come from `session/store/` (the precomputed
run); `RUNBOOK.md` has the same steps as `selfimprove` commands. Cells marked **Under the hood**
say which MLflow calls produced each step's evidence, and read it back with the raw MLflow API.
"""),
    code(SETUP.format(n=1)),
    md("""
## 1. Find: what v1 tells the agent

Look at the first bullet and the quick facts. Then compare them with `knowledge/`: refunds are
14 days, reset links last 15 minutes, lockouts 30 minutes, and Pro has 10 seats.
"""),
    code("""
from selfimprove import instructions

print(instructions.load(1).template)
"""),
    under("""
`instructions.load(1)` is `mlflow.genai.load_prompt("prompts:/acme-agents-md/1")`. The Prompt
Registry keeps every version of AGENTS.md with a commit message and tags, and the `production`
alias says which one the agent runs. Before each call the harness loads the version under test and
writes it into a fresh OpenClaw workspace as `AGENTS.md`. Shipping a change means moving that
alias; nothing is redeployed. At the end of the precomputed run, `production` points at:
"""),
    code("""
import mlflow
from selfimprove import config

name = config.settings.prompt_name
p = mlflow.genai.load_prompt(f"prompts:/{name}@production")
print(f"{p.name} @production -> v{p.version}: {p.commit_message}")
"""),
    md("""
## 2. Signal: production traffic, as the reviewers saw it

Eight customer questions went to production on v1. Each one is a trace. The reviewers (you, in
the room) labeled each answer pass or fail and attached the correct answer to every failure.
"""),
    code("""
before = s.loop_traces(1, "before")
s.show(s.answers(before))
"""),
    under("""
each question went through `agents.run()`, which opens an `AGENT` span with `mlflow.start_span(...)`
and tags the trace with `mlflow.update_current_trace(tags={"si_source": "production",
"si_prompt_version": ...})`. Every later step finds real traffic by that tag
(`tags.si_source = 'production'`) and keeps evaluation runs out of it (`si_source = 'eval'`). The
session also tagged each trace with its loop and phase, which is what `s.loop_traces` filters on:
"""),
    code("""
t = before[0]
print(t.info.trace_id, t.info.state)
{k: v for k, v in t.info.tags.items() if not k.startswith("mlflow.")}
"""),
    md("""
Why did it fail? The harness trace only says "OpenClaw answered". The connector's trace for the
same run shows what OpenClaw actually did: one model call and **no `read`**. The harness links
the two with the run's session key (`openclaw.session_key` → `mlflow.trace.session`).
"""),
    code("""
s.tree(before[0])
"""),
    under("""
the harness span records `openclaw.session_key`, and the connector writes the same key into its own
trace's metadata as `mlflow.trace.session`. One `mlflow.search_traces` call on that metadata finds
OpenClaw's trace for this run:
"""),
    code("""
key = next(sp.attributes["openclaw.session_key"] for sp in before[0].data.spans
           if "openclaw.session_key" in sp.attributes)
oc = mlflow.search_traces(locations=[experiment_id], return_type="list",
                          filter_string=f"metadata.`mlflow.trace.session` = '{key}'")[0]
print(key)
[(sp.name, sp.span_type) for sp in oc.data.spans]
"""),
    md("""
That observation is cheap to turn into code. Loop 1's verifier (`session/criteria/loop1.yaml`) adds
two checks:

* `read_knowledge`: the connector's trace must have a successful `read` of a `knowledge/` file
* `no_stale_facts`: **must-pass**, no quoted stale fact (30-day, 24 hours, 25 seats, 1 hour)

`selfimprove score` ran them, and the `quality` judge, on every production trace:
"""),
    code("""
s.show(s.verdicts(before))
"""),
    code("""
s.show(s.failure_rate(before).to_frame("failure rate"), pct=True)
"""),
    under("""
every verdict is an **assessment** attached to the trace, with a source. A reviewer's label is
`mlflow.log_feedback(name="quality", ..., source=HUMAN)`, plus `mlflow.log_expectation(name=
"expected_response")` for the correct answer. `selfimprove score` ran `mlflow.genai.evaluate(data=
<these traces>, scorers=[...])` with **no** `predict_fn`, so nothing re-ran: the checks and judges
read the recorded trace and attached `CODE` and `LLM_JUDGE` feedback next to the human label.
"""),
    code("""
for a in before[0].info.assessments:
    print(f"{a.name:<22} {str(a.value)[:14]:<14} {a.source.source_type:<12} {a.source.source_id}")
"""),
    md("""
## 3. Store: the failures become tests

Each labeled failure becomes a dataset row: the question as input, the reviewer's correct answer
as the expectation, and its provenance as tags. A hash of the question puts about one in five in
the **holdout**, which only the gate ever sees. The eight golden holdout cases were there from
the start (`session/golden.yaml`).
"""),
    code("""
from selfimprove import store

print(r["dataset"])
df = store.dataset().to_df()
df = df[df["tags"].map(lambda t: t.get("source") == "golden" or t.get("author") == "you")]  # as of loop 1
s.show(df.assign(question=df["inputs"].map(lambda i: i["question"][:70]),
                 split=df["tags"].map(lambda t: t.get("split")),
                 source=df["tags"].map(lambda t: t.get("source")),
                 author=df["tags"].map(lambda t: t.get("author")))[["question", "split", "source", "author"]])
"""),
    under("""
the dataset is an MLflow evaluation dataset (`mlflow.genai.datasets.create_dataset`), and new tests
arrive with `dataset.merge_records([...])`. Each record is `{inputs, expectations, tags}`; the tags
carry the provenance (`source`, `author`, `source_trace`) and the `split`. The gate and the
regression suite read it back with `get_dataset`:
"""),
    code("""
from mlflow.genai.datasets import get_dataset

ds = get_dataset(name=config.settings.dataset_name)
recs = ds.to_df()
rec = recs[recs["tags"].map(lambda t: t.get("source") == "human_fail")].iloc[0]   # a loop-1 failure
print(ds.name)
{"inputs": rec["inputs"], "expectations": rec["expectations"], "tags": rec["tags"]}
"""),
    md("""
## 4. Update: a person writes v2

The fix names the four policy files (OpenClaw's `read` can't list a folder) and deletes the
stale quick facts. Nothing else changes: the other three flaws are next loops' work. It is
registered as a new version; `production` still points at v1.
"""),
    code("""
print(instructions.diff(r["production_before"], r["candidate"]))
"""),
    under("""
the candidate is `mlflow.genai.register_prompt(...)` with the tags `author` and `status=candidate`.
Registering never moves an alias, so production keeps running v1 until someone promotes. The tags
below are as of the end of the run; the gate added its verdict and run id:
"""),
    code("""
c = mlflow.genai.load_prompt(f"prompts:/{name}/{r['candidate']}")
print(f"v{c.version}: {c.commit_message}")
{k: v for k, v in c.tags.items() if k in ("author", "gate", "gate_run")}
"""),
    md("""
## 5. Gate

Production and the candidate answer every dataset row fresh, scored by the same verifier. The
candidate ships only if every must-pass scorer is at 100%, it is not worse on train, it is not
worse on the holdout, and it is better somewhere.
"""),
    code("""
print(r["gate"]["outcome"], "-", r["gate"]["message"])
s.show(s.gate_table(1), pct=True)
"""),
    under("""
the gate is one MLflow run with four nested runs: train and holdout, for production and for the
candidate. Each nested run is `mlflow.genai.evaluate(data=<dataset rows>, predict_fn=<the agent
pinned to one version>, scorers=[...])`, so both versions answer every row fresh, and those traces
are tagged `si_source = 'eval'`. The rules themselves are harness code (`selfimprove/gate.py`). The
parent run keeps the decision: a metric per scorer, `prompt_growth`, the tags `si.gate_passed` and
`si.gate_reasons`, and the artifacts `gate_table.txt` and `candidate.diff`:
"""),
    code("""
gate_run = mlflow.search_runs(
    experiment_ids=[experiment_id], output_format="list", order_by=["start_time DESC"],
    filter_string=f"tags.`si.job` = 'gate' AND tags.`si.candidate_version` = '{r['candidate']}'")[0]
print(gate_run.info.run_name, {k: v for k, v in gate_run.data.tags.items() if k.startswith("si.")})
children = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list",
                              filter_string=f"tags.mlflow.parentRunId = '{gate_run.info.run_id}'")
print(sorted(ch.info.run_name for ch in children))
print(mlflow.artifacts.load_text(f"runs:/{gate_run.info.run_id}/gate_table.txt"))
"""),
    md("""
## 6. Measure: the same traffic, after the promotion

A person promoted v2. The same eight questions went to production again and were reviewed again:
"""),
    code("""
s.show(s.before_after(1).astype({"failure_rate": float}), pct=False)
"""),
    code("""
after = s.loop_traces(1, "after")
s.show(s.answers(after))
"""),
    code("""
s.tree(after[0])
"""),
    under("""
`selfimprove promote 2` is two alias moves, `mlflow.genai.set_prompt_alias(name, "previous", 1)`
and `set_prompt_alias(name, "production", 2)`, plus version tags that record who promoted, why, and
what it replaced. A rollback moves `production` back. The agent loads the alias on every call, so
neither needs a redeploy:
"""),
    code("""
v2 = mlflow.genai.load_prompt(f"prompts:/{name}/{r['candidate']}")
{k: v for k, v in v2.tags.items() if k in ("promoted_by", "promoted_because", "replaced_version", "replaced_failure_rate")}
"""),
    md("""
**What a person did in this loop:** everything. Read traces, labeled them, turned failures into
tests, wrote the fix, read the gate's table, promoted. The next loop hands the store and update
steps to the harness. Open the MLflow UI (`Traces`, `Prompts`, `Datasets`, and the `gate` run) to
see the same evidence there.
"""),
])

# --------------------------------------------------------------------------------------------
notebook("02_assisted.ipynb", [
    md("""
# Loop 2 · Assisted: the agent improvises when the docs are silent

AGENTS.md still says *"never say you don't know … answer with the specifics a SaaS company like
Acme would have"*. On v1 that produced invented facts. v2 reads the policy files first, so outright
inventions are now rare, but when the files don't cover a question (rate limits, certifications,
regions, payment methods) it still improvises: it hedges, sends the customer elsewhere, or pads the
answer with unrelated policy facts, instead of escalating to a specialist as
`knowledge/support_process.md` requires.
""" + LADDER + """
New in this loop: a **grounding judge** that reads the connector's `read` spans, and the harness
doing the **store** (triage + approve) and **update** (the analyst) steps. A person still
reviews the proposal and promotes.
"""),
    code(SETUP.format(n=2)),
    md("## 1. Find"),
    code("""
before = s.loop_traces(2, "before")
s.show(s.answers(before))
"""),
    md("""
## 2. Signal: is every claim backed by what the agent read?

`grounded_in_docs` (in `criteria/loop2.yaml`) is a span check plus a judge. It finds the `read`
spans in the connector's trace, fails the answer for free if there are none, and otherwise asks
the judge whether every claim is in the files the agent read. The judge sees the actual file
contents, from the spans, not a description of them.
"""),
    code("""
s.tree(before[0])
print()
print(s.rationale(before[0], "grounded_in_docs"))
"""),
    under("""
`grounded_in_docs` is a harness scorer (`@scorer`) in `selfimprove/signals.py`. It reads the `read`
spans; if there are none, or a read failed, it returns a failing verdict without asking a model.
Otherwise it calls a `make_judge(...)` judge whose `{{ inputs }}` are the question plus the file
contents those spans returned. The verdict lands on the trace as feedback, like every other:
"""),
    code("""
import mlflow
from selfimprove import config

name = config.settings.prompt_name
[(a.name, a.value, a.source.source_type) for a in before[0].info.assessments if a.name == "grounded_in_docs"]
"""),
    code("""
s.show(s.verdicts(before))
"""),
    code("""
s.show(s.failure_rate(before).to_frame("failure rate"), pct=True)
"""),
    md("""
## 3. Store: triage and approve

**Triage** reads the failures since its last run, sends them to an LLM analyst that groups them
into themes, and proposes one test per failure (capped per theme). Holdout questions are hidden
from the analyst. **Approve** is a separate step with its own checks: does the failure
reproduce, is it a near duplicate, does a reviewer agent sign off? It merges what passes and
queues the rest for a person.
"""),
    code("""
print(r["triage"]["message"])
print(r["approve"]["message"])
print(r["dataset"])
"""),
    code("""
import mlflow

# this loop's triage run: its id prefix is in the triage message ("proposal on run 1a2b3c4d")
prefix = r["triage"]["message"].rsplit("run ", 1)[1].rstrip(")")
run = next(x for x in mlflow.search_runs(filter_string="tags.`si.job` = 'triage'", output_format="list")
           if x.info.run_id.startswith(prefix))
analysis = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/analysis.json")
for th in analysis.get("themes", []):
    print(f"{th['slug']}: {th.get('criterion', '')}")
"""),
    under("""
triage and approve hand work to each other through MLflow runs. Triage is a run tagged
`si.job = triage`; it logs `analysis.json` and `proposal.json` with `mlflow.log_dict` and tags itself
`si.proposal = pending`. Approve finds pending proposals with `mlflow.search_runs`, reads them with
`mlflow.artifacts.load_dict`, merges what passes into the dataset (`merge_records`), and tags the
triage run `si.proposal = done`. The human queue is the one piece outside MLflow: a file in the
harness's state folder.
"""),
    code("""
{k: v for k, v in run.data.tags.items() if k.startswith("si.")}
"""),
    md("""
The approver flagged the proposed tests instead of merging them: its reviewer agent couldn't tell,
from the test alone, that the documentation really doesn't cover these questions. That is the
**human queue**: everything the automation refuses to decide waits there for a person.
"""),
    code("""
s.show(s.pd.DataFrame([{"kind": q["kind"], "status": q["status"], "question": (q.get("record") or {}).get("inputs", {}).get("question", "")[:60],
                        "why": q["reason"][:140], "note": q.get("note", "")}
                       for q in r["queue"] if q["kind"] == "flagged_test"]))
"""),
    md("""
## 4-5. Update: the analyst writes a candidate, the gate decides

The analyst reads the failure report and the current AGENTS.md and proposes a new version with a
change summary. It never promotes. If the gate rejects the candidate, a person reviews the human
queue (adding the flagged tests to the dataset) and the analyst gets one more try.
"""),
    code("""
for a in r["attempts"]:
    g = a.get("gate", {})
    print(f"v{a.get('candidate', '-')}  gate {g.get('outcome', '-'):<18} {g.get('reasons', '')[:110]}")
    print(f"     {a['message'][:160]}")
print()
print("queue review:", r["queue_review"])
"""),
    under("""
the analyst is one LLM call, traced as a `trace_analyst` span, that reads the failure report and
returns a new AGENTS.md. Each attempt is registered with `register_prompt` (tags `author`,
`status=candidate`). The gate then writes `gate=passed` or `gate=failed`, and its run id, back onto
that version, so a rejected version keeps the evidence for not shipping it:
"""),
    code("""
for a in r["attempts"]:
    if a.get("candidate"):
        pv = mlflow.genai.load_prompt(f"prompts:/{name}/{a['candidate']}")
        print(f"v{pv.version}", {k: pv.tags.get(k) for k in ("author", "gate", "gate_run")})
"""),
    md("""
Each rejected candidate is still a version in the Prompt Registry, with its gate run, so the
evidence for *not* shipping it is kept too. The first attempt's table:
"""),
    code("""
first = r["attempts"][0]["gate"]
order = ["train/baseline", "train/candidate", "holdout/baseline", "holdout/candidate"]
s.show(s.pd.DataFrame({k: first["cards"][k] for k in order if k in first["cards"]}).sort_index(), pct=True)
"""),
    md("The candidate that shipped:"),
    code("""
from selfimprove import instructions

print(instructions.diff(r["production_before"], r["candidate"]))
"""),
    code("""
print(r["gate"]["outcome"], "-", r["gate"]["message"])
s.show(s.gate_table(2), pct=True)
"""),
    md("""
Read the table, not only the verdict: the gate's rules are about the whole suite, so a scorer can
dip on a row while the candidate passes overall. A person promoting it should open those rows
(the `train/candidate` evaluation run in the UI) before pressing the button. In loop 4 nobody
will, which is why the must-pass rules matter there.
"""),
    md("## 6. Measure: the same traffic after the promotion"),
    code("""
s.show(s.before_after(2))
"""),
    code("""
s.show(s.answers(s.loop_traces(2, "after")))
"""),
    md("""
**What moved from the person to the harness:** grouping failures, writing tests, checking them,
and drafting the fix. The person still labels traffic, reviews the analyst's diff, and
promotes.
"""),
])

# --------------------------------------------------------------------------------------------
notebook("03_optimizer.ipynb", [
    md("""
# Loop 3 · Optimizer: unsafe account help

After loop 2, the support team wanted faster account recovery. They added a section to AGENTS.md
telling the agent to "verify" customers by asking for their password or a 2FA code, then say it
has reset 2FA, and shipped it straight to production, without the gate.

`knowledge/account_access.md` only describes the self-service paths (backup codes, waiting out a
lockout, identity verification through support). The rule that support never asks for a
credential or changes an account from chat lives with the support leads, in their labels. This
loop moves it from their labels into AGENTS.md, where an edit like this one can't silently undo it.
""" + LADDER + """
New in this loop: a **must-pass** safety check, **judge alignment** (the generic `quality` judge
doesn't know Acme's account policy, so it is aligned to the reviewers' labels), and an
**optimizer** (MetaPrompt) writing the candidate. A person still promotes.
"""),
    code(SETUP.format(n=3)),
    md("## 0. The change nobody gated"),
    code("""
from selfimprove import instructions

c = r["outside_change"]
print(f"v{c['version']} by {c['by']}: {c['message']}\\n")
print(instructions.diff(c["base_version"], c["version"]))
"""),
    md("## 1. Find"),
    code("""
before = s.loop_traces(3, "before")
s.show(s.answers(before))
"""),
    md("""
## 2. Signal: a must-pass check, and a judge that agrees with people

`no_credential_requests` is a regex check marked `must_pass: true`: a candidate that asks for a
credential or claims to turn off 2FA on **any** row can never ship, whatever else it improves.
`safe_account_help` is the same idea as a Guidelines judge, for phrasings no regex catches.
"""),
    code("""
s.show(s.verdicts(before))
"""),
    md("""
The `quality` judge only sees the question and the answer. To it, "send me your 2FA code and I'll
reset it" can look helpful. **Align** measures its agreement with the reviewers on a held-out
slice of labeled traces and, if it is below the threshold, aligns it to their labels and
registers the aligned judge only if agreement improved.
"""),
    code("""
print(r["align"]["message"])
r["align"]["metrics"]
"""),
    under("""
alignment is MLflow's `judge.align(traces, optimizer)`. With no embedding model configured, the
harness uses `SIMBAAlignmentOptimizer`, and `MemAlignOptimizer` otherwise; either tunes the judge's
instructions toward the reviewers' labels. Agreement is measured on held-out labeled traces before
and after, and kept on the experiment as the tag `si.judge_agreement`, which triage checks before it
trusts the judge. At the end of the run:
"""),
    code("""
import mlflow
from selfimprove import config

name = config.settings.prompt_name
{k: v for k, v in mlflow.get_experiment(experiment_id).tags.items() if k.startswith("si.")}
"""),
    md("## 3. Store: triage and approve"),
    code("""
print(r["triage"]["message"])
print(r["approve"]["message"])
print(r["dataset"])
"""),
    md("""
A person cleared the human queue before the optimizer ran, so its tests are in the train split:
"""),
    code("""
print(r["queue_review"])
"""),
    md("""
## 4-5. Update with an optimizer, then gate

`optimize_prompts` rewrites AGENTS.md against every scorer on the train split (each evaluation is
a set of real OpenClaw runs). The holdout is never part of its training data. MetaPrompt goes
first; if the gate rejects its candidate, GEPA gets a turn. Every attempt is shown, including
the ones the gate stopped.
"""),
    code("""
for a in r["attempts"]:
    g = a.get("gate", {})
    print(f"{a['improver']:<11} {a['outcome']:<10} {a['message'][:90]}")
    print(f"{'':<11} gate: {g.get('outcome', '-')}  {g.get('reasons', '')[:120]}")
"""),
    under("""
each optimizer is `mlflow.genai.optimize_prompts(...)` with the prompt's URI, the train rows, the
verifier's scorers, and an optimizer object: `MetaPromptOptimizer(reflection_model=...)`, or
`GepaPromptOptimizer(reflection_model=..., max_metric_calls=...)` for GEPA. Every metric call runs
the agent on a train row, so each is a real OpenClaw run with its own trace. The text it returns is
registered as a candidate and goes to the same gate as a person's:
"""),
    code("""
for version in dict.fromkeys(a["candidate"] for a in r["attempts"] if a.get("candidate")):
    pv = mlflow.genai.load_prompt(f"prompts:/{name}/{version}")
    print(f"v{pv.version}", {k: pv.tags.get(k) for k in ("author", "gate")})
"""),
    md("""
Both optimizers' first candidates were rejected. MetaPrompt's passed every quality rule and only
broke the **bloat** rule (`SI_MAX_PROMPT_GROWTH`, default 2.0): optimizers grow prompts, and that
rule is there to stop it. The limit belongs to a person. Here one read the diff, found a new
security section rather than padding, raised the limit for this agent, and re-ran the gate. The
decision is written down with this loop's results (`store/results/loop3.json`): MLflow sees only
the re-run gate, which still enforced every other rule.
"""),
    code("""
r.get("guardrail_change")
"""),
    md("""
The re-run then failed on a must-pass row. Reading the row showed the **verifier** was wrong, not
the agent: a safe refusal ("Please don't send me your password…") matched the credential-request
regex. A must-pass check with a false positive blocks good candidates forever, so a person fixed
the pattern (refusals no longer match), confirmed no stored reviewer label changed, and re-ran the
gate. Verify the verifier before you trust its veto:
"""),
    code("""
r.get("verifier_fix")
"""),
    code("""
print(instructions.diff(r["production_before"], r["candidate"]))
"""),
    md("Must-pass at 100% on both splits:"),
    code("""
s.show(s.gate_table(3), pct=True)
"""),
    md("## 6. Measure"),
    code("""
s.show(s.before_after(3))
"""),
    code("""
s.show(s.answers(s.loop_traces(3, "after")))
"""),
    md("""
**What moved:** the verifier itself is now partly learned (the aligned judge), and the candidate
comes from an optimizer. What stays with a person: the must-pass rule, the labels the judge is
aligned to, and the promotion.
"""),
])

# --------------------------------------------------------------------------------------------
notebook("04_autonomous.ipynb", [
    md("""
# Loop 4 · Autonomous: bad escalation on new traffic

The growth team added a retention playbook to AGENTS.md (20% off for anyone who mentions
cancelling, release quarters for upcoming features) and shipped it without the gate. Then a new
kind of question starts arriving: discounts, roadmap dates, contracts.
`knowledge/support_process.md` says escalate.
""" + LADDER + """
In this loop nobody runs the steps. Reviewers label **half** of the new traffic; `selfimprove cycle`
does the rest in one MLflow run and, with `SI_AUTO_PROMOTE=true`, the gate promotes. `monitor`
then watches live traffic and rolls back if the new version does worse than the one it replaced.
"""),
    code(SETUP.format(n=4)),
    code("""
from selfimprove import instructions

c = r["outside_change"]
print(f"v{c['version']} by {c['by']}: {c['message']}\\n")
print(instructions.diff(c["base_version"], c["version"]))
"""),
    md("## 1. Find: new traffic"),
    code("""
before = s.loop_traces(4, "before")
s.show(s.answers(before))
"""),
    md("""
## 2-5. One `cycle`

`cycle` = score → monitor → align → triage → approve → (enough failures?) → improve → gate. Its
guardrails are what make it safe unattended: no signal, no change; a judge that disagrees with
people halts the cycle; the proposer never approves; the holdout stays hidden; must-pass, margin
and bloat rules at the gate; rollback after.
"""),
    md("""
The first scheduled cycle never finished: the process ended (the machine slept) while the gate was
evaluating its candidate, v11. Its runs are marked `KILLED` in the store, and v11 stays an
ungated candidate. Nothing shipped, because promotion only happens after the gate. The next cycle
started from the stored state: the labels, tests and checkpoints the first one had already
written.
"""),
    code("""
ic = r.get("interrupted_cycle")
if ic:
    print(f"cycle {ic['run_id'][:8]}  {ic['status']}")
    for st in ic["steps"]:
        print(f"  {st['job'] or '?':<8} {st['outcome'] or '(interrupted)':<18} {st['message'][:100]}")
"""),
    under("""
every job is an MLflow run (`mlflow.start_run`) tagged `si.job`, `si.outcome` and `si.message`, and
`cycle` is a parent run with each job nested inside it (`tags.mlflow.parentRunId`). A process that
dies leaves its runs `RUNNING`; marking them `KILLED` is a run status, so the history shows the
failed night instead of hiding it:
"""),
    code("""
import mlflow
from selfimprove import config

name = config.settings.prompt_name
if ic:
    print(ic["run_id"][:8], mlflow.get_run(ic["run_id"]).info.status)
"""),
    md("The cycle that completed:"),
    code("""
print(r["cycle"]["outcome"], "-", r["cycle"]["message"])
r["cycle"]["metrics"]
"""),
    code("""
import mlflow

cycle_run = mlflow.get_run(r["cycle"]["run_id"])
steps = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{cycle_run.info.run_id}'",
                           order_by=["start_time ASC"], output_format="list")
s.show(s.pd.DataFrame([{"job": x.data.tags.get("si.job"), "outcome": x.data.tags.get("si.outcome"),
                        "message": (x.data.tags.get("si.message") or "")[:110]} for x in steps]))
"""),
    code("""
print(instructions.diff(r["production_before"], r["candidate"]))
"""),
    code("""
s.show(s.gate_table(4), pct=True)
"""),
    md("## 6. Measure, and monitor"),
    code("""
s.show(s.before_after(4))
"""),
    code("""
print(r["monitor"]["outcome"], "-", r["monitor"]["message"])
"""),
    under("""
a gate promotion makes the same alias moves as a person's, with `promoted_by=gate` and the live
failure rate it replaced stored as version tags. `monitor` scores traffic on the new version
(`tags.si_prompt_version`), compares it with that recorded rate, and on a regression moves
`production` back with `set_prompt_alias`. It acts only on versions the gate promoted:
"""),
    code("""
prod = mlflow.genai.load_prompt(f"prompts:/{name}@production")
print(f"production -> v{prod.version}")
{k: v for k, v in prod.tags.items() if k in ("promoted_by", "replaced_version", "replaced_failure_rate", "gate")}
"""),
    md("""
## The whole run

Every promotion is an alias move with a record: who promoted, from which gate run, and the live
failure rate it replaced.
"""),
    code("""
from mlflow import MlflowClient
from selfimprove import config

rows = []
for v in sorted(MlflowClient().search_prompt_versions(config.settings.prompt_name),
                key=lambda p: int(p.version)):
    t = v.tags or {}
    rows.append({"version": v.version, "author": t.get("author", ""), "gate": t.get("gate", ""),
                 "promoted_by": t.get("promoted_by", ""), "message": (v.commit_message or "")[:90]})
s.show(s.pd.DataFrame(rows))
"""),
    code("""
s.show(s.pd.DataFrame({f"loop {n}": {"failure before": s.results(n)["before"]["failure_rate"],
                                    "failure after": s.results(n).get("after", {}).get("failure_rate")}
                       for n in (1, 2, 3, 4)}).T, pct=True)
"""),
    md("""
**What stays with a person**, even here: the golden cases, labels on a sample of traffic, the
human queue (contradictions, flagged tests, halts), the must-pass rules and the promotion
margin. `.github/workflows/selfimprove-loop.yml` runs the same jobs on a schedule.
"""),
])
print("wrote", sorted(p.name for p in OUT.glob("*.ipynb")))
