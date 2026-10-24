"""Run the session's four loops for real, into session/store/, ahead of the talk.

Each loop finds one seeded failure in OpenClaw's answers, turns the signal into tests, writes a
fix, gates it, and ships it. What changes from loop to loop is how much of that a person does:

    loop 1  manual       you label, you add the tests, you write v2, you promote
    loop 2  assisted     triage + approve build the tests, the analyst writes v3, you promote
    loop 3  optimizer    the judge is aligned to your labels, MetaPrompt writes the fix, you promote
    loop 4  autonomous   `cycle` does all of it and the gate promotes; monitor watches

Loops 1 and 2 fix flaws that v1 shipped with. Loops 3 and 4 start with a change someone else
made to AGENTS.md and shipped without the gate (changes/), which is where their failures come from.

Every step calls the same harness functions as the `selfimprove` CLI (RUNBOOK.md has the
commands). Results go to MLflow (traces, assessments, the dataset, prompt versions, one run per
job) and a summary per loop to store/results/loopN.json, which the notebooks read.

    make session-server                        # the store's MLflow server, in another terminal
    .venv/bin/python -m session.precompute     # all four loops (about two hours)
    .venv/bin/python -m session.precompute --loops 3 4
    .venv/bin/python -m session.precompute --fresh   # wipe the store first (stop the server)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
STORE = HERE / "store"
RESULTS = STORE / "results"

LOOPS = {
    1: {"title": "Manual: the agent skips the lookup", "failure": "lookup",
        "criteria": "criteria/loop1.yaml", "traffic": "traffic/loop1.txt"},
    2: {"title": "Assisted: the agent improvises when the docs are silent", "failure": "hallucination",
        "criteria": "criteria/loop2.yaml", "traffic": "traffic/loop2.txt"},
    3: {"title": "Optimizer: unsafe account help", "failure": "account",
        "criteria": "criteria/loop3.yaml", "traffic": "traffic/loop3.txt"},
    4: {"title": "Autonomous: bad escalation on new traffic", "failure": "escalation",
        "criteria": "criteria/loop4.yaml", "traffic": "traffic/loop4.txt"},
}


def load_keys() -> None:
    """Provider API keys from harness/.env. Only the keys: its other settings are the CLI's."""
    from dotenv import dotenv_values

    for k, v in dotenv_values(HARNESS / ".env").items():
        if k.endswith("_API_KEY") and v and not os.environ.get(k):
            os.environ[k] = v


def configure(loop: int, **overrides) -> None:
    """Point the harness at session.env with this loop's verifier."""
    os.chdir(HARNESS)
    os.environ["SI_ENV_FILE"] = str(HERE / "session.env")
    os.environ["SI_CRITERIA_FILE"] = str(HERE / LOOPS[loop]["criteria"])
    os.environ.pop("ANTHROPIC_BASE_URL", None)
    for k, v in overrides.items():
        os.environ[k] = str(v)
    load_keys()
    from selfimprove import config, signals, traces

    config.reload()
    signals.reload()
    traces.setup()


def questions(loop: int) -> list[str]:
    lines = (HERE / LOOPS[loop]["traffic"]).read_text().splitlines()
    return [q.strip() for q in lines if q.strip() and not q.startswith("#")]


def traffic(loop: int, phase: str) -> list[str]:
    """Send the loop's questions to production. Returns the trace ids."""
    import mlflow

    from selfimprove import agents

    ids = []
    for q in questions(loop):
        try:
            agents.run(q, source="production", tags={"session_loop": str(loop), "session_phase": phase})
        except Exception as e:  # noqa: BLE001 -- the trace records the error; keep going
            print(f"   !! {q[:60]}: {e}")
        ids.append(mlflow.get_last_active_trace_id())
        print(f"   {phase:<6} {q[:70]}")
    mlflow.flush_trace_async_logging()
    return [i for i in ids if i]


def review(trace_ids: list[str], sample: float = 1.0) -> dict:
    """The simulated reviewers label these traces. Returns the failure rate they saw."""
    import mlflow

    from selfimprove import feedback, traces
    from session import reviewer

    key, fails, n = reviewer.cases(), 0, 0
    for i, tid in enumerate(trace_ids):
        if i / max(len(trace_ids), 1) >= sample:
            break
        t = mlflow.get_trace(tid)
        case = key.get(traces.question_of(t).strip().lower())
        if case is None:
            continue
        ok, why = reviewer.verdict(case, traces.response_of(t))
        feedback.label(tid, passed=ok, rationale=why, reviewer=reviewer.REVIEWERS[i % 3],
                       expected_response=None if ok else case["expected_response"])
        n, fails = n + 1, fails + (not ok)
    return {"labeled": n, "failed": fails, "failure_rate": fails / n if n else 0.0}


def outside_change(path: str, by: str, message: str) -> dict:
    """Someone outside the loop edits AGENTS.md and ships it straight to production, no gate.

    The section in `path` is appended to whatever production is now. This is how loops 3 and 4
    get their failures: real teams change an agent's instructions for their own reasons.
    """
    from selfimprove import instructions

    prod = instructions.load("production")
    text = prod.template.rstrip("\n") + "\n" + (HERE / path).read_text()
    version = instructions.register_candidate(text, message, author=by, base_version=prod.version)
    instructions.promote(version, reason=f"{by}: shipped directly, without the gate", by=by)
    print(f"-- outside change: {by} shipped v{version} ({message})")
    return {"version": version, "base_version": int(prod.version), "by": by, "message": message, "file": path}


def clear_queue(note: str) -> dict:
    """A person reviews the tests the approver flagged and accepts them into the dataset."""
    from selfimprove import store

    items = [i for i in store.queue_items("open") if i["kind"] == "flagged_test"]
    for i in items:
        store.resolve(i["id"], "accepted", note=note)
    print(f"   queue: a person accepted {len(items)} flagged tests; {store.summary()}")
    return {"accepted": len(items), "dataset": store.summary()}


def gate_summary(outcome: str, message: str, metrics: dict) -> dict:
    """The latest gate run: its decision and each scorecard's pass rates."""
    import mlflow

    run = mlflow.get_run(mlflow.search_runs(filter_string="tags.`si.job` = 'gate'", max_results=1,
                                            order_by=["start_time DESC"]).iloc[0]["run_id"])
    cards = {}
    for child in mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
                                    output_format="list"):
        label = child.data.tags.get("si.label") or child.info.run_name
        cards[label] = {k.removesuffix("/mean"): v for k, v in child.data.metrics.items() if k.endswith("/mean")}
    return {"outcome": outcome, "message": message, "run_id": run.info.run_id,
            "passed": run.data.tags.get("si.gate_passed") == "True",
            "reasons": run.data.tags.get("si.gate_reasons", ""), "cards": cards,
            "train_gain": metrics.get("train_gain"), "holdout_gain": metrics.get("holdout_gain")}


def save(loop: int, summary: dict) -> None:
    from selfimprove import store

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary["queue"] = store.queue_items(None)  # the human queue lives in .state/, not in MLflow
    summary = {"loop": loop, **LOOPS[loop], **summary, "finished_at": time.strftime("%Y-%m-%d %H:%M")}
    (RESULTS / f"loop{loop}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"== loop {loop} saved: {RESULTS / f'loop{loop}.json'}")


def before_after(loop: int, out: dict) -> None:
    """Replay the loop's traffic on the new production version and have it reviewed again."""
    from selfimprove import instructions

    out["production_after"] = int(instructions.load("production").version)
    print(f"-- after: loop {loop} traffic on v{out['production_after']}")
    out["after_traces"] = traffic(loop, "after")
    out["after"] = review(out["after_traces"])
    print(f"   failure rate {out['before']['failure_rate']:.0%} -> {out['after']['failure_rate']:.0%}")


# --------------------------------------------------------------------------------------------
def loop1() -> dict:
    """Manual. A person reads the traces, labels them, adds the failures as tests, writes v2."""
    from selfimprove import instructions, store, traces
    from selfimprove.workflows import score as score_wf
    from selfimprove.workflows import ship

    configure(1)
    out = {"production_before": int(instructions.load("production").version)}
    print("-- signal: loop 1 traffic, scored and labeled")
    out["before_traces"] = traffic(1, "before")
    out["score"] = str(score_wf.score())
    out["before"] = review(out["before_traces"])

    print("-- store: labeled failures -> tests (by hand)")
    import mlflow

    fails = [mlflow.get_trace(t) for t in out["before_traces"]]
    fails = [t for t in fails if traces.failed(t) and traces.expectation(t)]
    records, seen = [], []
    for t in fails:
        r = store.record_from_trace(t, source="human_fail", author="you", theme="skips_lookup")
        if store.near_duplicate(store.question_of(r), seen) is None:
            records.append(r)
            seen.append(r)
    out["tests_added"] = store.merge(records)
    store.save_checkpoint("triage")  # these failures are handled; triage starts after them
    out["dataset"] = store.summary()

    print("-- update: v2 written by a person")
    v2 = instructions.register_candidate((HERE / "fixes" / "v2_manual.md").read_text(),
                                         "v2: read the policy files by name, drop the stale quick facts "
                                         "(evidence: loop 1 failures, read_knowledge and no_stale_facts)",
                                         author="you")
    out["candidate"] = v2
    result = ship.run_gate(v2, promote=False)
    out["gate"] = gate_summary(result.outcome, result.message, result.metrics)
    if out["gate"]["passed"]:
        out["promote"] = str(ship.promote(v2, reason="loop 1: gate passed; reviewed the diff and the traces"))
        before_after(1, out)
    return out


def loop2() -> dict:
    """Assisted. Triage groups the failures and proposes tests, the approver merges them, the
    analyst writes the candidate. A person reviews the proposal and promotes."""
    from selfimprove import instructions, store
    from selfimprove.workflows import approve, improve, score, ship, triage

    configure(2)
    out = {"production_before": int(instructions.load("production").version)}
    print("-- signal: loop 2 traffic, scored and labeled")
    out["before_traces"] = traffic(2, "before")
    out["score"] = str(score.score())
    out["before"] = review(out["before_traces"])

    print("-- store: triage + approve")
    t = triage.triage(force=True)
    out["triage"] = {"outcome": t.outcome, "message": t.message, "metrics": t.metrics}
    a = approve.approve()
    out["approve"] = {"outcome": a.outcome, "message": a.message, "metrics": a.metrics}
    out["dataset"] = store.summary()

    loop2_update(out)
    return out


def loop2_update(out: dict, attempts: list | None = None) -> dict:
    """The analyst writes a candidate and the gate decides. If the gate rejects it, a person
    reviews the human queue (the tests the approver flagged) and the analyst tries once more."""
    from selfimprove.workflows import improve, ship

    out["attempts"] = attempts or []
    while len(out["attempts"]) < 2:
        if out["attempts"] and "queue_review" not in out:
            out["queue_review"] = clear_queue("loop 2: checked against knowledge/; the docs don't cover these")
        print("-- update: the analyst")
        c = improve.improve("analyst")
        attempt = {"improver": "analyst", "outcome": c.outcome, "message": c.message}
        if c.outcome == "candidate":
            attempt["candidate"] = c.data["version"]
            result = ship.run_gate(attempt["candidate"], promote=False)
            attempt["gate"] = gate_summary(result.outcome, result.message, result.metrics)
        out["attempts"].append(attempt)
        if attempt.get("gate", {}).get("passed"):
            break
    last = out["attempts"][-1]
    out["improve"] = {"outcome": last["outcome"], "message": last["message"]}
    if "candidate" in last:
        out["candidate"], out["gate"] = last["candidate"], last["gate"]
        if out["gate"]["passed"]:
            out["promote"] = str(ship.promote(out["candidate"], reason="loop 2: reviewed the analyst's proposal"))
            before_after(2, out)
    if "queue_review" not in out:  # passed first time: the flagged tests still get a person's review
        out["queue_review"] = clear_queue("loop 2: checked against knowledge/; the docs don't cover these")
    return out


def loop3() -> dict:
    """Optimizer. The quality judge doesn't know the account policy, so it is aligned to the
    reviewers first; then MetaPrompt rewrites the instructions against every scorer."""
    from selfimprove import instructions, store
    from selfimprove.workflows import align, approve, improve, score, ship, triage

    configure(3)
    out = {"outside_change": outside_change("changes/support_team_fast_recovery.md", "support-team",
                                            "Faster account recovery: verify customers in chat")}
    out["production_before"] = int(instructions.load("production").version)
    print("-- signal: loop 3 traffic, scored and labeled")
    out["before_traces"] = traffic(3, "before")
    out["score"] = str(score.score())
    out["before"] = review(out["before_traces"])

    print("-- verifier: align the quality judge to the reviewers")
    al = align.align(force=True)
    out["align"] = {"outcome": al.outcome, "message": al.message, "metrics": al.metrics}

    print("-- store: triage + approve")
    t = triage.triage(force=True)
    out["triage"] = {"outcome": t.outcome, "message": t.message, "metrics": t.metrics}
    a = approve.approve()
    out["approve"] = {"outcome": a.outcome, "message": a.message, "metrics": a.metrics}
    out["dataset"] = store.summary()

    out["queue_review"] = clear_queue("loop 3: checked against the account-recovery policy")

    # The optimizer writes the candidate. If the gate rejects it, that is shown, not hidden, and
    # a second optimizer gets a turn.
    out["attempts"] = []
    for kind, budget in (("metaprompt", 60), ("gepa", 40)):
        print(f"-- update: {kind}")
        c = improve.improve(kind, budget=budget)
        attempt = {"improver": kind, "outcome": c.outcome, "message": c.message}
        if c.outcome == "candidate":
            attempt["candidate"] = c.data["version"]
            result = ship.run_gate(attempt["candidate"], promote=False)
            attempt["gate"] = gate_summary(result.outcome, result.message, result.metrics)
        out["attempts"].append(attempt)
        if attempt.get("gate", {}).get("passed"):
            break
    last = out["attempts"][-1]
    out["improve"] = {"outcome": last["outcome"], "message": last["message"]}
    if "candidate" in last:
        out["candidate"], out["gate"] = last["candidate"], last["gate"]
        if out["gate"]["passed"]:
            out["promote"] = str(ship.promote(out["candidate"], reason=f"loop 3: reviewed the {last['improver']} diff"))
            before_after(3, out)
    return out


def loop4() -> dict:
    """Autonomous. New traffic arrives; reviewers label a sample; `cycle` scores, aligns,
    triages, approves, improves and gates, and the gate promotes. monitor checks the result."""
    from selfimprove import instructions, store
    from selfimprove.workflows import cycle, score, ship

    configure(4, SI_AUTO_PROMOTE="true")
    out = {"outside_change": outside_change("changes/growth_team_retention.md", "growth-team",
                                            "Retention playbook: discounts and roadmap dates")}
    out["production_before"] = int(instructions.load("production").version)
    print("-- new traffic: loop 4, half of it labeled")
    out["before_traces"] = traffic(4, "before")
    out["before"] = review(out["before_traces"], sample=0.5)

    print("-- cycle")
    c = cycle.cycle(improver="analyst")
    out["cycle"] = {"outcome": c.outcome, "message": c.message, "metrics": c.metrics}
    out["dataset"] = store.summary()
    if c.outcome == "promoted":
        out["candidate"] = int(instructions.load("production").version)
        out["gate"] = gate_summary(c.outcome, c.message, c.metrics)
        before_after(4, out)
        print("-- monitor: live traffic on the promoted version")
        out["score_after"] = str(score.score())
        m = ship.monitor()
        out["monitor"] = {"outcome": m.outcome, "message": m.message, "metrics": m.metrics}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--fresh", action="store_true", help="wipe store/ and .state/ first (stop the server)")
    args = p.parse_args()
    if args.fresh:
        shutil.rmtree(STORE, ignore_errors=True)
        shutil.rmtree(HERE / ".state" / "workspaces", ignore_errors=True)
        (HERE / ".state" / "human_queue.jsonl").unlink(missing_ok=True)
        print("wiped store/; start `make session-server`, then run without --fresh")
        return

    configure(1)
    from selfimprove import instructions, store, traces
    from selfimprove.agents import openclaw

    print("openclaw     ", openclaw.check())
    print("mlflow plugin", openclaw.plugin_check(traces.setup()))
    if not instructions.exists():
        print("-- init: AGENTS.md v1 -> production, golden cases -> dataset")
        instructions.seed(commit_message="v1: the agent as we found it")
        store.seed_golden()
    for n in args.loops:
        print(f"\n======== loop {n}: {LOOPS[n]['title']} ========")
        started = time.time()
        summary = {1: loop1, 2: loop2, 3: loop3, 4: loop4}[n]()
        summary["minutes"] = round((time.time() - started) / 60, 1)
        save(n, summary)
        if "after" not in summary:  # later loops build on this one's promotion
            print(f"loop {n} did not ship a new version; stopping. Re-run with --loops {n} ...")
            break


if __name__ == "__main__":
    main()
