"""Schedule the trace ingestion (Part 3, section 2).

Runs weekly, or sooner once SI_TRIAGE_THRESHOLD new labeled failures have arrived. It reads the
production traces since its checkpoint and sorts them:

    reviewers disagree on a trace      -> human queue (contradiction), never a test
    human FAIL with an expectation     -> a proposed test
    human FAIL without an expectation  -> human queue: someone must write the right answer
    only a judge or check failed it    -> human queue for a label, sampled. A judge verdict never
                                          becomes an expectation, and judge-only failures are
                                          ignored entirely while the judge disagrees with people.

The analyst (an LLM) groups the failures into themes and writes one criterion per theme. Proposed
tests are logged on the triage run (proposal.json, tag si.proposal=pending) for `approve`, the
separate step that decides what joins the dataset. The checkpoint only moves once the proposal is saved.
"""

from __future__ import annotations

import mlflow

from selfimprove import config, improvers, instructions, signals, store, traces
from selfimprove.workflows import Result, finish, job

MAX_JUDGE_ONLY_TO_QUEUE = 20


def judge_trusted() -> bool:
    exp = mlflow.get_experiment_by_name(config.settings.experiment)
    agreement = exp.tags.get("si.judge_agreement") if exp else None
    return agreement is None or float(agreement) >= config.settings.agreement_threshold


def triage(force: bool = False, use_analyst: bool = True) -> Result:
    s = config.settings
    with job("triage", force=force) as run:
        since = store.load_checkpoint("triage")
        started = traces.now_ms()
        trs = traces.production(since_ms=since or None)
        must = signals.must_pass()
        holdout_qs = store.holdout_questions()

        contradictions, proposals, judge_only, needs_exp = [], [], [], []
        for t in trs:
            labels = traces.human_labels(t)
            if len(set(labels.values())) > 1:
                contradictions.append(t)
                store.enqueue("contradiction", f"reviewers disagree: {labels}", trace_id=t.info.trace_id)
                continue
            fb = traces.human_feedback(t)
            if fb is not None:
                if not traces.is_fail(fb.value):
                    continue
                if traces.expectation(t) is None:
                    needs_exp.append(t)
                    store.enqueue("needs_expectation", f"failed by {fb.source.source_id}: {fb.rationale or ''}"[:300],
                                  trace_id=t.info.trace_id)
                else:
                    proposals.append(t)
            elif traces.failed(t, must):
                judge_only.append(t)

        trusted = judge_trusted()
        if trusted:
            for t in judge_only[:MAX_JUDGE_ONLY_TO_QUEUE]:
                store.enqueue("judge_only_failure", "; ".join(f["rationale"] for f in traces.failing_verdicts(t))[:300],
                              trace_id=t.info.trace_id)

        metrics = {"traces_in": len(trs), "human_failures": len(proposals), "contradictions": len(contradictions),
                   "needs_expectation": len(needs_exp), "judge_only_failures": len(judge_only)}
        if len(proposals) < s.triage_threshold and not force:
            return finish(Result("triage", "waiting",
                                 f"{len(proposals)}/{s.triage_threshold} labeled failures since the last triage",
                                 metrics))

        # The holdout is hidden from the analyst: it never reads a failure whose input is (or
        # would become) a holdout case. Those still become tests, in the holdout split.
        visible = [t for t in proposals if not store.is_holdout(traces.question_of(t), holdout_qs)]
        judged = [t for t in judge_only if not store.is_holdout(traces.question_of(t), holdout_qs)] if trusted else []
        report = improvers.failure_report(visible + judged)
        themes, theme_of = [], {}
        if use_analyst and s.has_llm and report:
            analysis = improvers.analyze(report, instructions.load("production").template)
            themes = analysis.get("themes", [])
            for th in themes:
                for tid in th.get("trace_ids", []):
                    theme_of.setdefault(tid, th["slug"])
            mlflow.log_dict(analysis, "analysis.json")
        noise = {th["slug"] for th in themes if th.get("is_reviewer_noise")}

        records, seen, covered = [], [], 0
        existing = store.records()
        for t in proposals:
            theme = theme_of.get(t.info.trace_id) or _fallback_theme(t)
            if theme in noise:
                continue
            if store.near_duplicate(traces.question_of(t), existing, threshold=1.0):
                covered += 1  # the suite already tests this input
                continue
            rec = store.record_from_trace(t, source="human_fail", author="triage-agent", theme=theme)
            rec["tags"]["source_run"] = run.info.run_id
            rec["tags"]["reason"] = "; ".join(f["rationale"] for f in traces.failing_verdicts(t))[:300]
            if store.near_duplicate(store.question_of(rec), seen) is None:  # forty complaints, one test
                records.append(rec)
                seen.append(rec)

        # The proposal lives on this run (artifact + tag), so `approve` can pick it up from any machine.
        mlflow.log_dict({"themes": themes, "records": records}, "proposal.json")
        mlflow.set_tag("si.proposal", "pending" if records else "empty")
        store.save_checkpoint("triage", started)
        metrics.update(themes_out=len(themes), tests_proposed=len(records), already_covered=covered)
        return finish(Result("triage", "proposed",
                             f"{len(records)} tests in {len(themes) or 'no'} themes from {len(trs)} traces, "
                             f"{covered} already covered "
                             f"(proposal on run {run.info.run_id[:8]})", metrics, {"proposal_run": run.info.run_id}))


def _fallback_theme(trace) -> str:
    """Without an analyst, the first failing verifier names the theme (e.g. no_stale_policy)."""
    fails = [f for f in traces.failing_verdicts(trace) if f["name"] != traces.QUALITY]
    return improvers.slugify(fails[0]["name"]) if fails else traces.QUALITY
