"""Approve the tests (Part 3, section 6). A separate step from triage, so the agent that proposed
a test never approves it.

It picks up every triage run tagged si.proposal=pending, checks each proposed test, merges the
approved ones into the dataset with provenance tags, and sends the rest to the human queue. Each
real (non-noise) theme with at least one approved test becomes a registered Guidelines judge,
theme_<slug>, so the verifier grows with the evidence (Part 2, section 6).
"""

from __future__ import annotations

import mlflow

from selfimprove import approver, config, signals, store
from selfimprove.workflows import Result, finish, job


def pending_runs() -> list:
    return mlflow.search_runs(filter_string="tags.`si.job` = 'triage' AND tags.`si.proposal` = 'pending'",
                              order_by=["start_time ASC"], output_format="list")


def approve(run_id: str | None = None, use_reviewer: bool = True) -> Result:
    client = mlflow.MlflowClient()
    with job("approve") as run:
        triage_runs = [client.get_run(run_id)] if run_id else pending_runs()
        if not triage_runs:
            return finish(Result("approve", "nothing_pending", "no pending proposals"))
        approved, flagged, registered, decided = 0, 0, [], []
        for tr in triage_runs:
            proposal = mlflow.artifacts.load_dict(f"runs:/{tr.info.run_id}/proposal.json")
            verdicts = approver.approve(proposal["records"], use_reviewer=use_reviewer)
            ok = []
            for v in verdicts:
                v.record["tags"]["approved_by"] = "approver"
                if v.approved:
                    ok.append(v.record)
                else:
                    store.enqueue("flagged_test", "; ".join(v.reasons)[:500],
                                  trace_id=v.record["tags"].get("source_trace"), record=v.record, checks=v.checks)
            store.merge(ok)
            approved += len(ok)
            flagged += len(verdicts) - len(ok)

            themes_with_tests = {r["tags"].get("theme") for r in ok}
            for th in proposal.get("themes", []):
                if th["slug"] in themes_with_tests and not th.get("is_reviewer_noise") and config.settings.has_llm:
                    signals.register_theme(th["slug"], th["criterion"])
                    registered.append(th["slug"])

            decided += [{"triage_run": tr.info.run_id, "input": store.question_of(v.record), "approved": v.approved,
                         "checks": v.checks, "reasons": v.reasons} for v in verdicts]
            client.set_tag(tr.info.run_id, "si.proposal", "done")
            client.set_tag(tr.info.run_id, "si.approved_by_run", run.info.run_id)

        mlflow.log_dict({"verdicts": decided}, "verdicts.json")
        store.tag_dataset(generated_by="triage-agent", source_run=run.info.run_id,
                          tests_per_theme_cap=config.settings.tests_per_theme_cap,
                          holdout_fraction=config.settings.holdout_fraction)
        return finish(Result("approve", "merged",
                             f"{approved} tests merged, {flagged} sent to the human queue, theme judges {registered}; "
                             f"dataset now {store.summary()}",
                             {"tests_approved": approved, "tests_flagged": flagged, "theme_judges": len(registered)}))
