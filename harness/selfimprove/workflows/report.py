"""Read the baseline (Part 2 section 7, Part 3 section 1): the numbers that say how much of the
loop to automate next.

    traces in        too few: not enough signal for downstream automation
    themes out       many traces in few consistent themes: automating theme assignment pays off
    tests generated  many per theme: the triage agent writes one test per complaint; cap it
    human edits      many: the agent's criteria or its judge don't match the reviewers yet

Plus the state of the verifier and the queue a person still owns.
"""

from __future__ import annotations

import datetime as dt

import mlflow

from selfimprove import config, instructions, store


def _runs(job: str, n: int):
    return mlflow.search_runs(filter_string=f"tags.`si.job` = '{job}'", order_by=["start_time ASC"],
                              max_results=n, output_format="list")


def report(n: int = 10) -> str:
    triages, approvals = _runs("triage", 200)[-n:], _runs("approve", 200)
    lines = ["triage runs",
             f"  {'started':<17}{'traces in':>10}{'themes':>8}{'proposed':>10}{'approved':>10}{'flagged':>9}"]
    for t in triages:
        m = t.data.metrics
        approver_run = t.data.tags.get("si.approved_by_run")
        a = next((r.data.metrics for r in approvals if r.info.run_id == approver_run), {})
        when = dt.datetime.fromtimestamp(t.info.start_time / 1000).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {when:<17}{m.get('traces_in', 0):>10.0f}{m.get('themes_out', 0):>8.0f}"
                     f"{m.get('tests_proposed', 0):>10.0f}{a.get('tests_approved', 0):>10.0f}"
                     f"{a.get('tests_flagged', 0):>9.0f}")
    if not triages:
        lines.append("  (none yet)")

    every = store.queue_items(status=None)
    edited = sum(1 for i in every if i.get("status") == "edited")
    accepted = sum(1 for i in every if i.get("status") == "accepted")
    open_by_kind: dict[str, int] = {}
    for i in store.queue_items():
        open_by_kind[i["kind"]] = open_by_kind.get(i["kind"], 0) + 1

    exp = mlflow.get_experiment_by_name(config.settings.experiment)
    agreement = (exp.tags.get("si.judge_agreement") if exp else None) or "not measured yet"
    lines += ["",
              f"human queue      {sum(open_by_kind.values())} open {open_by_kind or ''}; "
              f"{accepted} accepted as-is, {edited} edited by a person",
              f"judge agreement  {agreement}",
              f"dataset          {store.summary()}"]
    if instructions.exists():
        prod = instructions.load("production")
        lines.append(f"production       v{prod.version} (promoted by {(prod.tags or {}).get('promoted_by', 'seed')}), "
                     f"latest v{instructions.latest_version()}")
    return "\n".join(lines)
