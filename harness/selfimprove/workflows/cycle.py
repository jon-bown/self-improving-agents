"""One pass of the whole loop, as one MLflow run with each job nested inside it:

    score -> monitor -> align -> triage -> approve -> [enough failures?] -> improve -> gate

Guardrails that make it safe to run unattended:
  * no signal, no change     improve runs only above SI_FAIL_THRESHOLD with >= SI_MIN_TRACES
  * verifier health          if the judge disagrees with people and re-alignment does not fix it,
                             the cycle halts and queues it for a person instead of optimizing
  * separate approver        the analyst proposes tests; a separate step decides what merges
  * hidden holdout           no improver or approver sees it; only the gate does
  * promotion margin, bloat  the gate's rules, the same as for a person's candidate
  * rollback                 monitor undoes a promotion that live traffic disagrees with
"""

from __future__ import annotations

import mlflow

from selfimprove import config, instructions, store
from selfimprove.workflows import Result, finish, job
from selfimprove.workflows.align import align
from selfimprove.workflows.approve import approve
from selfimprove.workflows.improve import improve
from selfimprove.workflows.score import score
from selfimprove.workflows.ship import live_failure_rate, monitor, run_gate
from selfimprove.workflows.triage import triage


def cycle(improver: str | None = None, force: bool = False) -> Result:
    s = config.settings
    with job("cycle", improver=improver or s.improver, auto_promote=s.auto_promote):
        steps = [score(), monitor()]
        if steps[-1].outcome == "rolled_back":
            return finish(Result("cycle", "rolled_back", steps[-1].message))

        verifier = align()
        steps.append(verifier)
        if verifier.outcome == "not_improved":
            store.enqueue("halt", verifier.message)
            return finish(Result("cycle", "halted_untrusted_judge", verifier.message))

        steps += [triage(force=force), approve()]
        prod = instructions.load("production")
        rate, n = live_failure_rate(int(prod.version))
        mlflow.log_metrics({"live_failure_rate": rate, "live_judged_traces": n})
        if not force and n < s.min_traces:
            return finish(Result("cycle", "waiting", f"{n}/{s.min_traces} judged traces on v{prod.version}"))
        if not force and rate <= s.fail_threshold:
            return finish(Result("cycle", "healthy", f"v{prod.version} live failure {rate:.0%} <= {s.fail_threshold:.0%}"))

        candidate = improve(improver)
        if candidate.outcome != "candidate":
            return finish(Result("cycle", candidate.outcome, candidate.message))
        shipped = run_gate(candidate.data["version"])
        return finish(Result("cycle", shipped.outcome, shipped.message,
                             {"live_failure_rate": rate, "candidate_version": candidate.data["version"]}))
