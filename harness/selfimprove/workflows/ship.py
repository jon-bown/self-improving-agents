"""Promote the prompt, and take it back if live traffic says so (Part 3, sections 7-8).

`gate` scores the candidate and production on the same CI suite (train) and the same holdout and
applies selfimprove.gate's rules. It moves the production alias only when the gate passes AND
either SI_AUTO_PROMOTE=true or a person passes --promote. Each promotion records the version it
replaced and the live failure rate at the time.

`monitor` watches a version the gate promoted. If its live failure rate climbs more than
SI_ROLLBACK_MARGIN above the rate it replaced, production rolls back: one alias move.
"""

from __future__ import annotations

from selfimprove import config, gate, instructions, signals, traces
from selfimprove.workflows import Result, finish, job


def live_failure_rate(version: int, since_ms: int | None = None) -> tuple[float, int]:
    trs = traces.production(f"tags.si_prompt_version = '{version}'", since_ms=since_ms)
    must = signals.must_pass()
    judged = [t for t in trs if traces.has_verdict(t, must)]
    return (sum(traces.failed(t, must) for t in judged) / len(judged) if judged else 0.0), len(judged)


def run_gate(candidate: int | None = None, promote: bool | None = None) -> Result:
    s = config.settings
    candidate = candidate or instructions.latest_version()
    with job("gate", candidate=candidate):
        prod = instructions.load("production")
        if int(prod.version) == candidate:
            return finish(Result("gate", "nothing_to_do", f"v{candidate} is already production"))
        decision = gate.compare(candidate)
        print(decision.table())
        metrics = {"train_gain": decision.gain("train") or 0.0, "holdout_gain": decision.gain("holdout") or 0.0}
        if not decision.passed:
            return finish(Result("gate", "rejected", decision.summary(), metrics))
        if not (promote if promote is not None else s.auto_promote):
            return finish(Result("gate", "awaiting_approval",
                                 f"v{candidate} passed; run `selfimprove promote {candidate}` to ship it", metrics))
        rate, _ = live_failure_rate(int(prod.version))
        by = "human" if promote else "gate"
        instructions.promote(candidate, reason=f"gate run {decision.run_id}", by=by,
                             replaced_failure_rate=f"{rate:.4f}")
        return finish(Result("gate", "promoted", f"v{prod.version} -> v{candidate} (by {by})", metrics))


def promote(version: int, reason: str = "approved by a person") -> Result:
    with job("promote", version=version):
        prod = instructions.load("production")
        rate, _ = live_failure_rate(int(prod.version))
        previous = instructions.promote(version, reason=reason, by="human", replaced_failure_rate=f"{rate:.4f}")
        return finish(Result("promote", "promoted", f"v{previous} -> v{version}"))


def monitor() -> Result:
    s = config.settings
    with job("monitor"):
        prod = instructions.load("production")
        tags = prod.tags or {}
        rate, n = live_failure_rate(int(prod.version), since_ms=int(tags.get("promoted_at", 0)) or None)
        metrics = {"live_failure_rate": rate, "live_judged_traces": n, "production_version": int(prod.version)}
        if tags.get("promoted_by") != "gate" or "replaced_failure_rate" not in tags:
            return finish(Result("monitor", "watching", f"v{prod.version}: live failure {rate:.0%} on {n} traces "
                                 "(not auto-promoted, so no automatic rollback)", metrics))
        before = float(tags["replaced_failure_rate"])
        if n >= s.min_traces and rate > before + s.rollback_margin:
            restored = instructions.rollback(f"live failure {rate:.0%} vs {before:.0%} before promotion")
            return finish(Result("monitor", "rolled_back",
                                 f"v{prod.version} -> v{restored}: live failure {rate:.0%} vs {before:.0%}", metrics))
        return finish(Result("monitor", "healthy", f"v{prod.version}: live failure {rate:.0%} on {n} traces "
                             f"(replaced {before:.0%})", metrics))
