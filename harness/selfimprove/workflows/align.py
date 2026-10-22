"""Automate the judge alignment (Part 3, section 5).

The quality judge writes verdicts under the same name as reviewers, so a nightly job can measure
how often they agree on a held-out slice of human-labeled traces. When agreement drops below
SI_AGREEMENT_THRESHOLD, the job aligns the judge to the labels (MemAlign with an embedding model,
SIMBA otherwise) and registers the new version only if held-out agreement improved.

With a first-pass decision model (SI_FAST_JUDGE_MODEL, e.g. Jev) it also calibrates that: how
often the model agrees with people, and the pass probabilities of the verdicts it got wrong. A
decision model gives no rationale, so there is nothing to align; the band of probabilities that
gets re-judged (SI_FAST_JUDGE_BAND) is what you tune, and the job suggests one from your labels.

It skips nights with too little data: MLflow recommends at least ten labeled traces with at least
30% in each class. The agreement is stored on the experiment (si.judge_agreement), and triage
reads it to decide whether judge-only failures count.
"""

from __future__ import annotations

import hashlib

import mlflow

from selfimprove import config, models, signals, traces
from selfimprove.workflows import Result, finish, job


def split_traces(trs, check_fraction: float = 0.3):
    """Deterministic fit/check split by trace id."""
    def bucket(t):
        return int(hashlib.sha1(t.info.trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return [t for t in trs if bucket(t) >= check_fraction], [t for t in trs if bucket(t) < check_fraction]


def agreement(judge, trs) -> float:
    """Fraction of traces where the judge agrees with the latest human label."""
    hits = n = 0
    for t in trs:
        human = traces.human_feedback(t, judge.name)
        if human is None:
            continue
        verdict = judge(trace=t)
        n += 1
        hits += traces.is_fail(verdict.value) == traces.is_fail(human.value)
    return hits / n if n else float("nan")


def calibrate_first_pass(trs) -> dict:
    """Decision-model verdicts vs human labels: agreement, and where its misses fall.

    Returns agreement, the share of verdicts the current band would re-judge, how many misses
    that band catches, and a suggested band: the narrowest range of pass probabilities that
    contains every miss.
    """
    s = config.settings
    spec = signals.criteria().get("quality") or {}
    fast = signals.fast_judge("quality", spec.get("fast_instructions") or spec.get("instructions", signals.DEFAULT_QUALITY))
    rows = []
    for t in trs:
        human = traces.human_feedback(t)
        if human is None:
            continue
        fb = fast(inputs={s.input_key: traces.question_of(t)}, outputs=traces.response_of(t))
        p = signals.probability(fb)
        if p is not None:
            rows.append((p, not traces.is_fail(human.value)))
    if not rows:
        return {}
    misses = [p for p, human_pass in rows if (p >= 0.5) != human_pass]
    lo, hi = s.fast_judge_band
    return {
        "fast_agreement": 1 - len(misses) / len(rows),
        "fast_rejudge_share": sum(lo <= p <= hi for p, _ in rows) / len(rows),
        "fast_misses": len(misses),
        "fast_misses_in_band": sum(lo <= p <= hi for p in misses),
        "suggested_band": f"{min(misses):.2f},{max(misses):.2f}" if misses else f"{lo},{hi}",
        "labels": len(rows),
    }


def optimizer():
    from mlflow.genai.judges.optimizers import MemAlignOptimizer, SIMBAAlignmentOptimizer

    s = config.settings
    if s.embedding_model and models.has_credentials(s.embedding_model):
        return MemAlignOptimizer(reflection_lm=s.judge_model, embedding_model=s.embedding_model)
    return SIMBAAlignmentOptimizer(model=s.judge_model)


def align(force: bool = False) -> Result:
    s = config.settings
    with job("align", threshold=s.agreement_threshold):
        judge = signals.quality_judge()
        if judge is None:
            return finish(Result("align", "skipped", "no judge model credentials"))
        paired = [t for t in traces.production() if traces.human_feedback(t) is not None]
        fails = sum(traces.is_fail(traces.human_feedback(t).value) for t in paired)
        minority = min(fails, len(paired) - fails) / max(len(paired), 1)
        fit, check = split_traces(paired)
        metrics = {"labeled_traces": len(paired), "minority_class_share": minority}
        if len(paired) < s.min_alignment_traces or minority < 0.3 or not check:
            return finish(Result("align", "skipped",
                                 f"{len(paired)} labeled traces, minority class {minority:.0%} "
                                 f"(need >= {s.min_alignment_traces} and >= 30%)", metrics))
        before = agreement(judge, check)
        metrics["agreement_before"] = before
        mlflow.set_experiment_tag("si.judge_agreement", f"{before:.4f}")
        if s.has_fast_judge:
            cal = calibrate_first_pass(check)
            metrics.update({k: v for k, v in cal.items() if isinstance(v, (int, float))})
            if cal:
                mlflow.set_experiment_tag("si.fast_band_suggested", cal["suggested_band"])
                print(f"[align] {s.fast_judge_model}: agrees with people on {cal['fast_agreement']:.0%} of "
                      f"{cal['labels']} labels; {cal['fast_misses_in_band']}/{cal['fast_misses']} misses fall in the "
                      f"re-judge band {s.fast_judge_band}, which re-judges {cal['fast_rejudge_share']:.0%} of verdicts. "
                      f"Band that would catch every miss: {cal['suggested_band']}")
        if before >= s.agreement_threshold and not force:
            return finish(Result("align", "healthy", f"judge agrees with people on {before:.0%} of "
                                 f"{len(check)} held-out labels", metrics))
        aligned = judge.align(fit, optimizer=optimizer())
        after = agreement(aligned, check)
        metrics["agreement_after"] = after
        if after > before:
            aligned.register(name="quality")
            mlflow.set_experiment_tag("si.judge_agreement", f"{after:.4f}")
            return finish(Result("align", "aligned", f"agreement {before:.0%} -> {after:.0%}; registered", metrics))
        return finish(Result("align", "not_improved",
                             f"agreement {before:.0%} -> {after:.0%}; kept the current judge. "
                             "A person should look at where reviewers and the judge disagree.", metrics))
