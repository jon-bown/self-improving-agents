"""Keep the signal flowing (Part 2 section 1, Part 3 section 3).

`score` runs the reference-free scorers over production traces that have no verdict yet, with
no predict_fn, so nothing is re-run: the verdicts land on the existing traces as assessments
(source CODE or LLM_JUDGE), next to whatever reviewers have said. With a first-pass decision
model configured (SI_FAST_JUDGE_MODEL, e.g. Jev), the quality and grounded judges ask it first
and re-judge only the uncertain band with the text judge.

`online` registers judges for MLflow's automatic evaluation instead, so the server samples live
traces as they arrive. With a decision model, it samples at a high rate (it is cheap) and writes
`quality_first_pass`; `score` then turns each of those into the loop's `quality` verdict: copied
when the model is confident, re-judged by the text judge (with a rationale) when it is not. Online scoring needs `mlflow server` with a database backend.
Code scorers (checks, span checks) cannot be registered on open-source MLflow, so `score` runs
those either way.
"""

from __future__ import annotations

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from selfimprove import config, signals, store, traces
from selfimprove.workflows import Result, finish, job


def _pending_first_pass(trace):
    """A server-written first-pass verdict not yet turned into a `quality` verdict, or None."""
    first = traces.feedback(trace, signals.FIRST_PASS)
    if first is None or any(a.name == traces.QUALITY and a.feedback is not None
                            and a.source.source_type == AssessmentSourceType.LLM_JUDGE
                            for a in trace.info.assessments):
        return None
    return first


def resolve_first_pass(trs) -> tuple[int, int]:
    """Turn first-pass verdicts into `quality`: copy confident ones, re-judge the uncertain band."""
    s = config.settings
    text = signals.quality_judge()
    copied = rejudged = 0
    for t in trs:
        first = _pending_first_pass(t)
        if first is None:
            continue
        p = signals.probability(first)
        if signals.uncertain(p) and text is not None:
            verdict = text(inputs={s.input_key: traces.question_of(t)}, outputs=traces.response_of(t))
            value, source = verdict.value, s.judge_model
            why = f"[re-judged: {s.fast_judge_model} p(pass)={p if p is None else round(p, 2)}] {verdict.rationale or ''}"
            rejudged += 1
        else:
            value, source = first.value, first.source.source_id
            why = f"{source} p(pass)={p if p is None else round(p, 2)}, outside the uncertain band; not re-judged."
            copied += 1
        mlflow.log_feedback(trace_id=t.info.trace_id, name=traces.QUALITY, value=value, rationale=why,
                            source=AssessmentSource(AssessmentSourceType.LLM_JUDGE, source),
                            metadata={"si.escalated": str(source == s.judge_model).lower(),
                                      signals.PROBABILITY: str(p)})
    return copied, rejudged


def score(limit: int = 500) -> Result:
    with job("score"):
        since = store.load_checkpoint("score")
        started = traces.now_ms()
        recent = traces.production(since_ms=since or None, max_results=limit)
        todo = [t for t in recent if traces.unscored(t)]
        scorers = signals.live()
        if todo and scorers:
            mlflow.genai.evaluate(data=todo, scorers=scorers)
        copied, escalated = resolve_first_pass([t for t in recent if t not in todo])  # the server's verdicts
        store.save_checkpoint("score", started)
        names = ", ".join(s.name for s in scorers)
        return finish(Result("score", "scored" if todo or copied or escalated else "nothing_new",
                             f"{len(todo)} new production traces scored with [{names}]"
                             + (f"; server first-pass verdicts: {copied} kept, {escalated} re-judged"
                                if copied or escalated else ""),
                             {"traces_scored": len(todo), "scorers": len(scorers), "first_pass_kept": copied,
                              "escalated": escalated}))


def online(sample_rate: float = 0.1, fast_sample_rate: float = 1.0, filter_string: str | None = None) -> Result:
    """Register LLM judges for server-side sampling of production traces."""
    s = config.settings
    with job("online", sample_rate=sample_rate, fast_sample_rate=fast_sample_rate):
        from mlflow.genai.scorers import ScorerSamplingConfig

        flt = filter_string or "tags.si_source = 'production'"
        plan = [(j, sample_rate) for j in signals.guidelines() + signals.themes()]
        if s.has_fast_judge:
            spec = signals.criteria().get("quality") or {}
            plan.append((signals.fast_judge(signals.FIRST_PASS, spec.get("fast_instructions")
                                            or spec.get("instructions", signals.DEFAULT_QUALITY)), fast_sample_rate))
        elif (q := signals.quality_judge()) is not None:
            plan.append((q, sample_rate))
        started = []
        for judge, rate in plan:
            try:
                registered = judge.register(name=judge.name)
            except Exception:
                registered = judge  # already registered (themes, an aligned quality judge)
            registered.start(sampling_config=ScorerSamplingConfig(sample_rate=rate, filter_string=flt))
            started.append(f"{judge.name}@{rate:.0%}")
        return finish(Result("online", "started", f"sampling '{flt}' with {started}", {"judges": len(started)}))
