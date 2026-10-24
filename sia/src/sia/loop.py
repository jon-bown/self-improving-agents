"""Level 5: one autonomous improvement cycle. Signal -> verifier check -> store -> update -> gate -> ship.

Each cycle is one MLflow run, so the loop's own history (what it saw, what it tried, why it
did or did not ship) is as inspectable as the agent's.

Guardrails that make it safe to run unattended:
  * trigger threshold   -- no signal, no change
  * verifier health     -- if the judge stops agreeing with humans, halt and ask for help
  * holdout + must-pass -- the same gate as every other level
  * bloat guard         -- reject candidates that balloon AGENTS.md (a common judge-pleasing hack)
  * rollback watch      -- if a version the loop shipped does worse live, roll it back
Humans stay in the loop where they are irreplaceable: they label a sample of traffic, which
supplies ground truth, keeps the judge honest, and grows the dataset.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import mlflow
from mlflow.entities import AssessmentSourceType

from sia import config, data, gate, improvers, signals, tracking
from sia.agent import load_agents_md, register_candidate


@dataclass
class LoopConfig:
    improver: str = "gepa"          # gepa | analyst | openclaw  (offline when no API key)
    window_hours: float = 24.0
    min_traces: int = 8
    fail_threshold: float = 0.15    # act when more than this fraction of judged traces fail
    min_judge_agreement: float = 0.75
    rollback_margin: float = 0.10
    max_growth: float = 2.5         # candidate AGENTS.md may be at most this many times longer
    auto_promote: bool = True
    gepa_budget: int = 40


def _failed(trace) -> bool:
    """A trace fails if its human label, the quality judge, or a must-pass code check says so."""
    for a in trace.info.assessments:
        if a.feedback is None or a.valid is False:
            continue
        relevant = a.name == "quality" or a.name in gate.MUST_PASS
        if relevant and str(a.value).lower() in ("false", "no"):
            return True
    return False


def _has_verdict(trace) -> bool:
    return any(a.feedback is not None and (a.name == "quality" or a.name in gate.MUST_PASS)
               for a in trace.info.assessments)


def _last_cycle_start_ms() -> int | None:
    runs = mlflow.search_runs(filter_string="tags.`sia.cycle` = 'true'", order_by=["start_time DESC"],
                              max_results=1, output_format="list")
    return runs[0].info.start_time if runs else None


def _unscored(trace) -> bool:
    return not any(a.source.source_type in (AssessmentSourceType.CODE, AssessmentSourceType.LLM_JUDGE)
                   for a in trace.info.assessments)


def _quality_judge():
    if not config.HAS_LLM:
        return None
    from mlflow.genai.scorers import get_scorer

    try:
        return get_scorer(name="quality")
    except Exception:
        return signals.quality_judge()


def cycle(cfg: LoopConfig) -> str:
    experiment_id = tracking.setup()
    prod = load_agents_md("production")
    # Read what arrived since the previous cycle (bounded by the window), so each cycle judges
    # fresh traffic rather than re-reading traffic from before the last promotion.
    since = max(tracking.since_ms(cfg.window_hours), _last_cycle_start_ms() or 0)
    with mlflow.start_run(run_name=f"improvement-cycle {dt.datetime.now():%Y-%m-%d %H:%M:%S}") as run:
        mlflow.set_tag("sia.cycle", "true")
        mlflow.log_params({k: v for k, v in cfg.__dict__.items()})
        mlflow.set_tag("sia.production_version", prod.version)

        # ---- 1. SIGNAL ------------------------------------------------------------------
        live = f"tags.prompt_version = '{prod.version}' AND trace.timestamp_ms > {since}"
        traces = tracking.agent_traces(live)
        unscored = [t for t in traces if _unscored(t)]
        if unscored:
            # Online monitoring (01_monitor.py) scores LLM judges server-side; code scorers can't be
            # registered on OSS, so the loop runs them here, attached to the same traces.
            quality = _quality_judge()
            mlflow.genai.evaluate(data=unscored, scorers=signals.code_scorers() + ([quality] if quality else []))
            traces = tracking.agent_traces(live)
        judged = [t for t in traces if _has_verdict(t)]
        rate = sum(map(_failed, judged)) / len(judged) if judged else 0.0
        mlflow.log_metrics({"live_traces": len(traces), "judged_traces": len(judged), "live_failure_rate": rate})
        print(f"[signal] production v{prod.version}: {len(traces)} new traces, {len(judged)} with a verdict, "
              f"failure rate {rate:.0%}")

        # ---- rollback watch -------------------------------------------------------------
        tags = mlflow.MlflowClient().get_prompt_version(config.PROMPT_NAME, prod.version).tags
        if tags.get("promoted_by") == "loop" and len(traces) >= cfg.min_traces:
            before = float(tags.get("replaced_failure_rate", "1"))
            if rate > before + cfg.rollback_margin:
                restored = gate.rollback()
                mlflow.set_tag("sia.outcome", "rolled_back")
                return f"ROLLED BACK v{prod.version} -> v{restored}: live failure {rate:.0%} vs {before:.0%} before"

        if len(traces) < cfg.min_traces:
            mlflow.set_tag("sia.outcome", "waiting")
            return f"waiting for signal ({len(traces)}/{cfg.min_traces} traces)"
        if rate <= cfg.fail_threshold:
            mlflow.set_tag("sia.outcome", "healthy")
            return f"healthy ({rate:.0%} <= {cfg.fail_threshold:.0%})"

        # ---- 2. VERIFIER HEALTH ---------------------------------------------------------
        labeled = [t for t in traces if tracking.human_feedback(t) is not None]
        quality = _quality_judge()
        if quality is not None and len(labeled) >= 5:
            agreement = improvers.judge_agreement(quality, labeled)
            mlflow.log_metric("judge_agreement", agreement)
            print(f"[verify] judge agrees with humans on {agreement:.0%} of {len(labeled)} labels")
            if agreement < cfg.min_judge_agreement:
                all_labeled = [t for t in tracking.agent_traces() if tracking.human_feedback(t) is not None]
                if len(all_labeled) >= 10:
                    quality = improvers.align_judge(signals.quality_judge(), all_labeled)
                    agreement = improvers.judge_agreement(quality, labeled)
                    mlflow.log_metric("judge_agreement_realigned", agreement)
                if agreement < cfg.min_judge_agreement:
                    mlflow.set_tag("sia.outcome", "halted_untrusted_judge")
                    return (f"HALT: judge agreement {agreement:.0%} < {cfg.min_judge_agreement:.0%}. "
                            "A person needs to look at reviewer/judge disagreement before optimizing.")
                quality.register(name="quality")

        # ---- 3. STORE -------------------------------------------------------------------
        dataset = data.get_or_create_dataset(experiment_id)
        existing = {r["question"].strip().lower() for r in dataset.to_df()["inputs"]} if len(dataset.to_df()) else set()
        new = [r for r in data.dedupe([data.trace_to_record(t, "loop:human_fail")
                                       for t in data.labeled_failures(f"trace.timestamp_ms > {since}")])
               if r["inputs"]["question"].strip().lower() not in existing]
        if new:
            dataset.merge_records(new)
        # Failures only a scorer caught have no ground truth yet; they wait for a human label.
        unlabeled_fails = [t for t in traces if _failed(t) and tracking.human_feedback(t) is None]
        mlflow.log_metrics({"dataset_added": len(new), "dataset_size": len(dataset.to_df()),
                            "unlabeled_failures": len(unlabeled_fails)})
        print(f"[store]  +{len(new)} dataset rows ({data.growth_report(dataset)}); "
              f"{len(unlabeled_fails)} scorer-only failures await human labels")

        # ---- 4. UPDATE ------------------------------------------------------------------
        report = improvers.failure_report(traces)
        improver = cfg.improver if config.HAS_LLM else "offline"
        if improver == "gepa":
            scorers = signals.code_scorers() + signals.judge_scorers() + signals.theme_scorers() + [quality]
            res = improvers.gepa_optimize(data.rows(dataset), scorers, max_metric_calls=cfg.gepa_budget)
            text, why = res.optimized_prompts[0].template, f"GEPA {res.initial_eval_score:.2f}->{res.final_eval_score:.2f}"
        elif improver == "analyst":
            p = improvers.analyze(report, prod.template)
            text, why = p["agents_md"], p["change_summary"]
            for t in p["themes"]:
                if not t["is_reviewer_noise"]:
                    signals.register_theme(t["slug"], t["criterion"])
        elif improver == "openclaw":
            from sia.config import WORKSPACES_DIR

            p = improvers.openclaw_rewrite(report, prod.template, WORKSPACES_DIR / f"self-edit-{run.info.run_id[:8]}")
            text, why = p["agents_md"], "OpenClaw self-edit: " + p["change_summary"][:300]
        else:
            p = improvers.offline_analyze(report, prod.template)
            text, why = p["agents_md"], p["change_summary"]
        if text.strip() == prod.template.strip():
            mlflow.set_tag("sia.outcome", "no_change_proposed")
            return "improver proposed no change"
        candidate = register_candidate(text, f"loop ({improver}): {why}"[:900], author=f"loop:{improver}",
                                       level=5, cycle_run=run.info.run_id)
        mlflow.set_tag("sia.candidate_version", candidate)
        print(f"[update] {improver} proposed v{candidate}: {why[:120]}")

        # ---- 5. GATE --------------------------------------------------------------------
        if len(text) > cfg.max_growth * len(prod.template):
            mlflow.set_tag("sia.outcome", "rejected_bloat")
            return f"REJECTED v{candidate}: AGENTS.md grew {len(text) / len(prod.template):.1f}x"
        decision = gate.compare(candidate, "production", train=data.rows(dataset), holdout=data.holdout_rows(),
                                scorers=signals.all_scorers(quality))
        print(decision.table())
        if not decision.passed:
            mlflow.set_tag("sia.outcome", "rejected_by_gate")
            return f"REJECTED v{candidate}: " + "; ".join(decision.reasons)

        # ---- 6. SHIP --------------------------------------------------------------------
        if not cfg.auto_promote:
            mlflow.set_tag("sia.outcome", "awaiting_approval")
            return f"v{candidate} passed the gate; awaiting human approval (auto_promote off)"
        gate.promote(candidate, reason=f"loop cycle {run.info.run_id}: {why}")
        client = mlflow.MlflowClient()
        client.set_prompt_version_tag(config.PROMPT_NAME, candidate, "promoted_by", "loop")
        client.set_prompt_version_tag(config.PROMPT_NAME, candidate, "replaced_failure_rate", f"{rate:.4f}")
        mlflow.set_tag("sia.outcome", "promoted")
        return f"PROMOTED v{prod.version} -> v{candidate}"
