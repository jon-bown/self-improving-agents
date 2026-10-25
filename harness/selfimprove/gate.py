"""Gate: score a candidate and production on the same data, then apply the promotion rules.

Every level of automation shares this step. What changes is who writes the candidate (a person,
an analyst, an optimizer, a coding agent) and who presses the button (a person, or the gate).

Rules (Part 2, section 4 and Part 3, sections 7-8):
  1. must-pass   every must-pass scorer is at 100% for the candidate, on both splits
  2. CI suite    the candidate is not worse than production on the train split
  3. holdout     the candidate beats production on the holdout by at least SI_PROMOTION_MARGIN.
                 An optimizer raises the train score by design; the holdout is the check that
                 matters, because nothing that wrote the candidate has seen it.
  4. progress    the candidate is better somewhere (a no-op never ships)
  5. bloat       the candidate is at most SI_MAX_PROMPT_GROWTH times longer than production
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import pandas as pd

from selfimprove import agents, config, instructions


def passed(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        return 1.0 if v.strip().lower() in ("yes", "true", "pass") else 0.0
    return float(bool(v)) if isinstance(v, bool) else float(v)


@dataclass
class Scorecard:
    label: str
    version: int
    run_id: str
    pass_rates: dict[str, float]
    rows: int

    @property
    def overall(self) -> float:
        return sum(self.pass_rates.values()) / max(len(self.pass_rates), 1)


def scorecard(ref, data: list[dict], scorers, label: str) -> Scorecard:
    version = int(instructions.load(ref).version)
    result = mlflow.genai.evaluate(data=data, predict_fn=agents.predict_fn(version), scorers=scorers)
    df = result.result_df
    rates = {}
    for s in scorers:
        col = f"{s.name}/value"
        if col in df:
            vals = [x for x in map(passed, df[col]) if x is not None]
            if vals:
                rates[s.name] = sum(vals) / len(vals)
    client = mlflow.MlflowClient()
    client.set_tag(result.run_id, "si.label", label)
    client.set_tag(result.run_id, "si.prompt_version", str(version))
    return Scorecard(label, version, result.run_id, rates, len(df))


@dataclass
class Decision:
    passed: bool
    candidate: int
    baseline: int
    reasons: list[str] = field(default_factory=list)
    cards: dict[str, Scorecard] = field(default_factory=dict)
    run_id: str | None = None

    def gain(self, split: str) -> float | None:
        c, b = self.cards.get(f"{split}/candidate"), self.cards.get(f"{split}/baseline")
        return None if c is None or b is None else c.overall - b.overall

    def table(self) -> str:
        names = sorted({n for c in self.cards.values() for n in c.pass_rates})
        lines = [f"{'scorer':<26}" + "".join(f"{c:>19}" for c in self.cards)]
        for n in names + ["OVERALL"]:
            vals = [c.overall if n == "OVERALL" else c.pass_rates.get(n) for c in self.cards.values()]
            lines.append(f"{n:<26}" + "".join(f"{'-' if v is None else f'{v:.0%}':>19}" for v in vals))
        return "\n".join(lines)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"gate {verdict}: v{self.candidate} vs v{self.baseline}" + (
            "" if self.passed else " -- " + "; ".join(self.reasons))


def compare(candidate, baseline="production", train: list[dict] | None = None,
            holdout: list[dict] | None = None, scorers=None, must_pass: tuple[str, ...] | None = None) -> Decision:
    from selfimprove import signals, store

    s = config.settings
    scorers = scorers if scorers is not None else signals.full()
    must_pass = must_pass if must_pass is not None else signals.must_pass()
    train = store.rows("train") if train is None else train
    holdout = store.rows("holdout") if holdout is None else holdout
    cand, base = instructions.load(candidate), instructions.load(baseline)
    d = Decision(passed=True, candidate=int(cand.version), baseline=int(base.version))

    with mlflow.start_run(run_name=f"gate: v{cand.version} vs v{base.version}",
                          nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"si.job": "gate", "si.candidate_version": cand.version, "si.baseline_version": base.version})
        for split, data in (("train", train), ("holdout", holdout)):
            if not data:
                continue
            for role, ref in (("baseline", base.version), ("candidate", cand.version)):
                with mlflow.start_run(run_name=f"{split}/{role}", nested=True):
                    d.cards[f"{split}/{role}"] = scorecard(ref, data, scorers, f"{split}/{role}")

        if not d.cards:
            d.passed, d.reasons = False, ["no dataset rows to evaluate on"]
        for split in ("train", "holdout"):
            card = d.cards.get(f"{split}/candidate")
            for name in must_pass if card else ():
                if card.pass_rates.get(name, 1.0) < 1.0:
                    d.passed = False
                    d.reasons.append(f"{split}: must-pass '{name}' at {card.pass_rates[name]:.0%}")
        train_gain, holdout_gain = d.gain("train"), d.gain("holdout")
        if train_gain is not None and train_gain < 0:
            d.passed = False
            d.reasons.append(f"train: worse than production ({train_gain:+.0%})")
        if holdout_gain is not None and holdout_gain < s.promotion_margin:
            d.passed = False
            d.reasons.append(f"holdout: gain {holdout_gain:+.0%} below margin {s.promotion_margin:+.0%}")
        if not any(g is not None and g > 0 for g in (train_gain, holdout_gain)):
            d.passed = False
            d.reasons.append("no improvement on either split")
        growth = len(cand.template) / max(len(base.template), 1)
        if growth > s.max_prompt_growth:
            d.passed = False
            d.reasons.append(f"bloat: candidate is {growth:.1f}x the length of production")

        for k, c in d.cards.items():
            mlflow.log_metric(f"{k.replace('/', '_')}_overall", c.overall)
        mlflow.log_metric("prompt_growth", growth)
        mlflow.set_tags({"si.gate_passed": str(d.passed), "si.gate_reasons": "; ".join(d.reasons)[:5000]})
        mlflow.log_text(d.table(), "gate_table.txt")
        mlflow.log_text(instructions.diff(base.version, cand.version), "candidate.diff")
        d.run_id = run.info.run_id
    instructions.set_tags(d.candidate, gate=("passed" if d.passed else "failed"), gate_run=d.run_id)
    return d


def scorer_failures(result, scorers) -> list[str]:
    """Like EvaluationResult.reason, but only for the given scorers.

    In MLflow 3.16 through 3.17.2 `result.passed` also treats expectation columns (expected_response/value)
    as assertions, so any row with an expected response "fails". This checks scorer columns only.
    """
    out = []
    for _, row in result.result_df.iterrows():
        for s in scorers:
            err = row.get(f"{s.name}/error_message")
            if isinstance(err, str) and err:
                out.append(f"{s.name}: error: {err[:200]}")
            elif passed(row.get(f"{s.name}/value")) == 0.0:
                out.append(f"{s.name}: {row.get(f'{s.name}/rationale')}")
    return out
