"""Gate: measure a candidate against production on the same data and decide whether to promote.

The gate is the one piece every level shares. What changes from level to level is only *who
produces the candidate* (a person, a coding agent, an optimizer, the agent itself) and *who
presses the button* (a person, or the gate itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import pandas as pd

from sia import config
from sia.agent import AgentSpec, as_spec, load_agents_md, pin, predict_fn_for

# Scorers that are non-negotiable: a candidate that fails any row here never ships.
MUST_PASS = ("no_stale_policy",)


def _passed(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        return 1.0 if v.lower() in ("yes", "true", "pass") else 0.0
    return float(bool(v))


@dataclass
class Scorecard:
    label: str
    spec: AgentSpec
    run_id: str
    pass_rates: dict[str, float]

    @property
    def overall(self) -> float:
        return sum(self.pass_rates.values()) / max(len(self.pass_rates), 1)


def scorecard(prompt_ref, data: list[dict], scorers, label: str) -> Scorecard:
    spec = pin(prompt_ref)  # concrete versions: "production" can't move under a running evaluation
    result = mlflow.genai.evaluate(data=data, predict_fn=predict_fn_for(spec), scorers=scorers)
    df = result.result_df
    rates = {}
    for s in scorers:
        col = f"{s.name}/value"
        if col in df:
            vals = [x for x in map(_passed, df[col]) if x is not None]
            if vals:
                rates[s.name] = sum(vals) / len(vals)
    client = mlflow.MlflowClient()
    client.set_tag(result.run_id, "sia.label", label)
    client.set_tag(result.run_id, "sia.prompt_version", str(spec.agents))
    client.set_tag(result.run_id, "sia.agent_spec", spec.label)
    return Scorecard(label, spec, result.run_id, rates)


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    cards: dict[str, Scorecard] = field(default_factory=dict)

    def table(self) -> str:
        names = sorted({n for c in self.cards.values() for n in c.pass_rates})
        cols = list(self.cards)
        lines = [f"{'scorer':<22}" + "".join(f"{c:>18}" for c in cols)]
        for n in names + ["OVERALL"]:
            vals = [c.overall if n == "OVERALL" else c.pass_rates.get(n) for c in self.cards.values()]
            lines.append(f"{n:<22}" + "".join(f"{'-' if v is None else f'{v:.0%}':>18}" for v in vals))
        return "\n".join(lines)


def compare(
    candidate,
    baseline="production",
    train: list[dict] | None = None,
    holdout: list[dict] | None = None,
    scorers=None,
    min_gain: float = 0.0,
    holdout_tolerance: float = 0.05,
) -> GateDecision:
    """Evaluate baseline and candidate on train + holdout and apply the promotion rules:

    1. every MUST_PASS scorer is 100% for the candidate on both splits,
    2. the candidate beats the baseline on train by more than min_gain,
    3. the candidate is not worse than the baseline on holdout by more than holdout_tolerance
       (an optimizer can overfit the train set; holdout is where that shows up).
    """
    from sia.signals import all_scorers

    scorers = scorers or all_scorers()
    d = GateDecision(passed=True)
    with mlflow.start_run(run_name=f"gate: {pin(candidate).label} vs {as_spec(baseline).label}",
                          nested=mlflow.active_run() is not None) as run:
        for split, data in (("train", train), ("holdout", holdout)):
            if not data:
                continue
            with mlflow.start_run(run_name=f"{split}/baseline", nested=True):
                d.cards[f"{split}/baseline"] = scorecard(baseline, data, scorers, f"{split}/baseline")
            with mlflow.start_run(run_name=f"{split}/candidate", nested=True):
                d.cards[f"{split}/candidate"] = scorecard(candidate, data, scorers, f"{split}/candidate")

        for split in ("train", "holdout"):
            cand = d.cards.get(f"{split}/candidate")
            if cand is None:
                continue
            for name in MUST_PASS:
                if cand.pass_rates.get(name, 1.0) < 1.0:
                    d.passed = False
                    d.reasons.append(f"{split}: must-pass scorer '{name}' at {cand.pass_rates[name]:.0%}")

        if "train/candidate" in d.cards:
            gain = d.cards["train/candidate"].overall - d.cards["train/baseline"].overall
            if gain <= min_gain:
                d.passed = False
                d.reasons.append(f"train: no improvement over baseline ({gain:+.0%})")
        if "holdout/candidate" in d.cards:
            drop = d.cards["holdout/baseline"].overall - d.cards["holdout/candidate"].overall
            if drop > holdout_tolerance:
                d.passed = False
                d.reasons.append(f"holdout: regressed by {drop:.0%} (tolerance {holdout_tolerance:.0%})")

        for k, c in d.cards.items():
            mlflow.log_metric(f"{k.replace('/', '_')}_overall", c.overall)
        mlflow.set_tags({"sia.gate_passed": str(d.passed), "sia.gate_reasons": "; ".join(d.reasons)[:5000]})
        mlflow.log_text(d.table(), "gate_table.txt")
        d.run_id = run.info.run_id
    return d


def promote(candidate, reason: str) -> dict:
    """Move @production for whatever the candidate changes: AGENTS.md and/or skills. The previous
    version keeps an @previous alias. Returns a receipt that rollback() can undo exactly."""
    spec = as_spec(candidate)
    receipt = {"reason": reason[:500], "agents": None, "skills": {}}
    if str(spec.agents) != "production":
        version = int(load_agents_md(spec.agents).version)
        previous = int(load_agents_md("production").version)
        if version != previous:
            mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="previous", version=previous)
            mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=version)
            mlflow.genai.set_prompt_version_tag(config.PROMPT_NAME, version, "promoted_because", reason[:500])
            receipt["agents"] = {"from": previous, "to": version}
    for name, version in spec.skills:
        full = f"{config.SKILL_PREFIX}{name}"
        current = pin("production").skills
        previous = dict(current).get(name)
        if version is None:
            mlflow.genai.delete_prompt_alias(full, alias="production")
        else:
            if previous is not None:
                mlflow.genai.set_prompt_alias(full, alias="previous", version=previous)
            mlflow.genai.set_prompt_alias(full, alias="production", version=version)
            mlflow.genai.set_prompt_version_tag(full, version, "promoted_because", reason[:500])
        receipt["skills"][name] = {"from": previous, "to": version}
    return receipt


def rollback(receipt: dict | None = None) -> str:
    """Undo a promotion. Without a receipt, moves AGENTS.md back to @previous (the original
    behavior). With one, restores exactly the AGENTS.md and skill versions it replaced."""
    if receipt is None:
        previous = load_agents_md("previous").version
        mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=previous)
        return f"AGENTS.md -> v{previous}"
    done = []
    if receipt.get("agents"):
        mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=receipt["agents"]["from"])
        done.append(f"AGENTS.md -> v{receipt['agents']['from']}")
    for name, change in receipt.get("skills", {}).items():
        full = f"{config.SKILL_PREFIX}{name}"
        if change["from"] is None:
            mlflow.genai.delete_prompt_alias(full, alias="production")
            done.append(f"skill {name} removed")
        else:
            mlflow.genai.set_prompt_alias(full, alias="production", version=change["from"])
            done.append(f"skill {name} -> v{change['from']}")
    return ", ".join(done) or "nothing to roll back"


def scorer_failures(result, scorers) -> list[str]:
    """Like EvaluationResult.reason, but only for the given scorers.

    In MLflow 3.16.1, result.passed also treats *expectation* columns (e.g.
    expected_response/value) as assertions, so any dataset row with an expected_response
    "fails". Checking the scorers' own columns avoids that.
    """
    out = []
    df = result.result_df
    for _, row in df.iterrows():
        for s in scorers:
            v = row.get(f"{s.name}/value")
            err = row.get(f"{s.name}/error_message")
            if isinstance(err, str) and err:
                out.append(f"{s.name}: error: {err[:200]}")
            elif _passed(v) == 0.0:
                out.append(f"{s.name}: {row.get(f'{s.name}/rationale')}")
    return out
