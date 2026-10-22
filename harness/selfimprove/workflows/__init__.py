"""The scheduled jobs from Part 3, one module each. Every job is one MLflow run (tag si.job), so
the loop's own history (what it saw, what it tried, why it did or did not ship) is as easy to
inspect as the agent's traces.

    job        module      Part 3   what it automates                       stays with a person
    ---------  ----------  -------  --------------------------------------  -----------------------------
    score      score.py    sec. 3   judges label new production traces      golden cases, expectations
    triage     triage.py   sec. 2   read failures, group themes, propose    contradictions
    approve    approve.py  sec. 6   approver checks, merge into dataset     flagged tests, audit sample
    align      align.py    sec. 5   judge-vs-human agreement, re-align      the agreement threshold
    improve    improve.py  sec. 4   analyst / GEPA / MetaPrompt / agent     accepting skill/tool changes
    gate       ship.py     sec. 7   candidate vs production, CI + holdout   the promotion margin
    monitor    ship.py     sec. 8   live failure rate, automatic rollback   rollback review
    cycle      cycle.py    all      the whole loop, in order

`selfimprove <job>` runs one from the command line; .github/workflows/selfimprove-*.yml schedule them.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass, field

import mlflow


@dataclass
class Result:
    job: str
    outcome: str
    message: str
    metrics: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.job}] {self.outcome}: {self.message}"


@contextmanager
def job(name: str, **params):
    with mlflow.start_run(run_name=f"{name} {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                          nested=mlflow.active_run() is not None) as run:
        mlflow.set_tag("si.job", name)
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        yield run


def finish(result: Result) -> Result:
    mlflow.set_tag("si.outcome", result.outcome)
    mlflow.set_tag("si.message", result.message[:5000])
    numeric = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric)
    print(result)
    return result
