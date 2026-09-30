"""Simulated production: customers ask questions, reviewers label a sample of the answers."""

from __future__ import annotations

import random

import mlflow

from sia import tracking
from sia.agent import run_agent
from sia.reviewers import log_review
from sia.scenario import case_for_question, load_cases


def send(splits=("traffic",), n: int = 12, seed: int | None = None) -> list[str]:
    """Send n questions drawn from the given splits through production. Returns trace ids."""
    rng = random.Random(seed)
    pool = [c.question for s in splits for c in load_cases(s)]
    ids = []
    for q in (rng.choice(pool) for _ in range(n)):
        try:
            run_agent(q, source="production")
        except Exception as e:
            print(f"  agent error: {e}")
        ids.append(mlflow.get_last_active_trace_id())
    mlflow.flush_trace_async_logging()
    return [i for i in ids if i]


def label(trace_ids: list[str], rate: float = 0.5, seed: int | None = None) -> int:
    """Simulated reviewers label a fraction of traces (humans never see everything)."""
    rng = random.Random(seed)
    n = 0
    for tid in trace_ids:
        if rng.random() > rate:
            continue
        t = mlflow.get_trace(tid)
        case = case_for_question(tracking.question_of(t))
        if case is None:
            continue
        log_review(tid, case, tracking.response_of(t))
        n += 1
    return n
