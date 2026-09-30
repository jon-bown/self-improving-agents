"""The regression gate: every row of the curated dataset is a behavioral test.

A test here is a dataset row + scorers + the agent. As the dataset grows from feedback
(Levels 2-5), this suite grows with it -- no one writes these tests by hand.

    pytest -p mlflow.pytest.plugin tests/agent                          # against production
    SIA_PROMPT_REF=5 pytest -p mlflow.pytest.plugin tests/agent -n 4    # against candidate v5

The MLflow pytest plugin (required by @mlflow.test) is enabled with -p rather than in a root
conftest.py, so it only wraps this suite.
"""

import os

import mlflow
import pytest
from mlflow.genai.datasets import get_dataset

from sia import config, data, gate, signals, tracking
from sia.agent import predict_fn_for

tracking.setup()
PROMPT_REF = os.getenv("SIA_PROMPT_REF", "production")
SCORERS = signals.all_scorers()

try:
    ROWS = data.rows(get_dataset(name=config.DATASET_NAME))
except Exception:
    ROWS = []


@mlflow.test
@pytest.mark.skipif(not ROWS, reason="no regression dataset yet (run Level 2 first)")
@pytest.mark.parametrize("row", ROWS, ids=[r["inputs"]["question"][:40] for r in ROWS])
def test_regression_dataset(row):
    result = mlflow.genai.evaluate(data=[row], predict_fn=predict_fn_for(PROMPT_REF), scorers=SCORERS)
    failures = gate.scorer_failures(result, SCORERS)  # not result.passed: see gate.scorer_failures
    assert not failures, "\n".join(failures)


@mlflow.test
@pytest.mark.parametrize("row", data.holdout_rows(), ids=lambda r: r["inputs"]["question"][:40])
def test_holdout(row):
    result = mlflow.genai.evaluate(data=[row], predict_fn=predict_fn_for(PROMPT_REF), scorers=SCORERS)
    failures = gate.scorer_failures(result, SCORERS)  # not result.passed: see gate.scorer_failures
    assert not failures, "\n".join(failures)
