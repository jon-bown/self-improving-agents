"""The regression gate as a test suite (Part 2, sections 4 and 7): every dataset row is a test.

    SI_REGRESSION=1 pytest -p mlflow.pytest.plugin tests/regression -n 4
    SI_PROMPT_REF=7 ...      test a candidate version instead of production

Each pytest session is one MLflow run with a row per case, and a failing scorer prints its
rationale. CI runs this on every pull request that touches the agent, its instructions or its
criteria (.github/workflows/selfimprove-regression.yml). It calls your real agent and the judges,
so it is skipped unless SI_REGRESSION=1.
"""

from __future__ import annotations

import os

import mlflow
import pytest

if os.getenv("SI_REGRESSION") != "1":
    pytest.skip("set SI_REGRESSION=1 to run the regression suite (calls the agent and judges)",
                allow_module_level=True)

from selfimprove import agents, gate, signals, store, traces  # noqa: E402

traces.setup()
REF = os.getenv("SI_PROMPT_REF", "production")
ROWS = store.records()
SCORERS = signals.full()


def _id(r: dict) -> str:
    q = store.question_of(r)[:40]
    return f"{r['tags'].get('split', 'train')}:{r['tags'].get('theme') or r['tags'].get('source', '')}:{q}"


@mlflow.test
@pytest.mark.parametrize("row", ROWS, ids=[_id(r) for r in ROWS])
def test_dataset_row(row):
    result = mlflow.genai.evaluate(data=[{"inputs": row["inputs"], "expectations": row["expectations"]}],
                                   predict_fn=agents.predict_fn(REF), scorers=SCORERS)
    failures = gate.scorer_failures(result, SCORERS)
    assert not failures, "\n".join(failures)
