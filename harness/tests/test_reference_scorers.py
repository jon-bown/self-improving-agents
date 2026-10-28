"""The key_facts reference scorer reaches the gate's scorecard on rows that have facts to check."""

from __future__ import annotations

import mlflow

from selfimprove import gate, signals


def test_key_facts_value_survives_evaluate(tmp_path):
    # A feedback named like an expectation ("key_facts") comes back as NaN in result_df, and the
    # gate skips NaN: the scorer would pass every row that has nothing to check and ignore the rest.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("reference-scorers")
    data = [{"inputs": {"question": "How long do refunds take?"},
             "expectations": {"expected_response": "14 days", "key_facts": ["14[- ]day"]}}]
    result = mlflow.genai.evaluate(data=data, predict_fn=lambda question: "30 days",
                                   scorers=[signals.key_facts])
    values = result.result_df[f"{signals.key_facts.name}/value"].map(gate.passed).tolist()
    assert values == [0.0]
