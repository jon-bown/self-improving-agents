---
type: llm
focus: trace
weight: 1
---

The answer says the connector exports over HTTP, so MLFLOW_TRACKING_URI must point at a running mlflow server instead of sqlite:///mlflow.db.
