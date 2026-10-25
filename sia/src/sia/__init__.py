"""sia: a small toolkit for building a self-improving OpenClaw agent with MLflow."""

import os

# MLflow's usage telemetry (seen on 3.16.1) can deadlock on a lazy import when evaluate() runs scorers in
# worker threads. Opt out before mlflow is imported. Set MLFLOW_DISABLE_TELEMETRY=false to undo.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
