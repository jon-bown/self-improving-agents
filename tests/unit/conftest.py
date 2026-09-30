import mlflow
import pytest


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    """Traces from unit tests go to a throwaway store, never ./mlflow.db."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
