import mlflow
import pytest


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    """Traces from unit tests go to a throwaway store, never ./mlflow.db."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    # Unit tests never call a live model: a fake key and no base URL make any accidental call fail loudly.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-unit-tests-never-call-the-api")
