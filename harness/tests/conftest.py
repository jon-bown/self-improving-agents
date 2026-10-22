"""Exit like the CLI does when litellm is loaded.

LiteLLM's HTTP client finalizer can deadlock on an import lock at interpreter shutdown, which
would leave a finished pytest run hanging. Flush MLflow's trace queue, then exit with pytest's
status without running finalizers.
"""

import os
import sys

_status = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _status["code"] = int(exitstatus)


def pytest_unconfigure(config):
    if "litellm" not in sys.modules:
        return
    try:
        import mlflow

        mlflow.flush_trace_async_logging()
    except Exception:  # noqa: BLE001 -- best effort; never mask the test result
        pass
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter._tw.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_status["code"])
