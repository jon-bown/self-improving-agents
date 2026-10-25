# Troubleshooting

Problems met while building and running the loop, with their fixes. OpenClaw-specific ones are in
`openclaw.md`.

## Installing

| Symptom | Cause and fix |
|---|---|
| `import litellm` fails with `No module named 'mcp'` | litellm 1.106.x imports `mcp` without depending on it. Pin `litellm>=1.75,<1.106` (the harness and `sia/` both do). |
| `cryptography` fails to load: `symbol not found ... _OSSL_get_max_threads` | the cryptography 50.x wheel and this Mac's OpenSSL. Install `cryptography==46.0.7` in that venv. It's machine-specific, so don't pin it in the project. |
| the harness CLI hangs at exit after a run | litellm's HTTP client finalizer can deadlock at interpreter shutdown. The CLI flushes traces and exits hard; do the same in your own scripts (`mlflow.flush_trace_async_logging()`, then `os._exit`). |

## MLflow

Checked on 3.17.2 unless a row says otherwise.

| Symptom | Cause and fix |
|---|---|
| `assert result.passed` fails on rows that look fine | `EvaluationResult.passed` treats every `*/value` column as an assertion, including `expected_response/value`. Check only the scorers' own columns (`selfimprove.gate.scorer_failures`). |
| reading traces from a `sqlite:///` URI fails with "When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI" | the store was written through a server with `--artifacts-destination`. Serve it (`mlflow server --backend-store-uri sqlite:///... --artifacts-destination ...`) and read over HTTP. |
| the OpenClaw connector exports nothing | it exports over HTTP; `MLFLOW_TRACKING_URI` must be a server, not `sqlite:///`. |
| the Sessions page says the view has moved | in 3.17 it's the Traces tab, "Group by session" (`traces?groupBy=session`). |
| boolean feedback filters match nothing | feedback is stored as text: filter `feedback.quality = 'false'`, not `= False`. |
| `evaluate()` hangs with scorers in worker threads | usage telemetry can deadlock on a lazy import (seen on 3.16.1). Set `MLFLOW_DISABLE_TELEMETRY=true` before importing mlflow. |
| a pytest suite with the MLflow plugin hangs | seen on 3.16.1: register the plugin on the command line (`pytest -p mlflow.pytest.plugin tests/regression`), not in a root `conftest.py`. |

## Long runs

| Symptom | Cause and fix |
|---|---|
| a run looks hung: 0% CPU, the clock jumps | the laptop went to sleep. Keep it awake for long runs (`caffeinate -i` on macOS). |
| a `cycle` was interrupted | mark its `RUNNING` runs `KILLED`, then run the next `cycle` against the stored state; don't replay traffic or re-apply changes (`workflows.md`). |
| a gate takes much longer than expected | it runs every dataset row for production and the candidate, each an agent call. Budget the agent's latency times twice the dataset. |

## Agents

| Symptom | Cause and fix |
|---|---|
| an agent bills the user's subscription instead of the API key | it fell back to a saved login (Hermes 0.19 does this). Pass only the key from `.env`, and unset `ANTHROPIC_BASE_URL` if something set it. |
| the agent sees files it shouldn't (answer keys, other cases) | fence it to its workspace: an isolated state dir, and for Hermes a workspace that is its own git root. |
| a gate fails because of the verifier, not the agent | read the failing row. A regex that matches refusals, or a judge that can't see the policy, is a verifier bug: fix it as a person's decision and re-gate (`verifiers.md`). |
