"""The whole loop, offline: no API keys, the example agent's rule-based stand-in, code checks only.

Every step runs through the CLI in a throwaway store, the way a scheduled workflow would run it.
This is what CI runs on every push (.github/workflows/selfimprove-offline.yml).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    state = tmp_path_factory.mktemp("selfimprove")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SI_", "MLFLOW_"))}
    env.update({
        "SI_ENV_FILE": str(HARNESS / ".env.example"),
        "SI_STATE_DIR": str(state),
        "MLFLOW_TRACKING_URI": f"sqlite:///{state}/mlflow.db",
        "SI_EXPERIMENT": "offline-test",
        "SI_NO_LLM": "1",
        "SI_EXAMPLE_OFFLINE": "1",
        "SI_TRIAGE_THRESHOLD": "1",
        "SI_MIN_TRACES": "4",
        "ANTHROPIC_API_KEY": "",
    })

    def run(*args: str) -> str:
        proc = subprocess.run([sys.executable, "-m", "selfimprove.cli", *args], cwd=HARNESS, env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"selfimprove {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr[-3000:]}"
        return proc.stdout

    def python(*args: str) -> str:
        proc = subprocess.run([sys.executable, *args], cwd=HARNESS, env=env, capture_output=True, text=True,
                              timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
        return proc.stdout

    run.python = python
    return run


def test_loop(cli):
    out = cli("init")
    assert "v1" in out and "+8 golden cases" in out

    out = cli("doctor", "--smoke", "How long can I wait to request a refund?")
    assert "adapter       loads" in out and "30 days" in out          # v1's stale quick fact

    cli("replay", "examples/questions.txt", "--repeat", "2")
    out = cli.python("-m", "examples.reviewer", "--all")
    assert "fail" in out

    assert "scored" in cli("score")
    out = cli("triage", "--force")
    assert "proposed" in out
    out = cli("approve", "--no-reviewer")
    assert "tests merged" in out

    out = cli("improve", "--from-file", "examples/fixes/v2_manual.md", "-m", "lead with the answer; look it up")
    assert "v2" in out
    out = cli("gate", "2", "--promote")
    assert "promoted" in out, out
    assert "no_stale_policy" in out                                 # the gate table

    cli("replay", "examples/questions.txt")
    out = cli("monitor")
    assert "v2" in out

    out = cli("report")
    assert "production       v2" in out and "triage runs" in out


def test_gate_rejects_regression(cli):
    # v1 again, as a "candidate": worse on everything, so the gate must refuse it
    out = cli("improve", "--from-file", "examples/instructions.md", "-m", "revert")
    version = out.split(" v", 1)[1].split()[0]
    out = cli("gate", version, "--promote")
    assert "rejected" in out and "must-pass 'no_stale_policy'" in out
    assert "production       v2" in cli("report")
