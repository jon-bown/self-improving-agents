"""Fast checks that need no API key and no OpenClaw: the scenario, reviewers, and Levels 0-2 offline."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sia.reviewers import review
from sia.scenario import load_cases, route_topic

ROOT = Path(__file__).resolve().parents[2]


def test_every_case_routes_to_its_topic():
    wrong = [c.id for c in load_cases() if route_topic(c.question) != c.topic]
    assert not wrong, wrong


def test_reviewers_disagree_on_style_but_not_facts():
    case = next(c for c in load_cases() if c.id == "refund_window")
    caveated = "Policies can vary, so double-check the Terms of Service. Refunds are available for 14 days."
    assert review("maria", case, caveated).passed
    assert not review("dev", case, caveated).passed          # dev fails disclaimers
    assert not review("maria", case, "You have 30 days.").passed


@pytest.fixture
def offline_env(tmp_path):
    # Don't leak the pytest/MLflow-plugin session (active run ids, test markers) into the scripts.
    base = {k: v for k, v in os.environ.items() if not k.startswith(("MLFLOW_", "PYTEST_", "SIA_"))}
    return {**base, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path}/mlflow.db", "SIA_BACKEND": "offline",
            "SIA_NO_LLM": "1", "SIA_WORKSPACES_DIR": str(tmp_path / "ws"),
            "SIA_OPENCLAW_STATE_DIR": str(tmp_path / "openclaw"), "OPENCLAW_BIN": "openclaw-not-needed"}


def run(script, env, *args):
    cwd = Path(env["SIA_WORKSPACES_DIR"]).parent  # run artifacts land in ./mlruns: keep them in tmp
    out = subprocess.run([sys.executable, str(ROOT / "levels" / script), *args], env=env, cwd=cwd,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_levels_0_to_2_close_the_loop(offline_env):
    run("level0_instrument/01_setup.py", offline_env)
    run("level0_instrument/02_first_traces.py", offline_env)
    run("level1_signal/01_human_feedback.py", offline_env)
    run("level2_manual_loop/01_build_dataset.py", offline_env)
    run("level2_manual_loop/02_write_fix.py", offline_env)
    out = run("level2_manual_loop/03_measure_and_promote.py", offline_env, "--promote")
    assert "Gate: PASS" in out and "production -> v2" in out
