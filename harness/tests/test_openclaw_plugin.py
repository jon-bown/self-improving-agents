"""The OpenClaw adapter's MLflow plugin wiring: no OpenClaw, no network.

  * rewriting the isolated profile keeps what `openclaw plugins install` added to it
  * the plugin is off unless SI_OPENCLAW_MLFLOW=1, and skipped for a non-HTTP tracking store
  * doctor installs it once, with the experiment id, into the isolated profile only
  * each call runs in its own workspace: the profile names none, so OPENCLAW_WORKSPACE_DIR applies
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from selfimprove import config
from selfimprove.agents import openclaw


@pytest.fixture
def settings(tmp_path, monkeypatch):
    def use(**overrides):
        s = dataclasses.replace(config.Settings(), state_dir=tmp_path, **overrides)
        monkeypatch.setattr(config, "settings", s)
        return s
    return use


def test_write_config_keeps_plugin_entries(settings):
    settings()
    path = openclaw.state_dir() / "openclaw.json"
    path.parent.mkdir(parents=True)
    plugins = {"entries": {"mlflow-openclaw": {"enabled": True, "hooks": {"allowConversationAccess": True}}}}
    path.write_text(json.dumps({"plugins": plugins, "gateway": {"port": 1}}))

    openclaw.write_config()

    cfg = json.loads(path.read_text())
    assert cfg["plugins"] == plugins
    assert cfg["gateway"]["port"] == 19555  # ours win for the keys the harness owns


def test_plugin_off_by_default(settings):
    settings(openclaw_mlflow=False)
    assert openclaw.plugin_check("1").startswith("off")


def test_plugin_skipped_for_sqlite_store(settings):
    settings(openclaw_mlflow=True, tracking_uri="sqlite:////tmp/mlflow.db")
    assert openclaw.plugin_check("1").startswith("skipped")


def test_plugin_installed_once_into_isolated_profile(settings, monkeypatch):
    settings(openclaw_mlflow=True, tracking_uri="http://127.0.0.1:5000")
    calls = []

    def fake_run(cmd, env, check):
        calls.append((cmd, env))
        index = openclaw.state_dir() / "extensions" / "mlflow-openclaw" / "dist" / "index.js"
        index.parent.mkdir(parents=True)
        index.write_text("")

    monkeypatch.setattr(openclaw.subprocess, "run", fake_run)
    assert openclaw.plugin_check("7").startswith("installed")
    assert openclaw.plugin_check("7").startswith("installed")

    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[1:] == ["--tracking-uri", "http://127.0.0.1:5000", "--experiment-id", "7"]
    assert env["OPENCLAW_STATE_DIR"] == str(openclaw.state_dir())


def test_profile_leaves_workspace_to_env(settings):
    # OpenClaw 2026.10 prefers a configured agents.defaults.workspace over OPENCLAW_WORKSPACE_DIR.
    settings()
    cfg = json.loads(open(openclaw.write_config()).read())
    assert "workspace" not in cfg["agents"]["defaults"]
    assert "list" not in cfg["agents"]


def test_call_uses_its_workspace_and_rewrites_stale_profile(settings, monkeypatch, tmp_path):
    settings()
    path = openclaw.state_dir() / "openclaw.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"agents": {"defaults": {"workspace": "/stale"}}}))
    calls = []

    def fake_run(cmd, env, **kwargs):
        calls.append(env)
        return openclaw.subprocess.CompletedProcess(cmd, 0, json.dumps({"payloads": [{"text": "30 days"}]}), "")

    monkeypatch.setattr(openclaw.subprocess, "run", fake_run)
    ws = tmp_path / "workspaces" / "abc"
    assert openclaw.call_openclaw("refund window?", "be Acme", ws) == "30 days"

    assert calls[0]["OPENCLAW_WORKSPACE_DIR"] == str(ws)
    assert "workspace" not in json.loads(path.read_text())["agents"]["defaults"]
