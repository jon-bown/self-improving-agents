"""connectors/connect.py and the repo's own OpenClaw profile wire the plugin in correctly."""

import importlib.util
import json
import sys

from sia.scenario import REPO_ROOT

spec = importlib.util.spec_from_file_location("connect", REPO_ROOT / "connectors" / "connect.py")
connect = importlib.util.module_from_spec(spec)
sys.modules["connect"] = connect
spec.loader.exec_module(connect)


def test_openclaw_patch_loads_plugin_and_opts_into_conversation_hooks():
    patch = connect.openclaw_patch("http://127.0.0.1:5000", "7", True)
    plugins = patch["plugins"]
    assert plugins["load"]["paths"] == [str(connect.OPENCLAW_PLUGIN)]
    entry = plugins["entries"]["mlflow-otel"]
    assert entry["hooks"]["allowConversationAccess"] is True
    assert entry["config"] == {"trackingUri": "http://127.0.0.1:5000", "experimentId": "7", "captureContent": True}
    assert "allow" not in plugins   # never create an allowlist: it would block every other plugin


def test_openclaw_patch_keeps_existing_paths_and_allowlist():
    patch = connect.openclaw_patch("http://x", "1", False, load_paths=["/other", str(connect.OPENCLAW_PLUGIN)],
                                   allow=["discord"])
    assert patch["plugins"]["load"]["paths"] == ["/other", str(connect.OPENCLAW_PLUGIN)]
    assert patch["plugins"]["allow"] == ["discord", "mlflow-otel"]


def test_hermes_env_file_is_updated_in_place(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=abc\nMLFLOW_TRACKING_URI=http://old\n")
    connect._set_env_lines(env, {"MLFLOW_TRACKING_URI": "http://new", "MLFLOW_EXPERIMENT_ID": "3"})
    assert env.read_text() == "ANTHROPIC_API_KEY=abc\nMLFLOW_TRACKING_URI=http://new\nMLFLOW_EXPERIMENT_ID=3\n"


def test_plugin_manifests_agree():
    manifest = json.loads((connect.OPENCLAW_PLUGIN / "openclaw.plugin.json").read_text())
    package = json.loads((connect.OPENCLAW_PLUGIN / "package.json").read_text())
    assert manifest["id"] == connect.PLUGIN_ID and manifest["activation"]["onStartup"] is True
    assert package["openclaw"]["extensions"] == ["./index.ts"]
    assert f"name: {connect.PLUGIN_ID}" in (connect.HERMES_PLUGIN / "plugin.yaml").read_text()


def test_repo_profile_enables_plugin(tmp_path, monkeypatch):
    from sia import config, openclaw_setup

    monkeypatch.setattr(config, "OPENCLAW_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "ws")
    cfg = openclaw_setup.write_config(mlflow_plugin=True, experiment_id="5")
    assert cfg["plugins"]["entries"]["mlflow-otel"]["config"]["experimentId"] == "5"
    assert cfg["plugins"]["load"]["paths"] == [str(openclaw_setup.OPENCLAW_PLUGIN_DIR)]
    assert "plugins" not in openclaw_setup.write_config()
