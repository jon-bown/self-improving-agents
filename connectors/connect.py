#!/usr/bin/env python3
"""Connect a personal agent (OpenClaw or Hermes) to a running MLflow server, in one command.

    python connectors/connect.py openclaw                      # uses `openclaw config file`
    python connectors/connect.py hermes                        # uses $HERMES_HOME or ~/.hermes
    python connectors/connect.py status

Options: --tracking-uri (default $MLFLOW_TRACKING_URI or http://127.0.0.1:5000), --experiment NAME
(created if missing; default "openclaw" / "hermes-agent"), --no-content (record structure, timing
and tokens, but no prompt or reply text), --dry-run (print what would change).

Standard library only, so it runs with any Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENCLAW_PLUGIN = HERE / "openclaw-mlflow"
HERMES_PLUGIN = HERE / "hermes-mlflow"
PLUGIN_ID = "mlflow-otel"


# ------------------------------------------------------------------------------------------ MLflow
def _http(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as res:
        return json.loads(res.read() or b"{}")


def check_server(uri: str) -> None:
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=5).read()
    except Exception as e:
        sys.exit(f"MLflow is not reachable at {uri} ({e}).\n"
                 f"Start one with:  mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000")


def experiment_id(uri: str, name: str, create: bool = True) -> str | None:
    base = f"{uri}/api/2.0/mlflow/experiments"
    try:
        return _http(f"{base}/get-by-name?experiment_name={urllib.parse.quote(name)}")["experiment"]["experiment_id"]
    except urllib.error.HTTPError as e:
        if e.code != 404 or not create:
            return None
    return _http(f"{base}/create", {"name": name})["experiment_id"]


# ---------------------------------------------------------------------------------------- OpenClaw
def _openclaw(*args: str, input: str | None = None) -> subprocess.CompletedProcess:
    exe = shutil.which(os.getenv("OPENCLAW_BIN", "openclaw"))
    if not exe:
        sys.exit("openclaw is not on PATH (npm i -g openclaw)")
    return subprocess.run([exe, *args], input=input, capture_output=True, text=True)


def _openclaw_get(path: str):
    out = _openclaw("config", "get", path, "--json")
    try:
        return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None
    except json.JSONDecodeError:
        return None


def openclaw_patch(tracking_uri: str, exp_id: str, capture_content: bool, plugin_dir: Path = OPENCLAW_PLUGIN,
                   load_paths: list | None = None, allow: list | None = None) -> dict:
    """The config patch that loads the plugin from this repo and points it at MLflow."""
    paths = [p for p in (load_paths or []) if Path(p).resolve() != plugin_dir.resolve()] + [str(plugin_dir)]
    plugins: dict = {
        "load": {"paths": paths},
        "entries": {PLUGIN_ID: {
            "enabled": True,
            # llm_input / llm_output / agent_end carry conversation content; OpenClaw requires an
            # explicit opt-in before a non-bundled plugin may observe them.
            "hooks": {"allowConversationAccess": True},
            "config": {"trackingUri": tracking_uri, "experimentId": exp_id, "captureContent": capture_content},
        }},
    }
    if allow is not None:   # only extend an allowlist that exists; creating one would block other plugins
        plugins["allow"] = sorted(set(allow) | {PLUGIN_ID})
    return {"plugins": plugins}


def connect_openclaw(a) -> None:
    exp = experiment_id(a.tracking_uri, a.experiment or "openclaw")
    cfg_file = _openclaw("config", "file").stdout.strip().splitlines()[-1:] or ["?"]
    patch = openclaw_patch(a.tracking_uri, exp, not a.no_content,
                           load_paths=_openclaw_get("plugins.load.paths"), allow=_openclaw_get("plugins.allow"))
    print(f"OpenClaw config: {cfg_file[0]}\nexperiment: {a.experiment or 'openclaw'} (id {exp})")
    if a.dry_run:
        print(json.dumps(patch, indent=2))
        return
    out = _openclaw("config", "patch", "--stdin", input=json.dumps(patch))
    if out.returncode != 0:
        sys.exit(f"openclaw config patch failed:\n{out.stderr or out.stdout}")
    valid = _openclaw("config", "validate")
    print("config:", "valid" if valid.returncode == 0 else f"INVALID\n{valid.stdout}{valid.stderr}")
    print(f"\nDone. Every `openclaw agent` run now appears in MLflow:\n  {a.tracking_uri}/#/experiments/{exp}/traces\n"
          "A running Gateway picks this up on its next config reload (or restart it).")


# ------------------------------------------------------------------------------------------ Hermes
def _set_env_lines(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in values:
            lines[i] = f"{key}={values[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in values.items() if k not in seen]
    path.write_text("\n".join(lines) + "\n")


def connect_hermes(a) -> None:
    home = Path(a.hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    name = a.experiment or "hermes-agent"
    exp = experiment_id(a.tracking_uri, name)
    link = home / "plugins" / PLUGIN_ID
    env = {"MLFLOW_TRACKING_URI": a.tracking_uri, "MLFLOW_EXPERIMENT_ID": exp,
           "MLFLOW_OTEL_CAPTURE_CONTENT": "false" if a.no_content else "true"}
    print(f"Hermes home: {home}\nexperiment: {name} (id {exp})")
    if a.dry_run:
        print(f"would link {link} -> {HERMES_PLUGIN}\nwould set in {home / '.env'}: {env}\nwould run: hermes plugins enable {PLUGIN_ID}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or not link.exists():
        link.unlink(missing_ok=True)
        link.symlink_to(HERMES_PLUGIN, target_is_directory=True)
    elif link.resolve() != HERMES_PLUGIN.resolve():
        sys.exit(f"{link} already exists and is not this plugin; remove it first")
    _set_env_lines(home / ".env", env)
    hermes = a.hermes_bin or shutil.which("hermes") or str(Path(sys.executable).parent / "hermes")
    # stdin closed: `enable` then answers "no" to its tool-override prompt (a tracer replaces no tools).
    out = subprocess.run([hermes, "plugins", "enable", PLUGIN_ID], capture_output=True, text=True, timeout=120,
                         stdin=subprocess.DEVNULL, env={**os.environ, "HERMES_HOME": str(home)})
    if out.returncode != 0:
        sys.exit(f"`hermes plugins enable {PLUGIN_ID}` failed:\n{out.stderr or out.stdout}")
    print(f"\nDone. Every Hermes turn now appears in MLflow:\n  {a.tracking_uri}/#/experiments/{exp}/traces\n"
          "Restart a running Hermes gateway to load the plugin.")


# ------------------------------------------------------------------------------------------ status
def status(a) -> None:
    print(f"MLflow {a.tracking_uri}: up")
    for name in filter(None, [a.experiment, "openclaw", "hermes-agent"]):
        exp = experiment_id(a.tracking_uri, name, create=False)
        if exp:
            n = len(_http(f"{a.tracking_uri}/api/2.0/mlflow/traces?experiment_ids={exp}&max_results=100").get("traces", []))
            print(f"  experiment {name!r} (id {exp}): {n}{'+' if n == 100 else ''} traces")
    if shutil.which(os.getenv("OPENCLAW_BIN", "openclaw")):
        entry = _openclaw_get(f"plugins.entries.{PLUGIN_ID}")
        print("OpenClaw:", f"connected -> {entry.get('config', {}).get('trackingUri')}" if entry else "not connected")
    home = Path(a.hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    print("Hermes:", f"plugin linked in {home}" if (home / "plugins" / PLUGIN_ID).exists() else "not connected")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", choices=["openclaw", "hermes", "status"])
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", help="experiment name (created if missing)")
    p.add_argument("--no-content", action="store_true", help="don't record prompt, reply or tool text")
    p.add_argument("--hermes-home", help="Hermes home directory (default $HERMES_HOME or ~/.hermes)")
    p.add_argument("--hermes-bin", help="path to the hermes executable")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.tracking_uri = a.tracking_uri.rstrip("/")
    if not a.tracking_uri.startswith(("http://", "https://")):
        sys.exit(f"--tracking-uri must be an MLflow server URL (http://...), got {a.tracking_uri!r}")
    check_server(a.tracking_uri)
    {"openclaw": connect_openclaw, "hermes": connect_hermes, "status": status}[a.agent](a)


if __name__ == "__main__":
    main()
