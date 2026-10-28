"""Deployed mode: an OpenClaw people talk to, taking part in its own loop (the
self-improving-with-mlflow skill). These are the operator's commands; the agent never runs them.

    self lessons   build the lessons pack from human-confirmed train-split failures, as a candidate
    self sync      write production's AGENTS.md (and LESSONS.md) into the live workspace
    self setup     write the deployed profile: read + exec only, exec allowlisted to one wrapper
                   that runs `selfimprove self status|note|recall|propose`, Skill Workshop off

The lessons pack is a block inside the registered instructions (see selfimprove.lessons), so it
goes through the same gate and rolls back with the same alias move. `self sync` logs which
version it wrote when: connector traces carry no prompt version, so `self status` credits
deployed traffic to the version that was live when the trace started (design D5).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow

from selfimprove import config, instructions, lessons, models, store, traces
from selfimprove.workflows import Result, finish, job

FAILURE_SOURCES = ("human_fail", "queue")   # dataset rows that reproduce a failure a person confirmed
AGENT_ACTIONS = ("status", "note", "recall", "propose")
SKILL = "self-improving-with-mlflow"
# harness/selfimprove/workflows/deployed.py -> <repo>/skills/self-improving-with-mlflow
SKILL_DIR = Path(__file__).resolve().parents[3] / "skills" / SKILL
LIVE_PORT = 19556   # never your gateway's 18789, nor the harness profile's 19555


# --------------------------------------------------------------------------------------------
# lessons
# --------------------------------------------------------------------------------------------
LESSONS_SYSTEM = """\
You write the lessons pack for an AI agent: one short lesson per theme of past failures that a
person confirmed. Each lesson is ONE imperative sentence of at most 30 words that tells the agent
what to do differently next time.

* Point at the source to check (a file in the agent's workspace) instead of stating the answer:
  facts change, and the agent must look them up.
* Describe the behavior, not the test cases. Never quote a question or an expected answer.
* No hedging, no "always be careful": a lesson the agent can't act on is noise."""

LESSONS_TOOL = models.tool("submit_lessons", "Submit one lesson per theme.", {
    "type": "object",
    "properties": {"lessons": {"type": "array", "items": {"type": "object", "properties": {
        "theme": {"type": "string"}, "lesson": {"type": "string"}}, "required": ["theme", "lesson"]}}},
    "required": ["lessons"],
})


def _workspace_files() -> list[str]:
    src = config.settings.workspace_dir
    if not src or not src.is_dir():
        return []
    return sorted(str(p.relative_to(src.parent)) for p in src.rglob("*") if p.is_file())[:50]


def _fallback_lesson(theme: str, recs: list[dict]) -> str:
    reason = next((r["tags"]["reason"] for r in recs if r["tags"].get("reason")), "")
    reason = reason.split(": ", 1)[-1].strip().rstrip(".")
    topic = theme.replace("_", " ")
    return (f"{reason[:160]}. " if reason else "") + f"Check your sources before answering questions about {topic}."


def build_lessons(use_llm: bool = True) -> tuple[list[dict], int]:
    """(pack, confirmed failures it came from). Train split only: the holdout stays with the gate."""
    s = config.settings
    hidden = store.holdout_questions()
    recs = [r for r in store.records()
            if r["tags"].get("source") in FAILURE_SOURCES and r["tags"].get("split", "train") == "train"
            and not store.is_holdout(store.question_of(r), hidden)]
    by_theme: dict[str, list[dict]] = {}
    for r in recs:
        by_theme.setdefault(r["tags"].get("theme") or traces.QUALITY, []).append(r)
    themes = sorted(by_theme, key=lambda t: (-len(by_theme[t]), t))[: s.lessons_max]
    if not themes:
        return [], 0

    written: dict[str, str] = {}
    if use_llm and s.has_llm:
        brief = [{"theme": t, "failures": [{"input": store.question_of(r)[:300],
                                            "expected": str(r["expectations"].get(traces.EXPECTED, ""))[:300],
                                            "why_it_failed": r["tags"].get("reason", "")[:300]}
                                           for r in by_theme[t][:5]]} for t in themes]
        user = (f"Files in the agent's workspace: {_workspace_files() or '(none listed)'}\n\n"
                f"Themes of confirmed failures:\n{json.dumps(brief, indent=1)}")
        out = models.structured(s.judge_model, LESSONS_SYSTEM, user, LESSONS_TOOL, max_tokens=4000)
        written = {les["theme"]: les["lesson"] for les in out.get("lessons", []) if les.get("lesson")}

    pack = []
    for t in themes:
        group = by_theme[t]
        pack.append({"id": lessons.lesson_id(t),
                     "lesson": written.get(t) or _fallback_lesson(t, group), "theme": t, "evidence": len(group),
                     "keys": lessons.keywords([store.question_of(r) for r in group] + [t.replace("_", " ")]),
                     "traces": [r["tags"]["source_trace"] for r in group if r["tags"].get("source_trace")][:5]})
    return pack, len(recs)


def lessons_job(gate: bool = False, use_llm: bool = True) -> Result:
    from selfimprove.workflows.ship import run_gate

    with job("lessons") as run:
        prod = instructions.load("production")
        pack, n = build_lessons(use_llm=use_llm)
        text = lessons.replace(prod.template, pack)
        mlflow.log_dict({"lessons": pack}, "lessons.json")
        if text.strip() == prod.template.strip():
            return finish(Result("lessons", "no_change", f"{len(pack)} lessons, same as production v{prod.version}",
                                 {"lessons": len(pack)}))
        version = instructions.register_candidate(
            text, f"lessons: {len(pack)} lessons from {n} confirmed failures", author="selfimprove:lessons",
            source_run=run.info.run_id, base_version=prod.version)
        result = Result("lessons", "candidate", f"v{version}: {len(pack)} lessons from {n} confirmed failures "
                        f"(train split); the gate decides", {"lessons": len(pack), "candidate_version": version},
                        {"version": version})
        finish(result)
    return run_gate(version) if gate else result


# --------------------------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------------------------
def _write(path: Path, text: str | None) -> bool:
    if text is None:
        if path.is_file():
            path.unlink()
            return True
        return False
    if path.is_file() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def sync(workspace: str | None = None) -> Result:
    s = config.settings
    ws = Path(workspace).expanduser().resolve() if workspace else s.self_workspace
    if ws is None:
        raise ValueError("give --workspace or set SI_SELF_WORKSPACE")
    with job("sync", workspace=ws):
        prod = instructions.load("production")
        version = int(prod.version)
        ws.mkdir(parents=True, exist_ok=True)
        agents_md, pack = lessons.split(prod.template)
        changed = _write(ws / "AGENTS.md", agents_md) | _write(ws / lessons.FILENAME, pack)
        from selfimprove import agentside

        log = [e for e in agentside.synced() if e.get("workspace") == str(ws)]
        if not log or log[-1]["version"] != version:
            entry = {"version": version, "written_at": traces.now_ms(), "workspace": str(ws)}
            with (s.state_dir / "self_sync.jsonl").open("a") as f:
                f.write(json.dumps(entry) + "\n")
            instructions.set_tags(version, synced_at=entry["written_at"])
        what = "wrote" if changed else "already had"
        return finish(Result("sync", "synced" if changed else "current",
                             f"{ws} {what} v{version} ({len(lessons.parse(prod.template))} lessons)",
                             {"production_version": version}))


# --------------------------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------------------------
WRAPPER = """#!/bin/sh
# Written by `selfimprove self setup`. The deployed agent's only allowlisted command: it runs the
# agent-facing `selfimprove self` actions and refuses everything else (gate, promote, queue, ...).
case "$1 $2" in
  {cases}) ;;
  *) echo '{{"ok": false, "refused": "not available to you: only selfimprove self {actions}"}}'; exit 0 ;;
esac
{env} exec {binary} "$@"
"""


def _wrapper(profile: Path, ws: Path) -> Path:
    s = config.settings
    venv_bin = Path(sys.executable).parent / "selfimprove"
    binary = shlex.quote(str(venv_bin)) if venv_bin.is_file() else f"{shlex.quote(sys.executable)} -m selfimprove.cli"
    env = {"SI_ENV_FILE": config.ENV_FILE, "SI_STATE_DIR": s.state_dir.resolve(), "SI_SELF_WORKSPACE": ws,
           "MLFLOW_TRACKING_URI": s.tracking_uri, "SI_EXPERIMENT": s.experiment, "SI_PROMPT_NAME": s.prompt_name,
           "SI_DATASET_NAME": s.dataset_name, "SI_PRODUCTION_FILTER": s.production_filter}
    path = profile / "bin" / "selfimprove"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WRAPPER.format(
        cases="|".join(f'"self {a}"' for a in AGENT_ACTIONS), actions="|".join(AGENT_ACTIONS), binary=binary,
        env=" ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v)))
    path.chmod(0o755)
    return path


def _profile_config(profile: Path, ws: Path, bin_dir: Path) -> Path:
    cfg = {
        "agents": {"defaults": {"model": {"primary": models.openclaw_model(config.settings.agent_model)},
                                "skipBootstrap": True, "workspace": str(ws)}},
        # read + exec, nothing that writes files: self-edits go through `self propose`, not `write`.
        "tools": {"allow": ["read", "exec"], "exec": {"mode": "allowlist", "pathPrepend": [str(bin_dir)]}},
        # OpenClaw's own ungated learning, off: every change to the agent goes through the gate (design D2).
        "skills": {"workshop": {"autonomous": {"mode": "off"}}},
        "gateway": {"mode": "local", "port": LIVE_PORT, "bind": "loopback"},
    }
    path = profile / "openclaw.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**existing, **cfg}, indent=2))
    return path


def setup(workspace: str, profile: str | None = None) -> Result:
    s = config.settings
    ws = Path(workspace).expanduser().resolve()
    prof = Path(profile).expanduser().resolve() if profile else (s.state_dir / "openclaw-live").resolve()
    with job("setup", workspace=ws, profile=prof):
        ws.mkdir(parents=True, exist_ok=True)
        prof.mkdir(parents=True, exist_ok=True)
        wrapper = _wrapper(prof, ws)
        cfg = _profile_config(prof, ws, wrapper.parent)
        shutil.copytree(SKILL_DIR, ws / "skills" / SKILL, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("evals", ".DS_Store"))
        oc_env = {"OPENCLAW_STATE_DIR": str(prof), "OPENCLAW_CONFIG_PATH": str(cfg)}
        (prof / "env.sh").write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in
                                             {**oc_env, "SI_SELF_WORKSPACE": str(ws)}.items()))
        notes = []
        if shutil.which(s.openclaw_bin):
            import os

            env = {**os.environ, **oc_env}
            add = subprocess.run([s.openclaw_bin, "approvals", "allowlist", "add", "--agent", "main", str(wrapper)],
                                 env=env, capture_output=True, text=True)
            notes.append("exec allowlist: " + ("the wrapper only" if add.returncode == 0 else
                                               f"FAILED ({add.stderr.strip()[-200:]})"))
            if s.openclaw_mlflow:
                from selfimprove.agents.openclaw import PLUGIN_DIR

                exp = mlflow.get_experiment_by_name(s.experiment)
                subprocess.run([str(PLUGIN_DIR / "install.sh"), "--tracking-uri", s.tracking_uri,
                                "--experiment-id", exp.experiment_id],
                               env={**env, "OPENCLAW_BIN": s.openclaw_bin}, check=True)
                notes.append("connector installed")
            valid = subprocess.run([s.openclaw_bin, "config", "validate"], env=env, capture_output=True, text=True)
            notes.append("config " + ("valid" if valid.returncode == 0 else f"INVALID: {valid.stderr[-300:]}"))
        else:
            notes.append(f"{s.openclaw_bin} not on PATH: files written, allowlist not set")
        synced = sync(str(ws)) if instructions.exists() else None
        if not s.production_filter:
            notes.append("set SI_PRODUCTION_FILTER so connector traces count as production traffic")
        return finish(Result("setup", "ready", f"profile {prof}, workspace {ws}; " + "; ".join(notes) +
                             (f"; {synced.message}" if synced else "; nothing to sync yet (run `selfimprove init`)") +
                             f". Use it with: source {prof / 'env.sh'}"))
