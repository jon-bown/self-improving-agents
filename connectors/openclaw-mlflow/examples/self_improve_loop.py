"""The automatic half of the live demo: OpenClaw proposes a fix to itself, and a gate decides.

    python self_improve_loop.py improve --demo ~/odsc-5min     # failing traces -> OpenClaw -> candidate version
    python self_improve_loop.py gate --demo ~/odsc-5min        # test the candidate; promote only if it passes

`improve` collects the traces that `score` failed, with the judge's reasons, and hands them to
OpenClaw together with the AGENTS.md in production. OpenClaw replies with a corrected AGENTS.md,
which is registered in the Prompt Registry as a candidate. The alias doesn't move: whoever writes
a fix doesn't get to ship it.

`gate` runs the candidate through OpenClaw on the questions that failed, plus a question the
improver never saw (the refund window, the demo's holdout), scores those runs with the same two
checks, and moves `@production` to the candidate only if every check passes. Otherwise production
stays where it is, and the reasons are recorded on the candidate.

Set MLFLOW_TRACKING_URI first (demo.sh does). Needs the harness venv's mlflow and an Anthropic key.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from self_improvement_demo import grounded_in_docs, quiet_evaluate, read_knowledge  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)

CHECKS = ("read_knowledge", "grounded_in_docs")
HOLDOUT = "What's your refund window?"  # never shown to the improver; only the gate asks it
MAX_GROWTH = 1.5                         # a candidate longer than this times production is rejected
SLEEP_GAP = 15                           # seconds a 1 s poll must take before we call it a sleep
FALLBACK = 'to finish by hand: ./demo.sh push agents/v2.md "v2: escalate when the files are silent"' 

IMPROVER_AGENTS_MD = """\
# Instruction editor

You edit the instruction files (AGENTS.md) of other agents. You get the current file and the
failures a reviewer found. You reply with exactly what the task asks for, and nothing else.
"""

TASK = """\
MLflow traced a customer support agent and a reviewer failed some of its answers. Fix the agent's
instructions so it stops making these mistakes.

## Failures

{failures}

## The agent's current AGENTS.md

```markdown
{agents_md}
```

## Rules

- Change only what these failures require. Keep every other line exactly as it is.
- Never add facts, numbers, limits or policies; the agent must get those from its policy files.
- If an instruction makes the agent state facts that aren't in its policy files, replace it with
  an instruction to say the files don't cover the question and that a specialist will reply by
  email within 1 business day.
- Keep it short: at most about as many lines as you replace, wrapped at about 100 characters like
  the rest of the file.
- Reply with only the complete corrected AGENTS.md, inside one ```markdown code block.
"""


def is_pass(assessment) -> bool:
    return assessment is not None and assessment.value in (True, "yes")


def verdicts(trace) -> dict:
    """The newest assessment per check on a trace."""
    out = {}
    for a in sorted(trace.info.assessments, key=lambda a: a.create_time_ms or 0):
        if a.name in CHECKS:
            out[a.name] = a
    return out


def agent_traces(experiment_id: str) -> list:
    return [t for t in mlflow.search_traces(locations=[experiment_id], return_type="list",
                                            order_by=["timestamp_ms ASC"])
            if t.data.spans and t.data.spans[0].name == "openclaw_agent"]


def answer_of(trace) -> str:
    out = trace.data.spans[0].outputs
    if isinstance(out, dict) and out.get("messages"):
        return str(out["messages"][-1].get("content", ""))
    return str(out)


def openclaw_reply(env: dict, message_file: Path | None = None, message: str | None = None) -> str:
    """One OpenClaw turn. A normal turn takes 20-40 s.
    Killing OpenClaw mid-turn can leave its profile locked for minutes, so this never does: it shows
    that it's still waiting, and Ctrl-C (which OpenClaw handles itself) is the way out. Output goes to
    files, not pipes, so a worker process that outlives the CLI can't block us.
    If the computer sleeps mid-turn, OpenClaw's own turn timeout counts the sleep and the turn fails
    when it wakes ("worker task timed out"). That failure is retried once, since nothing was wrong."""
    stdout, stderr, code, slept = _openclaw_turn(env, message_file, message)
    if code != 0 and slept:
        print("   the computer was asleep during that turn, so OpenClaw timed it out; asking again ...",
              flush=True)
        stdout, stderr, code, _ = _openclaw_turn(env, message_file, message)
    if code != 0:
        raise SystemExit(f"openclaw failed ({code}): {stderr[-600:]}\n{FALLBACK}")
    payload = json.loads(stdout)
    for key in ("payloads", "messages"):
        texts = [i.get("text", "") for i in payload.get(key) or [] if isinstance(i, dict)]
        if any(texts):
            return "\n".join(t for t in texts if t)
    return str(payload.get("final") or payload.get("text") or "")


def _openclaw_turn(env: dict, message_file: Path | None, message: str | None) -> tuple[str, str, int, bool]:
    cmd = ["openclaw", "agent", "--local", "--agent", "main", "--session-id",
           f"demo-{uuid.uuid4().hex[:8]}", "--json"]
    cmd += ["--message-file", str(message_file)] if message_file else ["-m", message]
    slept = False
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, text=True)
        start = last = time.time()
        while proc.poll() is None:
            time.sleep(1)
            now = time.time()
            if now - last > SLEEP_GAP:  # a 1 s poll that took much longer: the computer slept
                slept = True
                print(f"   ... the computer was asleep for {int(now - last)} s; OpenClaw's turn may time out "
                      f"(it will be retried once)", flush=True)
            last = now
            waited = int(now - start)
            if waited >= 60 and waited % 30 == 0:
                print(f"   ... still waiting for OpenClaw ({waited} s). Ctrl-C stops safely; {FALLBACK}",
                      flush=True)
        out.seek(0)
        err.seek(0)
        return out.read(), err.read(), proc.returncode, slept


def improver_env(demo: Path) -> dict:
    """A second, minimal OpenClaw profile for the improver: no connector (so its run isn't one of
    the support traces `score` and `gate` read), and a neutral AGENTS.md instead of the support
    agent's own, so the quick facts it's fixing don't steer it."""
    state, ws = demo / "improver" / "openclaw", demo / "improver" / "workspace"
    state.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "AGENTS.md").write_text(IMPROVER_AGENTS_MD)
    config = state / "openclaw.json"
    if not config.exists():
        config.write_text(json.dumps({
            "agents": {"defaults": {"workspace": str(ws), "model": {"primary": "anthropic/claude-haiku-5-5"},
                                    "skipBootstrap": True}},
            "tools": {"allow": ["read"]},
            "gateway": {"mode": "local", "port": 19558, "bind": "loopback"},
        }, indent=2))
    return {**os.environ, "OPENCLAW_STATE_DIR": str(state), "OPENCLAW_CONFIG_PATH": str(config)}


def improve(args) -> None:
    exp = mlflow.set_experiment(args.experiment)
    prod = mlflow.genai.load_prompt(f"prompts:/{args.name}@production")

    failing, seen = [], set()
    for t in reversed(agent_traces(exp.experiment_id)):  # newest first
        v = verdicts(t)
        question = t.info.request_preview or ""
        if v and not all(is_pass(v.get(n)) for n in CHECKS) and question not in seen and question != HOLDOUT:
            failing.append((t, v))
            seen.add(question)
    failing = failing[:3]
    if not failing:
        raise SystemExit("nothing has failed yet: ask a question, then run ./demo.sh score")

    report = []
    for t, v in failing:
        reasons = "; ".join(f"{n}: {v[n].rationale}" for n in CHECKS if n in v and not is_pass(v[n]))
        report.append(f"- Question: {t.info.request_preview}\n  Answer: {answer_of(t)[:600]}\n  Reviewer: {reasons[:800]}")
    task = args.demo / "improver" / "task.md"
    env = improver_env(args.demo)
    task.write_text(TASK.format(failures="\n".join(report), agents_md=prod.template.strip()))

    print(f"OpenClaw is rewriting v{prod.version} from {len(failing)} failing trace(s) ...")
    reply = openclaw_reply(env, message_file=task)
    m = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", reply, re.S)
    candidate = (m.group(1) if m else reply).strip() + "\n"

    if not candidate.startswith("# "):
        raise SystemExit(f"OpenClaw didn't return an AGENTS.md; nothing registered. Its reply:\n{reply[:800]}")
    if len(candidate) > MAX_GROWTH * len(prod.template):
        raise SystemExit(f"rejected: the candidate is {len(candidate) / len(prod.template):.1f}x the length "
                         f"of production (limit {MAX_GROWTH}x); nothing registered")
    if candidate.strip() == prod.template.strip():
        raise SystemExit("OpenClaw returned the file unchanged; nothing registered")

    pv = mlflow.genai.register_prompt(
        name=args.name, template=candidate,
        commit_message=f"proposed by OpenClaw from {len(failing)} failing trace(s)",
        tags={"author": "openclaw", "status": "candidate",
              "source_traces": ",".join(t.info.trace_id for t, _ in failing)[:500]})
    diff = difflib.unified_diff(prod.template.splitlines(), candidate.splitlines(),
                                f"v{prod.version} (@production)", f"v{pv.version} (candidate)", lineterm="", n=1)
    print("\n".join(diff))
    print(f"\nregistered v{pv.version} as a candidate; @production is still v{prod.version}. Next: ./demo.sh gate")


def gate(args) -> None:
    exp = mlflow.set_experiment(args.experiment)
    client = MlflowClient()
    prod = mlflow.genai.load_prompt(f"prompts:/{args.name}@production")
    version = args.version or max(int(v.version) for v in client.search_prompt_versions(args.name))
    if int(version) == int(prod.version):
        raise SystemExit(f"v{version} is already production; run ./demo.sh improve first")
    cand = mlflow.genai.load_prompt(f"prompts:/{args.name}/{version}")

    questions = []
    for t in agent_traces(exp.experiment_id):
        v, q = verdicts(t), t.info.request_preview or ""
        if v and not all(is_pass(v.get(n)) for n in CHECKS) and q not in questions and q != HOLDOUT:
            questions.append(q)
    questions = questions[-3:] + [HOLDOUT]

    workspace = args.demo / "workspace"
    (workspace / "AGENTS.md").write_text(cand.template)
    print(f"gate: v{version} (candidate) vs v{prod.version} (production), on {len(questions)} questions "
          f"(the last one the improver never saw)")
    runs = []
    try:
        for q in questions:
            print(f"   asking: {q}", flush=True)
            start = int(time.time() * 1000)
            openclaw_reply(dict(os.environ), message=q)
            t = next(t for t in reversed(agent_traces(exp.experiment_id)) if t.info.request_time >= start)
            client.set_trace_tag(t.info.trace_id, "gate_candidate", f"v{version}")
            runs.append(t)
        print("   scoring ...")
        quiet_evaluate(runs, [read_knowledge, grounded_in_docs(args.judge_model)])
    finally:
        (workspace / "AGENTS.md").write_text(prod.template)  # production keeps running until promoted

    failures = []
    for t in runs:
        v = verdicts(mlflow.get_trace(t.info.trace_id))
        marks = "  ".join(f"{n}={'pass' if is_pass(v.get(n)) else 'FAIL'}" for n in CHECKS)
        print(f"   {(t.info.request_preview or '')[:50]:<50} {marks}")
        for n in CHECKS:
            if not is_pass(v.get(n)):
                failures.append(f"{t.info.request_preview}: {n}: {(v[n].rationale if n in v else 'no verdict')[:200]}")

    if failures:
        mlflow.genai.set_prompt_version_tag(args.name, version, "gate", "failed")
        mlflow.genai.set_prompt_version_tag(args.name, version, "gate_reasons", "; ".join(failures)[:500])
        print(f"\nREJECTED: v{version} stays a candidate; @production is still v{prod.version}.")
        for f in failures:
            print(f"   {f}")
        return
    mlflow.genai.set_prompt_alias(args.name, alias="production", version=version)
    mlflow.genai.set_prompt_version_tag(args.name, version, "gate", "passed")
    mlflow.genai.set_prompt_version_tag(args.name, version, "promoted_by", "gate")
    mlflow.genai.set_prompt_version_tag(args.name, version, "replaced_version", str(prod.version))
    print(f"\nPROMOTED by the gate: @production v{prod.version} -> v{version}. (./demo.sh use {prod.version} rolls back)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", type=Path, required=True, help="the demo folder (demo.sh's $DEMO)")
    ap.add_argument("--name", default="openclaw-agents-md")
    ap.add_argument("--experiment", default="openclaw-demo")
    ap.add_argument("--judge-model", default=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "anthropic:/claude-sonnet-5-5"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("improve", help="OpenClaw proposes a corrected AGENTS.md from the failing traces")
    g = sub.add_parser("gate", help="test a candidate; promote it only if every check passes")
    g.add_argument("version", nargs="?", type=int, help="the candidate version (default: the newest)")
    args = ap.parse_args()
    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = args.judge_model
    try:
        improve(args) if args.cmd == "improve" else gate(args)
    except KeyboardInterrupt:
        raise SystemExit(f"\ninterrupted: @production is unchanged. Run the same command again, or {FALLBACK}") from None


if __name__ == "__main__":
    main()
