"""The automatic half of the live demo: MLflow finds the issue, OpenClaw proposes a fix, a gate decides.

    python self_improve_loop.py detect --demo ~/odsc-5min      # MLflow issue detection over the traces
    python self_improve_loop.py improve --demo ~/odsc-5min     # failing traces -> OpenClaw -> candidate version
    python self_improve_loop.py gate --demo ~/odsc-5min        # test the candidate; promote only if it passes

`detect` runs MLflow's automatic issue detection (`discover_issues`) over the agent's traces: our
grounding judge triages every trace, and MLflow clusters the failures into named Issues, each with
a root cause, and links them to the traces they affect. (MLflow's default triage judge rates user
satisfaction; it can't see that an answer isn't in the policies, so it passes confident made-up
answers.)

`improve` takes the newest open Issue (or, without one, the traces that `score` failed, with the
judge's reasons) and hands it to OpenClaw together with the AGENTS.md in production. OpenClaw replies with a corrected AGENTS.md,
which is registered in the Prompt Registry as a candidate. The alias doesn't move: whoever writes
a fix doesn't get to ship it.

`gate` runs the candidate through OpenClaw on the questions that failed (the Issue's traces and
any `score` failed), plus a question the
improver never saw (the refund window, the demo's holdout), scores those runs with the same two
checks, and moves `@production` to the candidate only if every check passes. Otherwise production
stays where it is, and the reasons are recorded on the candidate. A promotion marks the Issue
resolved.

Set MLFLOW_TRACKING_URI first (demo.sh does). Needs the harness venv's mlflow and an Anthropic key.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
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
from mlflow.entities.assessment import IssueReference  # noqa: E402
from mlflow.entities.issue import IssueStatus  # noqa: E402
from mlflow.tracking._tracking_service.utils import _get_store  # noqa: E402

from self_improvement_demo import grounded_in_docs, quiet_evaluate, read_knowledge  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)

CHECKS = ("read_knowledge", "grounded_in_docs")
HOLDOUT = "What's your refund window?"  # never shown to the improver; only the gate asks it
GATE_FAILED = 2                          # failed questions the gate re-asks (plus the holdout); ~30 s each
MAX_GROWTH = 1.5                         # a candidate longer than this times production is rejected
SLEEP_GAP = 15                           # seconds a 1 s poll must take before we call it a sleep
FALLBACK = 'to finish by hand: ./demo.sh push agents/v2.md "v2: escalate when the files are silent"' 

IMPROVER_AGENTS_MD = """\
# Instruction editor

You edit the instruction files (AGENTS.md) of other agents. You get the current file and the
failures a reviewer found. You reply with exactly what the task asks for, and nothing else.
"""

TASK = """\
MLflow traced a customer support agent and found a problem in its answers. Fix the agent's
instructions so it stops making these mistakes.

{issue}## Failures

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


def index_trace(trace, prompt_version) -> None:
    """Link a trace to the AGENTS.md version that produced it, and copy its spans into the server's
    span table. The connector uploads spans only as a JSON artifact, and the server reads the table
    for span.* searches, the Overview's tool charts and the judges `monitor` runs on the server.
    The connector's spans carry an OTel trace id that doesn't match the trace's own id (logged as
    is, they'd become a second trace), so each span is rebuilt with the trace's id first."""
    import base64

    from mlflow.entities import Span

    MlflowClient().link_prompt_versions_to_trace([prompt_version], trace.info.trace_id)
    otel_id = base64.b64encode(bytes.fromhex(trace.info.trace_id.removeprefix("tr-"))).decode()
    spans = [Span.from_dict({**s.to_dict(), "trace_id": otel_id}) for s in trace.data.spans]
    _get_store().log_spans(trace.info.experiment_id, spans)


def open_issues(experiment_id: str) -> list:
    """Issues `detect` found that nobody has resolved or rejected yet, worst first."""
    issues = _get_store().search_issues(experiment_id=experiment_id, filter_string="status = 'pending'",
                                        include_trace_count=True)
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted((i for i in issues if str(i.severity) != "not_an_issue"),
                  key=lambda i: (rank.get(str(i.severity), 3), -i.created_timestamp))


def issue_traces(traces: list, issue_id: str) -> list:
    """The traces `detect` linked to an Issue (it logs an IssueReference assessment on each)."""
    return [t for t in traces
            if any(isinstance(a, IssueReference) and a.issue_id == issue_id for a in t.info.assessments)]


def answer_of(trace) -> str:
    out = trace.data.spans[0].outputs
    if isinstance(out, dict) and out.get("messages"):
        return str(out["messages"][-1].get("content", ""))
    return str(out)


def openclaw_reply(env: dict, message_file: Path | None = None, message: str | None = None,
                   session_id: str | None = None) -> str:
    """One OpenClaw turn. A normal turn takes 20-40 s.
    Killing OpenClaw mid-turn can leave its profile locked for minutes, so this never does: it shows
    that it's still waiting, and Ctrl-C (which OpenClaw handles itself) is the way out. Output goes to
    files, not pipes, so a worker process that outlives the CLI can't block us.
    If the computer sleeps mid-turn, OpenClaw's own turn timeout counts the sleep and the turn fails
    when it wakes ("worker task timed out"). That failure is retried once, since nothing was wrong.
    Turns that share a session_id are one conversation; without one, every turn starts fresh."""
    session_id = session_id or f"demo-{uuid.uuid4().hex[:8]}"
    stdout, stderr, code, slept = _openclaw_turn(env, message_file, message, session_id)
    if code != 0 and slept:
        print("   the computer was asleep during that turn, so OpenClaw timed it out; asking again ...",
              flush=True)
        stdout, stderr, code, _ = _openclaw_turn(env, message_file, message, session_id)
    if code != 0:
        raise SystemExit(f"openclaw failed ({code}): {stderr[-600:]}\n{FALLBACK}")
    payload = json.loads(stdout)
    for key in ("payloads", "messages"):
        texts = [i.get("text", "") for i in payload.get(key) or [] if isinstance(i, dict)]
        if any(texts):
            return "\n".join(t for t in texts if t)
    return str(payload.get("final") or payload.get("text") or "")


def _openclaw_turn(env: dict, message_file: Path | None, message: str | None,
                   session_id: str) -> tuple[str, str, int, bool]:
    cmd = ["openclaw", "agent", "--local", "--agent", "main", "--session-id", session_id, "--json"]
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

    traces = agent_traces(exp.experiment_id)
    issues = open_issues(exp.experiment_id)
    issue = issues[0] if issues else None
    failing, seen, report = [], set(), []
    if issue:  # what `detect` found: the Issue, and the traces it's linked to
        for t in reversed(issue_traces(traces, issue.issue_id)):
            q = t.info.request_preview or ""
            if q not in seen and q != HOLDOUT:
                failing.append(t)
                seen.add(q)
                note = next(a.rationale for a in t.info.assessments
                            if isinstance(a, IssueReference) and a.issue_id == issue.issue_id)
                report.append(f"- Question: {q}\n  Answer: {answer_of(t)[:600]}\n  MLflow: {(note or '')[:800]}")
        failing, report = failing[:3], report[:3]
        header = (f"## Issue MLflow found: {issue.name} (severity {issue.severity})\n\n{issue.description}\n\n"
                  f"Root cause: {' '.join(issue.root_causes or [])}\n\n")
        source = f"Issue '{issue.name}' ({len(failing)} trace(s))"
    if not failing:  # no Issue yet: the traces `score` failed, with the judge's reasons
        issue, header = None, ""
        for t in reversed(traces):  # newest first
            v = verdicts(t)
            q = t.info.request_preview or ""
            if v and not all(is_pass(v.get(n)) for n in CHECKS) and q not in seen and q != HOLDOUT:
                failing.append(t)
                seen.add(q)
                reasons = "; ".join(f"{n}: {v[n].rationale}" for n in CHECKS if n in v and not is_pass(v[n]))
                report.append(f"- Question: {q}\n  Answer: {answer_of(t)[:600]}\n  Reviewer: {reasons[:800]}")
        failing, report = failing[:3], report[:3]
        source = f"{len(failing)} failing trace(s)"
    if not failing:
        raise SystemExit("nothing has failed yet: ask a question, then run ./demo.sh detect (or score)")

    task = args.demo / "improver" / "task.md"
    env = improver_env(args.demo)
    task.write_text(TASK.format(issue=header, failures="\n".join(report), agents_md=prod.template.strip()))

    print(f"OpenClaw is rewriting v{prod.version} from {source} ...")
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
        commit_message=f"proposed by OpenClaw from {source}",
        tags={"author": "openclaw", "status": "candidate",
              "source_traces": ",".join(t.info.trace_id for t in failing)[:500],
              **({"issue_id": issue.issue_id} if issue else {})})
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

    traces = [t for t in agent_traces(exp.experiment_id) if "gate_candidate" not in t.info.tags]
    issue_id = cand.tags.get("issue_id")
    questions = []
    for t in issue_traces(traces, issue_id) if issue_id else []:  # the Issue's traces first
        q = t.info.request_preview or ""
        if q not in questions and q != HOLDOUT:
            questions.append(q)
    questions = questions[-GATE_FAILED:]
    for t in traces:  # then anything `score` failed
        v, q = verdicts(t), t.info.request_preview or ""
        if v and not all(is_pass(v.get(n)) for n in CHECKS) and q not in questions and q != HOLDOUT:
            questions.append(q)
    questions = questions[:GATE_FAILED] + [HOLDOUT]

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
            index_trace(t, cand)  # ran the candidate, not production
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
    if issue_id:
        issue = _get_store().update_issue(issue_id, status=IssueStatus.RESOLVED)
        print(f"   Issue '{issue.name}' marked resolved in MLflow.")


def detect(args) -> None:
    from mlflow.genai.discovery.pipeline import discover_issues

    exp = mlflow.set_experiment(args.experiment)
    traces = [t for t in agent_traces(exp.experiment_id) if "gate_candidate" not in t.info.tags]
    if not traces:
        raise SystemExit("no traces yet: ./demo.sh traffic, or ./demo.sh ask a question")
    print(f"MLflow issue detection over {len(traces)} trace(s), with {args.judge_model} ...", flush=True)
    start = time.time()
    # Triage with our grounding judge: MLflow's default judge rates user satisfaction, and a
    # confident made-up answer satisfies. MLflow then does the clustering and root-cause analysis.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = discover_issues(experiment_id=exp.experiment_id, traces=traces, model=args.judge_model,
                                 scorers=[grounded_in_docs(args.judge_model), read_knowledge],
                                 categories=args.categories, max_issues=5)
    issues = [i for i in result.issues if str(i.severity) != "not_an_issue"]
    print(f"done in {time.time() - start:.0f} s"
          + (f", about ${result.total_cost_usd:.2f}" if result.total_cost_usd else "") + "\n")
    if not issues:
        print("no issues found")
        return
    for i in issues:
        n = len(issue_traces([mlflow.get_trace(t.info.trace_id) for t in traces], i.issue_id))
        print(f"Issue: {i.name}  [{i.severity}, {n} trace(s)]")
        print(f"   {i.description}")
        for cause in i.root_causes or []:
            print(f"   root cause: {cause}")
        print()
    print(f"In MLflow: {mlflow.get_tracking_uri()}/#/experiments/{exp.experiment_id}/evaluation-runs/"
          f"{result.triage_run_id}/issues\nNext: ./demo.sh improve")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", type=Path, required=True, help="the demo folder (demo.sh's $DEMO)")
    ap.add_argument("--name", default="openclaw-agents-md")
    ap.add_argument("--experiment", default="openclaw-demo")
    ap.add_argument("--judge-model", default=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "anthropic:/claude-sonnet-5-5"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="MLflow automatic issue detection over the agent's traces")
    d.add_argument("--categories", nargs="+", default=["correctness", "adherence", "relevance", "safety"],
                   help="the kinds of issue to look for")
    sub.add_parser("improve", help="OpenClaw proposes a corrected AGENTS.md from the failing traces")
    g = sub.add_parser("gate", help="test a candidate; promote it only if every check passes")
    g.add_argument("version", nargs="?", type=int, help="the candidate version (default: the newest)")
    args = ap.parse_args()
    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = args.judge_model
    try:
        {"detect": detect, "improve": improve, "gate": gate}[args.cmd](args)
    except KeyboardInterrupt:
        raise SystemExit(f"\ninterrupted: @production is unchanged. Run the same command again, or {FALLBACK}") from None


if __name__ == "__main__":
    main()
