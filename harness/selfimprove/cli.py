"""`selfimprove <command>`: every step of the loop from the command line.

    setup     doctor · init
    traffic   ask · replay
    signal    label · review-queue · score · online · align
    store     triage · approve · queue
    update    improve · diff
    ship      gate · promote · rollback · monitor
    loop      cycle · report
    agent     self status · note · recall · propose   (run by the agent itself; JSON out)
    deployed  self lessons · sync · setup              (the operator's side of the same skill)
"""

from __future__ import annotations

import argparse
import sys


def _doctor(args) -> None:
    from selfimprove import agents, config, instructions, models, traces

    s = config.settings
    print(s.describe())
    experiment_id = traces.setup()
    print(f"prompt        {'registered, production v' + str(instructions.load('production').version) if instructions.exists() else 'not registered yet (run init)'}")
    if s.agent == "openclaw":
        from selfimprove.agents import openclaw

        print(f"openclaw      {openclaw.check()}")
        print(f"mlflow plugin {openclaw.plugin_check(experiment_id)}")
    if s.agent in ("litellm", "openclaw") and not models.has_credentials(s.agent_model):
        print(f"WARNING       no credentials for {s.agent_model}")
    try:
        agents.adapter()
        print("adapter       loads")
    except Exception as e:  # noqa: BLE001 -- report any import problem in the user's agent
        print(f"adapter       FAILED: {e}")
    if args.smoke and not instructions.exists():
        print("smoke         skipped: run `selfimprove init` first")
    elif args.smoke:
        print(f"smoke         {agents.run(args.smoke, source='eval')[:300]}")


def _init(args) -> None:
    from selfimprove import instructions, store, traces

    traces.setup()
    v = instructions.seed()
    n = store.seed_golden()
    print(f"instructions  prompts:/{instructions.config.settings.prompt_name}@production -> v{v}")
    print(f"dataset       +{n} golden cases; {store.summary()}")


def _ask(args) -> None:
    from selfimprove import agents, traces

    traces.setup()
    print(agents.run(" ".join(args.question), ref=args.ref, source=args.source))


def _replay(args) -> None:
    from pathlib import Path

    from selfimprove import agents, traces

    traces.setup()
    questions = [q.strip() for q in Path(args.file).read_text().splitlines() if q.strip() and not q.startswith("#")]
    for i, q in enumerate(questions * args.repeat, 1):
        try:
            a = agents.run(q, ref=args.ref, source=args.source)
            print(f"{i:>3}. {q[:60]:<60} -> {a[:80]!r}")
        except Exception as e:  # noqa: BLE001 -- keep replaying; the trace records the error
            print(f"{i:>3}. {q[:60]:<60} !! {e}")


def _label(args) -> None:
    from selfimprove import feedback, traces

    traces.setup()
    if args.trace_id:
        feedback.label(args.trace_id, passed=args.verdict == "pass", rationale=args.why or "",
                       expected_response=args.expected)
        print("labeled")
    else:
        print(f"{feedback.label_terminal(args.limit)} traces labeled")


def _review_queue(args) -> None:
    from selfimprove import feedback, traces

    traces.setup()
    print(feedback.review_queue(users=[u for u in (args.users or "").split(",") if u]))


def _queue(args) -> None:
    import json

    from selfimprove import store, traces

    traces.setup()
    if args.action == "list":
        for item in store.queue_items(None if args.all else "open"):
            q = store.question_of(item["record"]) if item.get("record") else ""
            print(f"{item['id']}  {item['kind']:<19} {item.get('status'):<9} {item.get('trace_id') or '':<36} "
                  f"{q[:50]!r}  {item['reason'][:80]}")
        return
    rec = None
    if args.action == "edit":
        item = next(i for i in store.queue_items(None) if i["id"] == args.id)
        rec = item.get("record") or {}
        if args.expected:
            rec.setdefault("expectations", {})["expected_response"] = args.expected
        if args.input:
            rec.setdefault("inputs", {})[store.config.settings.input_key] = args.input
    status = {"accept": "accepted", "reject": "rejected", "edit": "edited"}[args.action]
    out = store.resolve(args.id, status, note=args.note or "", **({"record": rec} if rec else {}))
    print(json.dumps({k: out.get(k) for k in ("id", "kind", "status", "applied_to") if k != "applied_to" or k in out}))


def _self(args) -> None:
    import json

    from selfimprove import agentside, traces

    traces.setup()
    if args.action == "status":
        out = agentside.status()
    elif args.action == "note":
        out = agentside.note(args.trace, user_said=args.user_said, interpretation=args.interpretation,
                             expected=args.expected, session=args.session)
    elif args.action == "recall":
        out = agentside.recall(args.request, k=args.k)
    else:
        file = args.file
        if args.text is not None:  # the deployed agent has no `write` tool: it passes the text inline
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
                f.write(args.text)
            file = f.name
        out = agentside.propose(args.target, file or "", why=args.why,
                                evidence=[e for e in (args.evidence or "").split(",") if e])
    print(json.dumps(out, default=str))


def _job(fn):
    def run(args):
        from selfimprove import traces

        traces.setup()
        kwargs = {k: v for k, v in vars(args).items() if k not in ("func", "command", "action") and v is not None}
        result = fn(**kwargs)
        if isinstance(result, str):
            print(result)
    return run


def _diff(args) -> None:
    from selfimprove import instructions, traces

    traces.setup()
    print(instructions.diff(args.a, args.b) or "(identical)")


def _rollback(args) -> None:
    from selfimprove import instructions, traces

    traces.setup()
    print(f"production -> v{instructions.rollback(args.reason)}")


def main(argv: list[str] | None = None) -> None:
    from selfimprove.workflows import align, approve, cycle, deployed, improve, report, score, ship, triage

    p = argparse.ArgumentParser(prog="selfimprove", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("doctor", help="show the configuration and check that your agent loads")
    c.add_argument("--smoke", metavar="QUESTION", help="also send one question (tagged eval)")
    c.set_defaults(func=_doctor)
    sub.add_parser("init", help="register the seed instructions as v1@production; add golden cases").set_defaults(func=_init)

    c = sub.add_parser("ask", help="send one question to the agent")
    c.add_argument("question", nargs="+")
    c.add_argument("--ref", default="production")
    c.add_argument("--source", default="production", choices=["production", "eval"])
    c.set_defaults(func=_ask)
    c = sub.add_parser("replay", help="send every line of a file as production traffic")
    c.add_argument("file")
    c.add_argument("--repeat", type=int, default=1)
    c.add_argument("--ref", default="production")
    c.add_argument("--source", default="production", choices=["production", "eval"])
    c.set_defaults(func=_replay)

    c = sub.add_parser("label", help="label production traces (terminal), or one trace by id")
    c.add_argument("trace_id", nargs="?")
    c.add_argument("verdict", nargs="?", choices=["pass", "fail"])
    c.add_argument("--why")
    c.add_argument("--expected", help="the correct response")
    c.add_argument("--limit", type=int, default=20)
    c.set_defaults(func=_label)
    c = sub.add_parser("review-queue", help="create label schemas + an MLflow review queue")
    c.add_argument("--users", help="comma-separated reviewer usernames")
    c.set_defaults(func=_review_queue)

    sub.add_parser("score", help="run scorers over new production traces").set_defaults(func=_job(score.score))
    c = sub.add_parser("online", help="register LLM judges for server-side sampling")
    c.add_argument("--sample-rate", dest="sample_rate", type=float, default=0.1)
    c.add_argument("--fast-sample-rate", dest="fast_sample_rate", type=float, default=1.0,
                   help="sampling rate for the first-pass decision model (SI_FAST_JUDGE_MODEL)")
    c.set_defaults(func=_job(score.online))
    c = sub.add_parser("align", help="check judge-vs-human agreement; re-align if it is low")
    c.add_argument("--force", action="store_true", default=None)
    c.set_defaults(func=_job(align.align))
    c = sub.add_parser("triage", help="new failures -> themes -> proposed tests")
    c.add_argument("--force", action="store_true", default=None, help="ignore SI_TRIAGE_THRESHOLD")
    c.add_argument("--no-analyst", dest="use_analyst", action="store_false", default=None)
    c.set_defaults(func=_job(triage.triage))
    c = sub.add_parser("approve", help="approver checks; merge approved tests, queue the rest")
    c.add_argument("--run-id", dest="run_id", help="one triage run (default: every pending proposal)")
    c.add_argument("--no-reviewer", dest="use_reviewer", action="store_false", default=None)
    c.set_defaults(func=_job(approve.approve))

    c = sub.add_parser("queue", help="the human queue: list, accept, reject, edit")
    c.add_argument("action", choices=["list", "accept", "reject", "edit"])
    c.add_argument("id", nargs="?")
    c.add_argument("--all", action="store_true")
    c.add_argument("--note")
    c.add_argument("--expected", help="edit: the correct response")
    c.add_argument("--input", help="edit: a reworded input")
    c.set_defaults(func=_queue)

    c = sub.add_parser("improve", help="write a candidate (SI_IMPROVER); never promotes")
    c.add_argument("--improver", choices=["analyst", "gepa", "metaprompt", "command"])
    c.add_argument("--budget", type=int, help="optimizer metric calls")
    c.add_argument("--from-file", dest="from_file", help="register a hand-written candidate instead")
    c.add_argument("-m", "--message", help="commit message for --from-file")
    c.set_defaults(func=_job(improve.improve))
    c = sub.add_parser("diff", help="diff two instruction versions")
    c.add_argument("a", nargs="?", default="production")
    c.add_argument("b", nargs="?", default="latest")
    c.set_defaults(func=lambda a: _diff(argparse.Namespace(a=a.a, b=(_latest() if a.b == "latest" else a.b))))

    c = sub.add_parser("gate", help="candidate vs production on CI suite + holdout")
    c.add_argument("candidate", nargs="?", type=int)
    c.add_argument("--promote", action="store_true", default=None, help="move the alias if the gate passes")
    c.set_defaults(func=_job(ship.run_gate))
    c = sub.add_parser("promote", help="move production to a version (a person's decision)")
    c.add_argument("version", type=int)
    c.add_argument("--reason")
    c.set_defaults(func=_job(ship.promote))
    c = sub.add_parser("rollback", help="point production back at the version it replaced")
    c.add_argument("--reason", default="manual rollback")
    c.set_defaults(func=_rollback)
    sub.add_parser("monitor", help="roll back a gate promotion if live traffic got worse").set_defaults(func=_job(ship.monitor))

    c = sub.add_parser("cycle", help="the whole loop once")
    c.add_argument("--improver", choices=["analyst", "gepa", "metaprompt", "command"])
    c.add_argument("--force", action="store_true", default=None, help="skip the signal thresholds")
    c.set_defaults(func=_job(cycle.cycle))
    sub.add_parser("report", help="the baseline numbers and what a person still owns").set_defaults(
        func=_job(report.report))

    c = sub.add_parser("self", help="commands the agent runs about itself (JSON out)")
    actions = c.add_subparsers(dest="action", required=True)
    actions.add_parser("status", help="production version, last gate, live failure rate").set_defaults(func=_self)
    a = actions.add_parser("note", help="the user corrected an answer: report it for a person to confirm")
    a.add_argument("--trace", default="previous", help='a trace id, or "previous" (the newest answer)')
    a.add_argument("--user-said", dest="user_said", required=True, help="the user's exact words")
    a.add_argument("--interpretation", required=True, help="one sentence: what the agent got wrong")
    a.add_argument("--expected", help="the correct answer, only if the user stated it")
    a.add_argument("--session", help="the conversation's mlflow.trace.session key, if known")
    a.set_defaults(func=_self)
    a = actions.add_parser("recall", help="the confirmed lessons that fit a request")
    a.add_argument("request", help="the user's request, in their words")
    a.add_argument("--k", type=int, default=3, help="at most this many lessons (1-5)")
    a.set_defaults(func=_self)
    a = actions.add_parser("propose", help="propose a change to yourself; the gate or a person decides")
    a.add_argument("--target", required=True, help="AGENTS.md, or skill:<name>")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="the full proposed file (a skill folder works too)")
    src.add_argument("--text", help="the full proposed file, inline")
    a.add_argument("--why", required=True, help="two or three sentences: what the evidence shows")
    a.add_argument("--evidence", required=True, help="comma-separated trace ids of confirmed failures")
    a.set_defaults(func=_self)
    # Operator side: the deployed agent's wrapper refuses these.
    a = actions.add_parser("lessons", help="build the lessons pack from confirmed failures, as a candidate")
    a.add_argument("--gate", action="store_true", help="run the gate on it now")
    a.add_argument("--no-llm", dest="use_llm", action="store_false", help="write lessons without an LLM")
    a.set_defaults(func=_job(deployed.lessons_job))
    a = actions.add_parser("sync", help="write production AGENTS.md / LESSONS.md into the live workspace")
    a.add_argument("--workspace", help="the live workspace (default SI_SELF_WORKSPACE)")
    a.set_defaults(func=_job(deployed.sync))
    a = actions.add_parser("setup", help="write the deployed OpenClaw profile (read + exec, Workshop off)")
    a.add_argument("--workspace", required=True, help="the live workspace")
    a.add_argument("--profile", help="the profile folder (default .selfimprove/openclaw-live)")
    a.set_defaults(func=_job(deployed.setup))

    args = p.parse_args(argv)
    code = 0
    try:
        args.func(args)
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        _exit(code)


def _exit(code: int) -> None:
    """Flush MLflow's async trace queue, then exit without running finalizers.

    LiteLLM's HTTP client finalizer can deadlock on an import lock at interpreter shutdown
    (seen with litellm 1.103-1.106), which would hang a scheduled job forever after its work
    is done. Everything the job wrote is flushed first.
    """
    import os

    try:
        import mlflow

        mlflow.flush_trace_async_logging()
    except Exception:  # noqa: BLE001 -- best effort; never mask the job's own exit code
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _latest():
    from selfimprove import instructions, traces

    traces.setup()
    return instructions.latest_version()


if __name__ == "__main__":
    main(sys.argv[1:])
