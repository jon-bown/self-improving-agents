"""More of MLflow on the same OpenClaw agent: search, human feedback, judges, monitoring, sessions, datasets.

    python mlflow_features.py link --since MS              # link new traces to the AGENTS.md version that ran
    python mlflow_features.py traces ["FILTER"]            # search the traces; per-version latency, tokens, verdicts
    python mlflow_features.py prompts [A B]                # every AGENTS.md version: aliases, tags, traces; diff A B
    python mlflow_features.py feedback no "made it up"     # a person's verdict on the newest trace
    python mlflow_features.py expect "the right answer"    # a person's expected answer for the newest trace
    python mlflow_features.py judges                       # MLflow's built-in LLM judges on the newest trace
    python mlflow_features.py gateway                      # an MLflow AI Gateway endpoint for the judge model
    python mlflow_features.py monitor on|off|status        # the server scores every new trace by itself
    python mlflow_features.py chat --demo ~/odsc-5min      # a three-turn conversation, scored as one session
    python mlflow_features.py dataset                      # an evaluation dataset: golden cases + your expectations
    python mlflow_features.py eval [N] --demo ~/odsc-5min  # run version N on the dataset; compare runs in the UI

Each command is one or two MLflow APIs, and prints where to see the result in the UI. Only `chat`
and `eval` run OpenClaw (about 30 s a turn); `judges` and `chat` call the judge model; the rest
take a second or two.

Set MLFLOW_TRACKING_URI first (demo.sh does). Needs the harness venv's mlflow and an Anthropic key.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import getpass
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanType  # noqa: E402

from self_improve_loop import HOLDOUT, agent_traces, answer_of, index_trace, is_pass, openclaw_reply  # noqa: E402
from self_improvement_demo import _last_message, grounded_in_docs, quiet_evaluate, read_knowledge  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.ERROR)

HUMAN = "human_review"           # a person's verdict (feedback)
EXPECTED = "expected_response"   # a person's correct answer (expectation); Correctness reads this name
DATASET = "openclaw-golden"
MONITOR = "grounded_live"        # the judge `monitor on` registers on the server
ENDPOINT = "judge"               # the AI Gateway endpoint `gateway` creates; judges call gateway:/judge

# What a correct answer says, written from the policy files (and, for the SLA, from what they lack).
GOLDEN = [
    ("What's your refund window?",
     "Refunds are available within 14 days of purchase on every plan, requested at Settings -> "
     "Billing -> Request refund; there are no prorated refunds after 14 days."),
    ("How do I reset my password?",
     "Request a reset link; it expires 15 minutes after it's sent. If login still fails, clear the "
     "browser cache, request a new link and use it within 15 minutes."),
    ("Do you have an uptime SLA?",
     "The policy files don't cover an uptime SLA, so the agent gives no SLA figure or credits and "
     "says a specialist will reply by email within 1 business day."),
]

CHAT = [  # one conversation: a documented question, an undocumented one, then a recall check
    "I was charged but my card payment failed earlier this week. What happens now?",
    "Is there an uptime SLA credit I can claim for last week's outage?",
    "Thanks. Remind me: how many times will you retry my payment, and what happens after that?",
]

# For `monitor`: a judge that reads the trace itself (its spans, through tools), so it can run on
# the server with nothing but the trace. No code of ours runs there, only these instructions.
LIVE = """\
{{ trace }} is a customer support agent answering one question. Its `read` tool spans hold the
policy files it opened, and the root span's output is its answer.
Return true only if every factual claim in the answer (numbers, limits, SLAs, regions, steps) is
stated in a file it read, or the answer says the files don't cover the question and escalates to a
specialist. Return false if it states anything the files don't contain.
"""


# --- helpers --------------------------------------------------------------------------------------

def experiment():
    return mlflow.set_experiment(os.environ.get("DEMO_EXPERIMENT", "openclaw-demo"))


def ui(path: str) -> str:
    return f"{mlflow.get_tracking_uri()}/#/experiments/{experiment().experiment_id}/{path}"


def newest(trace_id: str | None):
    if trace_id:
        return mlflow.get_trace(trace_id)
    traces = agent_traces(experiment().experiment_id)
    if not traces:
        raise SystemExit("no traces yet: ./demo.sh ask a question first")
    return mlflow.get_trace(traces[-1].info.trace_id)


def question_of(trace) -> str:
    return (trace.info.request_preview or _last_message(trace.data.spans[0].inputs))[:70]


def version_of(trace) -> str:
    """The AGENTS.md version `link` (or the gate) tied this trace to."""
    linked = json.loads(trace.info.tags.get("mlflow.linkedPrompts", "[]") or "[]")
    return ",".join(f"v{p['version']}" for p in linked) or "-"


def latest(trace, name: str):
    found = [a for a in trace.info.assessments if a.name == name]
    return max(found, key=lambda a: a.create_time_ms or 0) if found else None


def mark(a) -> str:
    if a is None:
        return "  - "
    return "pass" if is_pass(a) else "FAIL"


def yes_no(a) -> str:
    return "-" if a is None else "yes" if is_pass(a) else "no"


def files_read(trace) -> list[str]:
    return [Path(str(s.inputs.get("path"))).name for s in trace.data.spans
            if s.span_type == SpanType.TOOL and s.name == "read" and isinstance(s.inputs, dict)]


def human() -> AssessmentSource:
    return AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=getpass.getuser())


def wrap(text: str, width: int = 96, indent: str = "      ") -> str:
    import textwrap
    return textwrap.fill(" ".join(str(text).split()), width, initial_indent=indent,
                         subsequent_indent=" " * len(indent))


# --- link: which AGENTS.md produced this trace ---------------------------------------------------

def link(args) -> None:
    """Link every trace since `--since` to the version @production points at (what `ask` just ran).
    The trace page then shows the prompt, and `prompt = 'name/N'` finds every trace a version made.
    This also copies the spans into the server's span table (see index_trace)."""
    pv = mlflow.genai.load_prompt(f"prompts:/{args.name}@production")
    n = 0
    for t in agent_traces(experiment().experiment_id):
        if t.info.request_time >= args.since and "mlflow.linkedPrompts" not in t.info.tags:
            index_trace(t, pv)
            n += 1
    if args.verbose:
        print(f"linked {n} trace(s) to {args.name} v{pv.version}")


# --- traces: search ------------------------------------------------------------------------------

EXAMPLES = [
    ("prompt = '{name}/1'", "every trace v1 produced (the link `ask` adds)"),
    ("feedback.grounded_in_docs = 'false'", "traces the grounding judge failed"),
    ("feedback.grounded_in_docs = 'false' AND prompt = '{name}/1'", "...and that v1 produced"),
    (f"feedback.{HUMAN} IS NOT NULL", "traces a person reviewed"),
    ("span.name = 'read' AND span.content ILIKE '%billing.md%'", "traces that opened billing.md"),
    ("metadata.`mlflow.traceOutputs` ILIKE '%99.9%'", "answers that say 99.9"),
    ("metadata.`mlflow.trace.session` LIKE '%chat-%'", "turns from ./demo.sh chat"),
    ("trace.execution_time_ms > 7000", "the slowest turns"),
]


def traces(args) -> None:
    exp = experiment()
    if args.filter:
        found = mlflow.search_traces(locations=[exp.experiment_id], filter_string=args.filter,
                                     return_type="list", order_by=["timestamp_ms ASC"])
        print(f'search_traces(filter_string="{args.filter}"): {len(found)} trace(s)\n')
    else:
        found = agent_traces(exp.experiment_id)
    if not found:
        print("no traces match")
        return

    print(f"{'time':<8} {'ver':<4} {'secs':>5} {'tokens':>6}  {'grounded':<8} {'live':<4} {'human':<5}  "
          f"{'question':<50} files read")
    by_version: dict[str, list] = {}
    for t in found:
        t = mlflow.get_trace(t.info.trace_id)
        v, g = version_of(t), latest(t, "grounded_in_docs")
        tokens = (t.info.token_usage or {}).get("total_tokens", 0)
        secs = (t.info.execution_duration or 0) / 1000
        by_version.setdefault(v, []).append((secs, tokens, g))
        when = datetime.fromtimestamp(t.info.request_time / 1000).strftime("%H:%M:%S")
        print(f"{when:<8} {v:<4} {secs:>5.0f} {tokens:>6}  {mark(g):<8} {mark(latest(t, MONITOR)):<4} {yes_no(latest(t, HUMAN)):<5}  "
              f"{question_of(t)[:50]:<50} {', '.join(files_read(t)) or '-'}")

    print(f"\n{'version':<8} {'traces':>6} {'avg s':>6} {'avg tokens':>10}  grounded pass rate")
    for v, rows in sorted(by_version.items()):
        scored = [g for _, _, g in rows if g is not None]
        rate = f"{sum(is_pass(g) for g in scored)}/{len(scored)}" if scored else "not scored"
        print(f"{v:<8} {len(rows):>6} {sum(r[0] for r in rows) / len(rows):>6.0f} "
              f"{sum(r[1] for r in rows) / len(rows):>10.0f}  {rate}")

    if not args.filter:
        print("\nThe same search the Traces page runs. Try one (or paste the filter into the UI's search box):")
        for f, why in EXAMPLES:
            f = f.format(name=args.name)
            try:
                n = len(mlflow.search_traces(locations=[exp.experiment_id], filter_string=f, return_type="list"))
            except Exception as e:  # noqa: BLE001 -- show the example even if this server can't run it
                n = f"error: {str(e)[:60]}"
            shell = f.replace("`", "\\`")  # a bare backtick inside "..." would run as a command
            print(f"   {n:>2} {why:<46} ./demo.sh traces \"{shell}\"")
    print(f"\nIn MLflow: {ui('traces')}")


# --- prompts: the registry's history ------------------------------------------------------------

def prompts(args) -> None:
    client = MlflowClient()
    exp = experiment()
    versions = sorted(client.search_prompt_versions(args.name), key=lambda v: int(v.version))
    print(f"{args.name}: {len(versions)} version(s)\n")
    for v in versions:
        pv = mlflow.genai.load_prompt(f"prompts:/{args.name}/{v.version}")
        n = len(mlflow.search_traces(locations=[exp.experiment_id], return_type="list",
                                     filter_string=f'prompt = "{args.name}/{v.version}"'))
        aliases = "".join(f" @{a}" for a in (pv.aliases or []))
        print(f"v{v.version}{aliases}  {n} trace(s)  {pv.commit_message}")
        tags = {k: val for k, val in (pv.tags or {}).items() if not k.startswith("mlflow.")}
        if tags:
            print(wrap("  ".join(f"{k}={val}" for k, val in tags.items())[:400]))
    if args.diff:
        a, b = (mlflow.genai.load_prompt(f"prompts:/{args.name}/{n}") for n in args.diff)
        print()
        print("\n".join(difflib.unified_diff(a.template.splitlines(), b.template.splitlines(),
                                             f"v{a.version}", f"v{b.version}", lineterm="", n=1)) or "identical")
    print(f"\nIn MLflow: {ui('prompts')}  (open one; Compare shows any two versions side by side)")


# --- feedback and expectations: what a person says -----------------------------------------------

def feedback(args) -> None:
    t = newest(args.trace)
    value = args.verdict in ("yes", "y", "good", "up", "pass", "true")
    mlflow.log_feedback(trace_id=t.info.trace_id, name=HUMAN, value=value, source=human(),
                        rationale=args.comment or None)
    print(f"logged {HUMAN}={'yes' if value else 'no'} from {getpass.getuser()} on: {question_of(t)}")
    g = latest(t, "grounded_in_docs")
    if g is not None:
        agree = "agrees" if is_pass(g) == value else "DISAGREES"
        print(f"   the grounded_in_docs judge said {'pass' if is_pass(g) else 'fail'}: it {agree} with you")
    print(f"In MLflow: the trace's Assessments panel, next to the judges' verdicts: {ui('traces')}")


def expect(args) -> None:
    t = newest(args.trace)
    mlflow.log_expectation(trace_id=t.info.trace_id, name=EXPECTED, value=args.answer, source=human())
    print(f"logged {EXPECTED} on: {question_of(t)}")
    print("   ./demo.sh judges now adds Correctness (the answer vs yours); ./demo.sh dataset makes it a test case")


# --- judges: MLflow's built-in LLM judges --------------------------------------------------------

def judges(args) -> None:
    from mlflow.genai.scorers import (Completeness, Correctness, Guidelines, RelevanceToQuery, Safety,
                                      ToolCallEfficiency)

    t = newest(args.trace)
    m = args.judge_model
    scorers = [
        RelevanceToQuery(model=m), Safety(model=m), Completeness(model=m), ToolCallEfficiency(model=m),
        Guidelines(name="no_unverified_promises", model=m,
                   guidelines="The response doesn't promise credits, refunds or guarantees."),
        grounded_in_docs(m), read_knowledge,
    ]
    if latest(t, EXPECTED) is not None:
        scorers.append(Correctness(model=m))
    names = [s.name for s in scorers]
    print(f"{len(scorers)} judges on: {question_of(t)}")
    print(wrap(answer_of(t)[:300], indent="   answer: "))
    start = time.time()
    quiet_evaluate([t], scorers)
    t = mlflow.get_trace(t.info.trace_id)
    print(f"done in {time.time() - start:.0f} s\n")
    for n in names:
        a = latest(t, n)
        kind = "code" if n == "read_knowledge" else "yours" if n == "grounded_in_docs" else "built-in"
        print(f"   {n:<24} {mark(a):<5} ({kind})")
        if a is not None and a.rationale:
            print(wrap(a.rationale[:260], indent="      "))
    print("\nThe built-in judges see the question and the answer, not the policies, so a confident made-up\n"
          "answer looks relevant, safe and complete. A one-line Guidelines judge, grounded_in_docs (it sees\n"
          "the files read) and Correctness (with your expected answer) are what catch it.")
    print(f"In MLflow: the trace's Assessments panel: {ui('traces')}")


# --- gateway: the judge model behind an MLflow AI Gateway endpoint -------------------------------

def ensure_gateway(judge_model: str, verbose: bool = False) -> str:
    """An AI Gateway endpoint on the tracking server for the judge model. The server keeps the API
    key as an encrypted secret, so anything that calls `gateway:/judge` (a judge running inside the
    server, a teammate's notebook) needs no key of its own, and every call is traced with its tokens."""
    from mlflow.entities.gateway_endpoint import GatewayEndpointModelConfig, GatewayModelLinkageType
    from mlflow.tracking._tracking_service.utils import _get_store

    store = _get_store()
    if any(e.name == ENDPOINT for e in store.list_gateway_endpoints()):
        if verbose:
            print(f"AI Gateway endpoint {ENDPOINT} already exists")
        return f"gateway:/{ENDPOINT}"
    provider, model = judge_model.split(":/", 1)
    key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not key:
        raise SystemExit(f"{provider.upper()}_API_KEY isn't set, so the gateway has no key to store")
    secret = next((s for s in store.list_secret_infos(provider=provider) if s.secret_name == f"{provider}-demo"), None)
    if secret is None:
        secret = store.create_gateway_secret(secret_name=f"{provider}-demo", secret_value={"api_key": key},
                                             provider=provider, created_by=getpass.getuser())
    md = store.create_gateway_model_definition(name=model, secret_id=secret.secret_id, provider=provider,
                                               model_name=model, created_by=getpass.getuser())
    store.create_gateway_endpoint(
        name=ENDPOINT, created_by=getpass.getuser(), usage_tracking=True,
        model_configs=[GatewayEndpointModelConfig(model_definition_id=md.model_definition_id,
                                                  linkage_type=GatewayModelLinkageType.PRIMARY)])
    print(f"AI Gateway: endpoint {ENDPOINT} -> {provider} {model}; the key is a secret on the server now")
    return f"gateway:/{ENDPOINT}"


def gateway(args) -> None:
    from mlflow.genai.judges import make_judge

    uri = ensure_gateway(args.judge_model, verbose=True)
    judge = make_judge(name="gateway_check", model=uri, feedback_value_type=bool,
                       instructions="Does {{ outputs }} answer a customer politely?")
    start = time.time()
    fb = judge(outputs="Thanks for asking! Refunds are available within 14 days of purchase.")
    tokens = (fb.metadata or {}).get("mlflow.assessment.judgeInputTokens", 0) + \
        (fb.metadata or {}).get("mlflow.assessment.judgeOutputTokens", 0)
    print(f"a judge on {uri}: {fb.value} in {time.time() - start:.1f} s, {tokens} tokens, no API key in this process's call")
    print(f"In MLflow: {mlflow.get_tracking_uri()}/#/gateway  (AI Gateway: endpoints, models, secrets, usage)")
    print("   any judge can use it: make_judge(..., model=\"gateway:/judge\"); ./demo.sh monitor on does")


# --- monitor: the server scores new traces by itself ---------------------------------------------

def monitor(args) -> None:
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import ScorerSamplingConfig, delete_scorer, get_scorer, list_scorers

    exp = experiment()
    if args.action == "on":
        # The server only runs judges that call a model through its own AI Gateway.
        judge = make_judge(name=MONITOR, instructions=LIVE, feedback_value_type=bool,
                           model=ensure_gateway(args.judge_model))
        with contextlib.suppress(Exception):  # replace an earlier registration, whatever its model
            delete_scorer(name=MONITOR, experiment_id=exp.experiment_id, version="all")
        registered = judge.register(name=MONITOR, experiment_id=exp.experiment_id)
        registered.start(name=MONITOR, experiment_id=exp.experiment_id,
                         sampling_config=ScorerSamplingConfig(sample_rate=args.rate))
        print(f"registered {MONITOR} on the server, scoring {args.rate:.0%} of new traces "
              f"(it checks about once a minute)")
        print("   ask something, wait a minute, then: ./demo.sh traces "
              f"\"feedback.{MONITOR} IS NOT NULL\"")
    elif args.action == "off":
        try:
            get_scorer(name=MONITOR, experiment_id=exp.experiment_id).stop(name=MONITOR, experiment_id=exp.experiment_id)
            delete_scorer(name=MONITOR, experiment_id=exp.experiment_id, version="all")
            print(f"stopped and removed {MONITOR}")
        except Exception as e:  # noqa: BLE001
            print(f"nothing to stop: {str(e)[:120]}")
        return
    else:
        found = list_scorers(experiment_id=exp.experiment_id)
        if not found:
            print("no registered judges: ./demo.sh monitor on")
        for s in found:
            print(f"   {s.name:<20} sample rate {s.sample_rate or 0:.0%}")
    scored = mlflow.search_traces(locations=[exp.experiment_id], return_type="list",
                                  filter_string=f"feedback.{MONITOR} IS NOT NULL")
    print(f"   {len(scored)} trace(s) scored by {MONITOR} so far")
    print(f"In MLflow: {ui('judges')}  (the experiment's Judges tab)")


# --- chat: a conversation, judged as one session -------------------------------------------------

def chat(args) -> None:
    from mlflow.genai.scorers import (ConversationalGuidelines, ConversationCompleteness, KnowledgeRetention,
                                      UserFrustration)

    exp = experiment()
    session = f"chat-{uuid.uuid4().hex[:6]}"
    start = int(time.time() * 1000)
    print(f"session {session}: {len(CHAT)} turns with the same session id, so OpenClaw keeps the history")
    for q in CHAT:
        print(f"\n   customer: {q}", flush=True)
        reply = openclaw_reply(dict(os.environ), message=q, session_id=session)
        print(wrap(reply[:400], indent="   agent:    "))
    link(argparse.Namespace(name=args.name, since=start, verbose=False))
    turns = [t for t in agent_traces(exp.experiment_id) if t.info.request_time >= start]
    sessions = {t.info.trace_metadata.get("mlflow.trace.session") for t in turns}
    print(f"\n{len(turns)} trace(s), session id(s) in MLflow: {', '.join(sorted(s or '-' for s in sessions))}")
    m = args.judge_model
    scorers = [UserFrustration(model=m), ConversationCompleteness(model=m), KnowledgeRetention(model=m),
               ConversationalGuidelines(name="no_unverified_promises", model=m,
                                        guidelines="The assistant never promises credits, refunds or guarantees.")]
    print("judging the whole conversation ...", flush=True)
    quiet_evaluate([mlflow.get_trace(t.info.trace_id) for t in turns], scorers)
    first = mlflow.get_trace(turns[0].info.trace_id)  # session verdicts land on the first turn
    for s in scorers:
        a = latest(first, s.name)
        value = a.value if a is not None else "-"
        print(f"   {s.name:<26} {value}")
        if a is not None and a.rationale:
            print(wrap(a.rationale[:240]))
    print(f"\nIn MLflow: {ui('chat-sessions')}  (Sessions: the turns as one conversation)")


# --- dataset and eval: a test set, and a version run against it ----------------------------------

def dataset(args) -> None:
    from mlflow.genai.datasets import create_dataset, get_dataset

    exp = experiment()
    try:
        ds = get_dataset(name=DATASET)
    except Exception:  # noqa: BLE001 -- not created yet
        ds = create_dataset(name=DATASET, experiment_id=[exp.experiment_id], tags={"purpose": "golden"})
    records = [{"inputs": {"question": q}, "expectations": {EXPECTED: e}, "tags": {"source": "golden"}}
               for q, e in GOLDEN]
    for t in agent_traces(exp.experiment_id):  # anything a person gave an expected answer for
        e = latest(mlflow.get_trace(t.info.trace_id), EXPECTED)
        if e is not None:
            records.append({"inputs": {"question": t.info.request_preview or ""},
                            "expectations": {EXPECTED: e.value},
                            "tags": {"source": "human", "source_trace": t.info.trace_id}})
    ds.merge_records(records)
    df = ds.to_df()
    print(f"evaluation dataset {DATASET}: {len(df)} case(s) (merge_records keeps one per question)")
    for _, row in df.iterrows():
        print(f"   {row['inputs'].get('question', '')[:60]:<60} expects: {str(row['expectations'].get(EXPECTED))[:60]}")
    print(f"\nIn MLflow: {ui('datasets')}  Next: ./demo.sh eval")


def evaluate_version(args) -> None:
    from mlflow.genai.datasets import get_dataset
    from mlflow.genai.scorers import Correctness

    exp = experiment()
    client = MlflowClient()
    pv = mlflow.genai.load_prompt(f"prompts:/{args.name}/{args.version}" if args.version
                                  else f"prompts:/{args.name}@production")
    prod = mlflow.genai.load_prompt(f"prompts:/{args.name}@production")
    try:
        rows = get_dataset(name=DATASET).to_df()
    except Exception:  # noqa: BLE001
        raise SystemExit("no dataset yet: ./demo.sh dataset") from None
    rows = rows.head(args.limit)
    workspace = args.demo / "workspace"
    (workspace / "AGENTS.md").write_text(pv.template)
    print(f"eval: v{pv.version} on {len(rows)} case(s) from {DATASET} (about 30 s each)")
    runs = []
    try:
        for _, row in rows.iterrows():
            q = row["inputs"]["question"]
            print(f"   asking: {q}", flush=True)
            start = int(time.time() * 1000)
            openclaw_reply(dict(os.environ), message=q)
            t = next(t for t in reversed(agent_traces(exp.experiment_id)) if t.info.request_time >= start)
            index_trace(t, pv)
            client.set_trace_tag(t.info.trace_id, "eval_of", f"v{pv.version}")
            mlflow.log_expectation(trace_id=t.info.trace_id, name=EXPECTED,
                                   value=row["expectations"][EXPECTED], source=human())
            runs.append(mlflow.get_trace(t.info.trace_id))
    finally:
        (workspace / "AGENTS.md").write_text(prod.template)  # production keeps running

    print("   scoring ...", flush=True)
    scorers = [Correctness(model=args.judge_model), grounded_in_docs(args.judge_model), read_knowledge]
    with mlflow.start_run(run_name=f"eval v{pv.version} on {DATASET}") as run:
        mlflow.set_tags({"agents_md_version": pv.version, "dataset": DATASET})
        client.link_prompt_version_to_run(run.info.run_id, pv)
        quiet_evaluate(runs, scorers)
    names = [s.name for s in scorers]
    print(f"\n   {'question':<44} " + " ".join(f"{n:<16}" for n in names))
    totals = dict.fromkeys(names, 0)
    for t in runs:
        t = mlflow.get_trace(t.info.trace_id)
        marks = {n: mark(latest(t, n)) for n in names}
        for n in names:
            totals[n] += marks[n] == "pass"
        print(f"   {question_of(t)[:44]:<44} " + " ".join(f"{marks[n]:<16}" for n in names))
    print(f"   {'v' + str(pv.version) + ' passed':<44} " + " ".join(f"{totals[n]}/{len(runs):<14}" for n in names))
    print(f"\nIn MLflow: {ui('evaluation-runs')}  (select two eval runs and Compare)")
    if HOLDOUT in set(rows["inputs"].map(lambda i: i["question"])):
        print("   (the refund question is the gate's holdout too: it checks a fix didn't break a documented answer)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="openclaw-agents-md")
    ap.add_argument("--judge-model", default=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "anthropic:/claude-sonnet-5-5"))
    ap.add_argument("--demo", type=Path, help="the demo folder (demo.sh's $DEMO); chat and eval need it")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("link")
    p.add_argument("--since", type=int, required=True, help="epoch ms; traces from then on")
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("traces")
    p.add_argument("filter", nargs="?")
    p = sub.add_parser("prompts")
    p.add_argument("diff", nargs="*", type=int, help="two versions to diff")
    p = sub.add_parser("feedback")
    p.add_argument("verdict", choices=["yes", "no", "y", "n", "good", "bad", "up", "down", "pass", "fail"])
    p.add_argument("comment", nargs="?", default="")
    p.add_argument("--trace")
    p = sub.add_parser("expect")
    p.add_argument("answer")
    p.add_argument("--trace")
    p = sub.add_parser("judges")
    p.add_argument("--trace")
    p = sub.add_parser("monitor")
    p.add_argument("action", nargs="?", default="status", choices=["on", "off", "status"])
    p.add_argument("--rate", type=float, default=1.0, help="share of new traces to score")
    sub.add_parser("gateway")
    sub.add_parser("chat")
    sub.add_parser("dataset")
    p = sub.add_parser("eval")
    p.add_argument("version", nargs="?", type=int, help="default: @production")
    p.add_argument("--limit", type=int, default=3, help="cases to run (about 30 s each)")
    args = ap.parse_args()
    if args.cmd == "prompts" and len(args.diff) not in (0, 2):
        ap.error("prompts takes no versions, or two to diff")
    if args.cmd in ("chat", "eval") and args.demo is None:
        ap.error(f"{args.cmd} needs --demo")
    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = args.judge_model
    {"link": link, "traces": traces, "prompts": prompts, "feedback": feedback, "expect": expect,
     "judges": judges, "gateway": gateway, "monitor": monitor, "chat": chat, "dataset": dataset,
     "eval": evaluate_version}[args.cmd](args)


if __name__ == "__main__":
    main()
