"""From OpenClaw traces to the first pieces of a self-improvement loop, with plain MLflow.

The connector puts every OpenClaw run in MLflow. This script adds three things on top, each a
single MLflow call you can point at in the UI afterwards:

  1. the update target: AGENTS.md registered in the Prompt Registry, with a `production` alias
  2. automated signal: a code check and a grounded judge, both built on the connector's `read`
     spans, run over the traces already there (mlflow.genai.evaluate with no predict_fn: nothing
     re-runs). The judge sees the files the agent actually read, so it can tell a documented fact
     from an invented one.
  3. the store: the failing traces become rows of an evaluation dataset, each tagged with its trace

    MLFLOW_TRACKING_URI=http://127.0.0.1:5050 python self_improvement_demo.py \\
        --experiment openclaw-demo --agents-md ~/odsc-live/workspace/AGENTS.md

Needs `mlflow[genai]` (the harness venv has it) and the judge model's API key, e.g. ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@scorer(name="read_knowledge")
def read_knowledge(trace) -> Feedback:
    """Did the agent read a policy file before answering? Only the connector's TOOL spans know."""
    reads = [s for s in trace.data.spans
             if s.span_type == SpanType.TOOL and s.name == "read" and "knowledge/" in str(s.inputs)]
    if reads:
        return Feedback(value=True, rationale=f"read {', '.join(str(s.inputs.get('path')) for s in reads)}")
    return Feedback(value=False, rationale="answered without reading anything in knowledge/")


def _last_message(value) -> str:
    """The connector records the conversation in chat format: {"messages": [{"role", "content"}]}."""
    if isinstance(value, dict) and value.get("messages"):
        return str(value["messages"][-1].get("content", ""))
    return str(value)


GROUNDED = """\
The inputs hold a customer's question and the policy files the support agent read before
answering: {{ inputs }}
The agent's answer: {{ outputs }}
Return true only if every factual claim in the answer is stated in those files, or the answer says
the documentation doesn't cover the question and escalates to a specialist. Return false if it
states a number, limit, certification or policy that the files don't contain, even with a caveat.
"""


def grounded_in_docs(judge_model: str):
    judge = make_judge(name="grounded_in_docs", instructions=GROUNDED, feedback_value_type=bool,
                       model=judge_model)

    @scorer(name="grounded_in_docs")
    def grounded(trace) -> Feedback:
        reads = [s for s in trace.data.spans if s.span_type == SpanType.TOOL and s.name == "read"]
        if not reads:
            return Feedback(value=False, rationale="no read spans: the answer has nothing behind it")
        files = {str(s.inputs.get("path")): str(s.outputs)[:6000] for s in reads}
        return judge(inputs={"question": _last_message(trace.data.spans[0].inputs), "files_read": files},
                     outputs=_last_message(trace.data.spans[0].outputs))

    return grounded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", default="openclaw-demo")
    ap.add_argument("--agents-md", type=Path, required=True, help="the AGENTS.md your OpenClaw agent runs with")
    ap.add_argument("--prompt-name", default="openclaw-agents-md")
    ap.add_argument("--dataset-name", default="openclaw-regressions")
    ap.add_argument("--judge-model", default=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "anthropic:/claude-sonnet-5-5"))
    args = ap.parse_args()

    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = args.judge_model
    exp = mlflow.set_experiment(args.experiment)

    # 1. The update target: every future change to AGENTS.md is a numbered version, and shipping
    #    one is an alias move.
    pv = mlflow.genai.register_prompt(name=args.prompt_name, template=args.agents_md.read_text(),
                                      commit_message="v1: the AGENTS.md OpenClaw runs today",
                                      tags={"author": "demo"})
    mlflow.genai.set_prompt_alias(args.prompt_name, alias="production", version=pv.version)
    print(f"1. Prompt Registry: prompts:/{args.prompt_name}@production -> v{pv.version}")

    # 2. Automated signal on the traces the connector already wrote.
    traces = [t for t in mlflow.search_traces(locations=[exp.experiment_id], return_type="list")
              if t.data.spans and t.data.spans[0].name == "openclaw_agent"]
    if not traces:
        raise SystemExit("no OpenClaw traces in this experiment yet: run `openclaw agent` first")
    print(f"2. Scoring {len(traces)} OpenClaw traces with read_knowledge (code) and grounded_in_docs "
          f"({args.judge_model}) ...")
    mlflow.genai.evaluate(data=traces, scorers=[read_knowledge, grounded_in_docs(args.judge_model)])

    failing = []
    for t in traces:
        verdicts = {a.name: a.value for a in mlflow.get_trace(t.info.trace_id).info.assessments}
        question = (t.info.request_preview or "")[:70]
        marks = "  ".join(f"{n}={'pass' if verdicts.get(n) in (True, 'yes') else 'FAIL'}"
                          for n in ("read_knowledge", "grounded_in_docs"))
        print(f"   {question:<70} {marks}")
        if any(verdicts.get(n) not in (True, "yes") for n in ("read_knowledge", "grounded_in_docs")):
            failing.append(t)

    # 3. The store: each failure becomes a test case. A person adds the correct answer
    #    (an expectation) in the MLflow UI; a judge's verdict never becomes one.
    try:
        ds = get_dataset(name=args.dataset_name)
    except Exception:  # noqa: BLE001 -- not created yet
        ds = create_dataset(name=args.dataset_name, experiment_id=[exp.experiment_id],
                            tags={"purpose": "regression"})
    ds.merge_records([{"inputs": {"question": t.info.request_preview or ""},
                       "tags": {"source": "openclaw_failure", "source_trace": t.info.trace_id}}
                      for t in failing])
    print(f"3. Evaluation dataset {args.dataset_name}: +{len(failing)} failing cases, each tagged with its trace")
    print(f"\nOpen the MLflow UI at {mlflow.get_tracking_uri()}: Traces (the assessments), Prompts, Datasets.")


if __name__ == "__main__":
    main()
