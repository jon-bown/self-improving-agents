"""Level 0 · Instrument: send a week of "customer traffic" through the agent and trace it.

Each question becomes one MLflow trace:

    acme_support_agent  (AGENT)      tags: topic, backend, prompt_version, source=production
      └─ openclaw.agent (CHAIN)      tool-call count, tools used, token usage
         (or LLM + TOOL spans with SIA_BACKEND=direct)

Open the MLflow UI -> Experiments -> acme-support-agent -> Traces and read a few.
Nothing is judged yet; this is raw signal waiting to be interpreted.

    python levels/level0_instrument/02_first_traces.py [--repeat 2]
"""

import argparse
import random

import mlflow

from sia import tracking
from sia.agent import run_agent
from sia.scenario import load_cases

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=int, default=1, help="send each question this many times")
parser.add_argument("--split", default="traffic")
args = parser.parse_args()

tracking.setup()
questions = [c.question for c in load_cases(args.split)] * args.repeat
random.Random(7).shuffle(questions)

for i, q in enumerate(questions, 1):
    try:
        answer = run_agent(q, source="production")
    except Exception as e:  # keep going; the failed trace is signal too
        answer = f"ERROR: {e}"
    print(f"[{i:>2}/{len(questions)}] {q}\n        -> {answer[:140].replace(chr(10), ' ')}")

mlflow.flush_trace_async_logging()
print(f"\n{len(questions)} traces logged. Next: levels/level1_signal/")
