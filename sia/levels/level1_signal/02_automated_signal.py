"""Level 1 · Signal from code and judges: score the traces we already have, in place.

mlflow.genai.evaluate() with traces as data and no predict_fn does not re-run the agent. It
attaches each scorer's verdict to the existing trace as an assessment (source CODE or
LLM_JUDGE), right next to the human labels from 01_human_feedback.py.

Afterwards each trace carries three kinds of signal: human, deterministic, and LLM judge.
The table printed at the end is the first "where is it failing?" view, sliced by topic.

    python levels/level1_signal/02_automated_signal.py
"""

from collections import defaultdict

import mlflow

from sia import signals, tracking

tracking.setup()
traces = tracking.agent_traces("tags.prompt_version = '1'")
print(f"Scoring {len(traces)} production traces (prompt v1)")

scorers = signals.code_scorers() + signals.judge_scorers()
if signals.config.HAS_LLM:
    scorers.append(signals.quality_judge())  # un-aligned for now; Level 4 aligns it
with mlflow.start_run(run_name="level1: score production traces"):
    result = mlflow.genai.evaluate(data=traces, scorers=scorers)

# ---- failure rate by topic x scorer ---------------------------------------------------------
df = result.result_df
topic_by_trace = {t.info.trace_id: t.info.tags.get("topic", "?") for t in traces}
fails = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for _, row in df.iterrows():
    topic = topic_by_trace.get(row.get("trace_id"), "?")
    for s in scorers:
        v = row.get(f"{s.name}/value")
        if v is None:
            continue
        ok = str(v).lower() in ("yes", "true", "1", "1.0")
        fails[s.name][topic][0] += 0 if ok else 1
        fails[s.name][topic][1] += 1

topics = sorted({t for s in fails.values() for t in s})
print(f"\nfailure rate   {''.join(f'{t:>16}' for t in topics)}")
for name, by_topic in fails.items():
    cells = "".join(f"{(f / n if n else 0):>16.0%}" for f, n in (by_topic.get(t, [0, 0]) for t in topics))
    print(f"{name:<15}{cells}")
print("\nNext: 03_signal_quality.py -- can we trust these verdicts?")
