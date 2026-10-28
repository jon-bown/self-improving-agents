"""Update: turn accumulated signal into a candidate. None of these can promote anything.

    failure_report()  the signal, compacted: failing traces with input, output, rationales, tools
    analyze()         an LLM trace analyst: themes, one criterion per theme, revised instructions
    optimize()        MLflow's prompt optimizers: GEPA (thorough) or MetaPrompt (cheap, little data)
    coding_agent()    a coding agent of your choice edits the instructions file (SI_IMPROVER_COMMAND)

Each one sees the train split only. The holdout stays with the gate.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess

import mlflow
from mlflow.entities import AssessmentSourceType

from selfimprove import agents, config, instructions, lessons, models, traces


def failure_report(trs, limit: int = 60) -> list[dict]:
    """Every trace with at least one failing verdict, human-labeled failures first."""
    out = []
    for t in trs:
        fails = traces.failing_verdicts(t)
        if not fails:
            continue
        out.append({
            "trace_id": t.info.trace_id,
            "prompt_version": t.info.tags.get("si_prompt_version"),
            "input": traces.question_of(t),
            "output": traces.response_of(t)[:1200],
            "expected_response": traces.expectation(t),
            "tool_calls": [c["name"] for c in traces.tool_calls(t)],
            "failures": fails,
            "human_failed": any(f["source"].startswith(AssessmentSourceType.HUMAN) for f in fails),
        })
    out.sort(key=lambda r: (not r["human_failed"], -len(r["failures"])))
    return out[:limit]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:40] or "theme"


# --------------------------------------------------------------------------------------------
# The trace analyst (Part 2, section 6): reads failures, proposes themes, criteria, a fix
# --------------------------------------------------------------------------------------------
ANALYST_SYSTEM = """\
You are improving an AI agent whose behavior is defined by an instructions file. You are given
the current instructions and a report of failing traces with reviewers' and judges' rationales.

1. Group the failures into at most 5 themes. Many complaints about one bug are one theme. Cite
   representative trace ids. Mark a theme as reviewer noise when it is one reviewer's preference
   that other reviewers do not share.
2. For each theme write ONE pass/fail criterion as a requirement ("The response must ..."). It
   becomes an LLM-judge guideline, so it must describe the behavior reviewers want, not a proxy.
3. Propose revised instructions that fix the real themes. Change only what the evidence supports,
   keep the file about the same length, and keep the facts the agent should look up out of it
   (point the agent at its sources instead)."""

PROPOSAL_TOOL = models.tool("submit_proposal", "Submit the analysis and the revised instructions.", {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": {"type": "object", "properties": {
            "slug": {"type": "string", "description": "snake_case, e.g. stale_refund_policy"},
            "summary": {"type": "string"},
            "count": {"type": "integer"},
            "trace_ids": {"type": "array", "items": {"type": "string"}},
            "criterion": {"type": "string", "description": "The response must ..."},
            "is_reviewer_noise": {"type": "boolean"},
        }, "required": ["slug", "summary", "count", "trace_ids", "criterion", "is_reviewer_noise"]}},
        "instructions": {"type": "string", "description": "The full revised instructions file"},
        "change_summary": {"type": "string"},
    },
    "required": ["themes", "instructions", "change_summary"],
})


def analyze(report: list[dict], current: str) -> dict:
    user = (f"## Current instructions\n\n{current}\n\n"
            f"## Failure report ({len(report)} traces)\n\n```json\n{json.dumps(report, indent=1, default=str)}\n```")
    with mlflow.start_span("trace_analyst") as span:
        span.set_inputs({"failures": len(report)})
        out = models.structured(config.settings.judge_model, ANALYST_SYSTEM, user, PROPOSAL_TOOL)
        for t in out.get("themes", []):
            t["slug"] = slugify(t.get("slug", ""))
        span.set_outputs({"themes": [t["slug"] for t in out.get("themes", [])], "summary": out.get("change_summary")})
        return out


# --------------------------------------------------------------------------------------------
# Prompt optimizers (Part 3, sections 4 and 7)
# --------------------------------------------------------------------------------------------
def _as_float(v) -> float | None:
    if isinstance(v, list):
        vals = [x for x in map(_as_float, v) if x is not None]
        return sum(vals) / len(vals) if vals else None
    v = getattr(v, "value", v)
    if v is None:
        return None
    if isinstance(v, str):
        return 1.0 if v.strip().lower() in ("yes", "true", "pass") else 0.0
    return float(v)


def mean_of_scores(scores: dict) -> float:
    vals = [x for x in map(_as_float, scores.values()) if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


def optimize(train_rows: list[dict], scorers, kind: str = "gepa", ref="production", budget: int = 60):
    """Run optimize_prompts on the instructions. Returns (text, summary).

    predict_fn loads the prompt inside each call; during optimization MLflow swaps each
    candidate's text into the loaded prompt, and the harness renders it into its own workspace.
    """
    from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer

    base = instructions.load(ref)
    judge = config.settings.judge_model
    optimizer = (GepaPromptOptimizer(reflection_model=judge, max_metric_calls=budget, display_progress_bar=True)
                 if kind == "gepa" else MetaPromptOptimizer(reflection_model=judge))
    result = mlflow.genai.optimize_prompts(
        predict_fn=agents.predict_fn(int(base.version)), train_data=train_rows, prompt_uris=[base.uri],
        optimizer=optimizer, scorers=scorers, aggregation=mean_of_scores,
    )
    best = result.optimized_prompts[0]
    return best.template, f"{kind} {result.initial_eval_score or 0:.2f}->{result.final_eval_score or 0:.2f}"


# --------------------------------------------------------------------------------------------
# A coding agent of your choice (Claude Code, OpenClaw, Codex, Hermes, ...)
# --------------------------------------------------------------------------------------------
TASK = """\
You are improving an AI agent. This folder holds a copy of its files.

- {instructions_file} is the agent's instructions file. It is the ONLY file you may edit.
- failure_report.json lists recent responses that reviewers or judges failed, with their reasons.
- Any other files are the agent's reference material. Read them; do not edit them.

Edit {instructions_file} so those failures stop happening. Change only what the evidence
supports and keep it short. Do not paste reference facts into it; tell the agent where to look.
When you are done, reply with two or three sentences on what you changed and why.
"""


def coding_agent(report: list[dict], current: str, label: str) -> tuple[str, str]:
    """Run SI_IMPROVER_COMMAND in a scratch copy of the workspace; read back the edited file."""
    s = config.settings
    if not s.improver_command:
        raise ValueError("SI_IMPROVER=command needs SI_IMPROVER_COMMAND")
    src = agents.workspace_for(current)
    ws = s.state_dir / "improver" / label
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(src, ws)
    name = agents.instructions_filename()
    (ws / "failure_report.json").write_text(json.dumps(report, indent=1, default=str))
    (ws / "TASK.md").write_text(TASK.format(instructions_file=name))
    values = {"workspace": str(ws), "instructions_file": str(ws / name), "task_file": str(ws / "TASK.md")}
    # The template may use $(cat ...) so it runs in a shell; only harness-made paths are substituted.
    cmd = s.improver_command.format_map({k: shlex.quote(v) for k, v in values.items()})
    with mlflow.start_span("coding_agent") as span:
        span.set_inputs({"command": s.improver_command, "workspace": str(ws)})
        proc = subprocess.run(cmd, shell=True, cwd=ws, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"improver command exited {proc.returncode}: {proc.stderr[-1500:]}")
        text = lessons.join((ws / name).read_text(), current)  # the lessons block is not the editor's to change
        span.set_outputs({"stdout": proc.stdout[-2000:], "changed": text != current})
    return text, (proc.stdout.strip()[-400:] or "coding agent edit")
