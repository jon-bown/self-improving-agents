"""Update: things that turn accumulated signal into a candidate change.

    failure_report()   the signal, compacted: failing traces with question, answer, rationales
    analyze()          an LLM "trace analyst" -> themes, criteria, and a proposed AGENTS.md
    openclaw_rewrite() OpenClaw edits its *own* AGENTS.md from the same report
    gepa_optimize()    MLflow's GEPA prompt optimizer, scored by our scorers (Level 4+)

Every improver returns proposed text. None of them can promote anything: that is the gate's job.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from mlflow.entities import AssessmentSourceType

from sia import config, llm, tracking


# --------------------------------------------------------------------------------------------
# The signal, compacted
# --------------------------------------------------------------------------------------------
def failure_report(traces, limit: int = 60) -> list[dict]:
    """Every trace with at least one failing verdict (human, judge, or code), with rationales."""
    out = []
    for t in traces:
        fails = []
        for a in t.info.assessments:
            if a.feedback is None or a.valid is False:
                continue
            if str(a.value).lower() in ("false", "no", "0", "0.0"):
                fails.append({"name": a.name, "source": f"{a.source.source_type}:{a.source.source_id}",
                              "rationale": (a.rationale or "")[:400]})
        if fails:
            out.append({
                "trace_id": t.info.trace_id,
                "topic": t.info.tags.get("topic"),
                "prompt_version": t.info.tags.get("prompt_version"),
                "question": tracking.question_of(t),
                "answer": tracking.response_of(t)[:800],
                "expected_response": tracking.expectation(t),
                "failures": fails,
                "human_failed": any(f["source"].startswith(AssessmentSourceType.HUMAN) for f in fails),
            })
    out.sort(key=lambda r: (not r["human_failed"], -len(r["failures"])))
    return out[:limit]


# --------------------------------------------------------------------------------------------
# Level 3: an LLM trace analyst proposes; a person approves
# --------------------------------------------------------------------------------------------
ANALYST_SYSTEM = """\
You are a senior engineer improving a customer-support agent. The agent's behavior is defined
entirely by its AGENTS.md file; it also has read access to knowledge/*.md (the policy source of
truth). You are given the current AGENTS.md and a report of failing production traces with the
reviewers' and judges' rationales.

1. Group the failures into at most 5 themes. Merge duplicates: many complaints about one bug are
   one theme. Cite representative trace ids. Ignore failures that look like reviewer noise
   (e.g. one reviewer's style preference that others do not share) and say so.
2. For each theme, write ONE pass/fail criterion as a requirement ("The response must ...").
   It will become an LLM-judge guideline, so it must describe the behavior reviewers want,
   not a proxy for it.
3. Propose a revised AGENTS.md that fixes the themes. Change only what the evidence supports.
   Do not paste policy facts into AGENTS.md -- point the agent at knowledge/ instead, so the
   next policy change does not require another prompt change.
Call the submit_proposal tool exactly once."""

PROPOSAL_TOOL = llm.tool(
    "submit_proposal",
    "Submit the analysis and the proposed AGENTS.md.",
    {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string", "description": "snake_case, e.g. stale_refund_policy"},
                        "summary": {"type": "string"},
                        "count": {"type": "integer"},
                        "trace_ids": {"type": "array", "items": {"type": "string"}},
                        "criterion": {"type": "string", "description": "The response must ..."},
                        "is_reviewer_noise": {"type": "boolean"},
                    },
                    "required": ["slug", "summary", "count", "trace_ids", "criterion", "is_reviewer_noise"],
                },
            },
            "agents_md": {"type": "string", "description": "The full proposed AGENTS.md"},
            "change_summary": {"type": "string"},
        },
        "required": ["themes", "agents_md", "change_summary"],
    },
)


def analyze(report: list[dict], current_agents_md: str) -> dict:
    """Ask the judge-tier model (any provider) for themes, criteria, and a revised AGENTS.md."""
    user = (f"## Current AGENTS.md\n\n{current_agents_md}\n\n"
            f"## Failure report ({len(report)} traces)\n\n```json\n{json.dumps(report, indent=1)}\n```")
    return llm.structured(config.JUDGE_MODEL, ANALYST_SYSTEM, user, PROPOSAL_TOOL)


def offline_analyze(report: list[dict], current_agents_md: str) -> dict:
    """No-LLM stand-in for analyze(), so the loop can be rehearsed without API keys.

    Groups failures by rationale keywords. Proposes the Level 2 fix if the agent is not using
    knowledge/ at all, and otherwise adds a pointer to any knowledge file whose topic is failing.
    """
    from sia.scenario import SCENARIO_DIR

    buckets = {
        "stale_policy": ("outdated", "The response must state current policy (14-day refunds, 15-minute reset links)."),
        "buried_answer": ("disclaimer", "The response must state the direct answer in its first sentence."),
        "missing_facts": ("Missing a key fact", "The response must include the specific facts from the policy files."),
    }
    themes = []
    for slug, (needle, criterion) in buckets.items():
        ids = [r["trace_id"] for r in report if any(needle in f["rationale"] for f in r["failures"])]
        if ids:
            themes.append({"slug": slug, "summary": needle, "count": len(ids), "trace_ids": ids[:3],
                           "criterion": criterion, "is_reviewer_noise": False})
    text = current_agents_md
    if "knowledge/" not in text:
        text = (SCENARIO_DIR / "fixes" / "level2_manual_AGENTS.md").read_text()
    for topic in sorted({r["topic"] for r in report if r["topic"]}):
        if f"{topic}.md" not in text:
            text = text.rstrip() + f"\n- For {topic.replace('_', ' ')} questions, read `knowledge/{topic}.md`.\n"
    return {"themes": themes, "agents_md": text, "change_summary": "offline analyst (rule-based)"}


# --------------------------------------------------------------------------------------------
# Level 5 option: OpenClaw rewrites its own instructions
# --------------------------------------------------------------------------------------------
SELF_EDIT_TASK = """\
You are improving yourself. This directory is a copy of your own workspace.
- AGENTS.md is your instruction file. knowledge/ holds the current policies (source of truth).
- failure_report.json lists recent answers of yours that reviewers or judges failed, with reasons.

Edit AGENTS.md so those failures stop happening. Rules:
- Change only what the evidence supports; keep it short.
- Do not copy policy facts into AGENTS.md; tell yourself to read knowledge/ instead.
- Do not edit anything other than AGENTS.md.
When done, reply with a 2-3 sentence summary of what you changed and why."""


def openclaw_rewrite(report: list[dict], current_agents_md: str, workdir: Path) -> dict:
    from sia.workspace import build_workspace

    ws = build_workspace(current_agents_md, workdir.name)
    (ws / "failure_report.json").write_text(json.dumps(report, indent=1))
    cmd = [config.OPENCLAW_BIN, "agent", "exec", SELF_EDIT_TASK, "--json", "--cwd", str(ws),
           "--isolated", "--auth-env-only", "--model", llm.openclaw_model(config.JUDGE_MODEL)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=dict(os.environ))
    if proc.returncode != 0:
        raise RuntimeError(f"openclaw agent exec failed ({proc.returncode}): {proc.stderr[-1500:]}")
    payload = json.loads(proc.stdout)
    new_text = (ws / "AGENTS.md").read_text()
    shutil.rmtree(ws / ".openclaw", ignore_errors=True)
    return {"themes": [], "agents_md": new_text,
            "change_summary": payload.get("final") or "", "tool_summary": payload.get("toolSummary")}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:40]


# --------------------------------------------------------------------------------------------
# Level 4: improve the verifier, then let an optimizer use it
# --------------------------------------------------------------------------------------------
def align_judge(judge, traces):
    """Align an LLM judge to the human labels on these traces.

    MemAlign (MLflow's default) needs an embedding model. config.EMBEDDING_MODEL is filled in
    automatically for providers that have one (OpenAI, Gemini, Mistral) or from
    SIA_EMBEDDING_MODEL. Otherwise (e.g. Anthropic, Groq) we use SIMBA, which needs only the
    judge LLM and at least 10 labeled traces.
    """
    from mlflow.genai.judges.optimizers import MemAlignOptimizer, SIMBAAlignmentOptimizer

    embed = config.EMBEDDING_MODEL
    if embed and llm.has_credentials(embed):
        optimizer = MemAlignOptimizer(reflection_lm=config.JUDGE_MODEL, embedding_model=embed)
    else:
        optimizer = SIMBAAlignmentOptimizer(model=config.JUDGE_MODEL)
    return judge.align(traces, optimizer=optimizer)


def judge_agreement(judge, traces) -> float:
    """Fraction of traces where the judge's verdict equals the latest human 'quality' label."""
    hits = n = 0
    for t in traces:
        human = tracking.human_feedback(t, judge.name)
        if human is None:
            continue
        fb = judge(trace=t)
        n += 1
        hits += bool(fb.value) == bool(human.value)
    return hits / n if n else float("nan")


def _as_float(v) -> float | None:
    if isinstance(v, list):
        vals = [x for x in map(_as_float, v) if x is not None]
        return sum(vals) / len(vals) if vals else None
    v = getattr(v, "value", v)
    if v is None:
        return None
    if isinstance(v, str):
        return 1.0 if v.lower() in ("yes", "true", "pass") else 0.0
    return float(v)


def mean_of_scores(scores: dict) -> float:
    vals = [x for x in map(_as_float, scores.values()) if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


def gepa_optimize(train_rows: list[dict], scorers, prompt_ref="production", max_metric_calls: int = 60):
    """Run MLflow's GEPA optimizer on AGENTS.md. Returns the new (registered) prompt version.

    predict_fn loads the prompt *inside* the call: during optimization MLflow patches
    PromptVersion.template to return each candidate, and run_agent renders whatever it gets
    into a fresh workspace. The optimizer never touches production.
    """
    import mlflow
    from mlflow.genai.optimize import GepaPromptOptimizer

    from sia.agent import load_agents_md, run_agent

    base = load_agents_md(prompt_ref)

    def predict_fn(question: str) -> str:
        return run_agent(question, prompt_ref=int(base.version))

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_rows,
        prompt_uris=[base.uri],
        optimizer=GepaPromptOptimizer(reflection_model=config.JUDGE_MODEL, max_metric_calls=max_metric_calls,
                                      display_progress_bar=True),
        scorers=scorers,
        aggregation=mean_of_scores,
    )
    return result
