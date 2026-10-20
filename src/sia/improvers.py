"""Update: things that turn accumulated signal into a candidate change.

    failure_report()   the signal, compacted: failing traces with question, answer, rationales,
                       and which policy files the agent actually read
    discover()         MLflow issue discovery (mlflow.genai discover_issues): failing traces ->
                       named issues with root causes, linked back to their traces
    analyze()          an LLM "trace analyst" -> themes, criteria, and a proposed change to
                       either AGENTS.md or a skill (skills/<name>/SKILL.md)
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
from sia.signals import knowledge_reads


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
                "files_read": knowledge_reads(t),
                "skills": t.info.tags.get("skills", ""),
                "failures": fails,
                "human_failed": any(f["source"].startswith(AssessmentSourceType.HUMAN) for f in fails),
            })
    out.sort(key=lambda r: (not r["human_failed"], -len(r["failures"])))
    return out[:limit]


# --------------------------------------------------------------------------------------------
# MLflow issue discovery: let MLflow find the themes
# --------------------------------------------------------------------------------------------
def _existing_verdicts():
    """Triage scorer for discover_issues that reuses the verdicts already on each trace (human
    labels, judges, code and tool checks) instead of paying an LLM to re-judge every trace."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="existing_verdicts")
    def existing_verdicts(trace) -> Feedback:
        fails = [f"{a.name}: {(a.rationale or '')[:200]}" for a in trace.info.assessments
                 if a.feedback is not None and a.valid is not False
                 and str(a.value).lower() in ("false", "no", "0", "0.0")]
        reads = ", ".join(dict.fromkeys(knowledge_reads(trace))) or "none"
        return Feedback(value=not fails,
                        rationale=("; ".join(fails) or "No failing verdicts.") + f" Policy files read: {reads}.")

    return existing_verdicts


def discover(traces, max_issues: int = 5) -> dict:
    """Run MLflow issue discovery over these traces. Returns {"issues": [...], "run_id", "summary"}.

    Each issue becomes a theme: a name, a description, root causes, a severity, and the trace ids
    MLflow linked it to (it logs an assessment named after the issue on every affected trace).
    """
    import mlflow
    from mlflow.genai.discovery.pipeline import discover_issues

    result = discover_issues(traces=traces, scorers=[_existing_verdicts()], model=config.JUDGE_MODEL,
                             max_issues=max_issues)
    issues = []
    for issue in result.issues:
        if str(issue.severity).lower().endswith("not_an_issue"):
            continue
        linked = [t.info.trace_id for t in traces
                  if any(a.name == issue.issue_id for a in mlflow.get_trace(t.info.trace_id).info.assessments)]
        issues.append({
            "issue_id": issue.issue_id, "name": issue.name.removeprefix("Issue: "),
            "description": issue.description, "root_causes": list(issue.root_causes or []),
            "severity": str(issue.severity).split(".")[-1].lower(), "categories": list(issue.categories or []),
            "trace_ids": linked[:5], "trace_count": issue.trace_count or len(linked),
        })
    return {"issues": issues, "run_id": result.triage_run_id, "summary": result.summary}


def issues_to_themes(issues: list[dict]) -> list[dict]:
    """Fallback when no analyst rewrites them: one theme (and judge criterion) per MLflow issue."""
    return [{"slug": slugify(i["name"]), "summary": i["description"], "count": i["trace_count"],
             "trace_ids": i["trace_ids"],
             "criterion": f"The response must not exhibit this problem: {i['description']}",
             "is_reviewer_noise": False} for i in issues]


# --------------------------------------------------------------------------------------------
# Level 3: an LLM trace analyst proposes; a person approves
# --------------------------------------------------------------------------------------------
ANALYST_SYSTEM = """\
You are a senior engineer improving a customer-support agent that runs on OpenClaw. Its behavior
is defined by AGENTS.md (always in context) plus optional skills: skills/<name>/SKILL.md files
whose name and description are listed to the agent, which reads one when a question matches it.
It also has read access to knowledge/*.md (the policy source of truth). You are given the current
AGENTS.md, the current skills, and a report of failing production traces with the reviewers'
and judges' rationales and the policy files the agent read (files_read).

1. Group the failures into at most 5 themes. Merge duplicates: many complaints about one bug are
   one theme. Cite representative trace ids. Ignore failures that look like reviewer noise
   (e.g. one reviewer's style preference that others do not share) and say so.
2. For each theme, write ONE pass/fail criterion as a requirement ("The response must ...").
   It will become an LLM-judge guideline, so it must describe the behavior reviewers want,
   not a proxy for it.
3. Choose ONE update target and propose the change. Change only what the evidence supports.
   - "agents_md": global behavior that applies to every answer (lead with the answer, treat
     knowledge/ as the source of truth, never guess). Return the full revised AGENTS.md.
   - "skill": a reusable procedure for one class of questions (e.g. how to handle escalations).
     Prefer this when the failures cluster in one kind of question: AGENTS.md stays short and
     the procedure is loaded only when needed. Return skill.name (kebab-case; reuse an existing
     name to update that skill), skill.description (<= 160 chars, says WHEN to use it), and
     skill.body (markdown instructions). Return AGENTS.md unchanged.
   Either way, do not paste policy facts -- point the agent at the right knowledge/ file, so the
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
            "update_target": {"type": "string", "enum": ["agents_md", "skill"]},
            "agents_md": {"type": "string", "description": "The full proposed AGENTS.md"},
            "skill": {
                "type": "object",
                "description": "Only when update_target is skill",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                               "body": {"type": "string"}},
            },
            "change_summary": {"type": "string"},
        },
        "required": ["themes", "update_target", "agents_md", "change_summary"],
    },
)


def analyze(report: list[dict], current_agents_md: str, skills: dict[str, str] | None = None,
            issues: list[dict] | None = None, target: str | None = None) -> dict:
    """Ask the judge-tier model (any provider) for themes, criteria, and a proposed change.

    issues: MLflow-discovered issues to use as the themes (instead of clustering from scratch).
    target: force "agents_md" or "skill" (default: the analyst chooses).
    """
    parts = [f"## Current AGENTS.md\n\n{current_agents_md}"]
    parts.append("## Current skills\n\n" + ("\n\n".join(f"### skills/{n}/SKILL.md\n{t}" for n, t in (skills or {}).items())
                                            or "(none)"))
    if issues:
        parts.append("## Issues discovered by MLflow (use these as your themes, one theme per issue)\n\n"
                     f"```json\n{json.dumps(issues, indent=1)}\n```")
    parts.append(f"## Failure report ({len(report)} traces)\n\n```json\n{json.dumps(report, indent=1)}\n```")
    if target:
        parts.append(f"The update target for this proposal must be: {target}.")
    out = llm.structured(config.JUDGE_MODEL, ANALYST_SYSTEM, "\n\n".join(parts), PROPOSAL_TOOL)
    out.setdefault("update_target", "agents_md")
    if out["update_target"] == "skill" and not (out.get("skill") or {}).get("body"):
        out["update_target"] = "agents_md"  # a skill proposal without a body is not a proposal
    if out["update_target"] == "skill":
        out["skill"]["name"] = re.sub(r"[^a-z0-9-]+", "-", out["skill"]["name"].lower()).strip("-")[:48]
        out["skill"]["description"] = out["skill"].get("description", "")[:160]
    return out


def offline_analyze(report: list[dict], current_agents_md: str, skills: dict[str, str] | None = None,
                    target: str | None = None) -> dict:
    """No-LLM stand-in for analyze(), so the loop can be rehearsed without API keys.

    Groups failures by rationale keywords. Proposes the Level 2 fix (an AGENTS.md change) if the
    agent is not using knowledge/ at all. Otherwise the failures are about a topic the
    instructions don't cover, and it proposes a *skill* for that topic (or, with
    target="agents_md", a pointer appended to AGENTS.md).
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
        return {"themes": themes, "update_target": "agents_md",
                "agents_md": (SCENARIO_DIR / "fixes" / "level2_manual_AGENTS.md").read_text(),
                "change_summary": "offline analyst: the agent ignores knowledge/; rewrite AGENTS.md"}
    covered = text + "".join((skills or {}).values())
    missing = sorted({r["topic"] for r in report if r["topic"] and f"{r['topic']}.md" not in covered})
    if not missing:
        return {"themes": themes, "update_target": "agents_md", "agents_md": text,
                "change_summary": "offline analyst: nothing to change"}
    topic = missing[0]
    if target == "agents_md":
        text = text.rstrip() + f"\n- For {topic.replace('_', ' ')} questions, read `knowledge/{topic}.md`.\n"
        return {"themes": themes, "update_target": "agents_md", "agents_md": text,
                "change_summary": f"offline analyst: point AGENTS.md at knowledge/{topic}.md"}
    name = f"handle-{topic.replace('_', '-')}"
    words = topic.replace("_", " ")
    body = (f"# Handle {words} questions\n\n"
            f"Use this skill when the question is about {words}: anything the billing, account access, "
            f"and product how-to policies do not cover.\n\n"
            f"1. Read `knowledge/{topic}.md`. It is the source of truth.\n"
            f"2. Follow it exactly, including any timelines or promises it allows (and the ones it forbids).\n"
            f"3. Lead with what will happen next for the customer.\n")
    return {"themes": themes, "update_target": "skill", "agents_md": text,
            "skill": {"name": name, "description": f"How to answer {words} questions (escalations, "
                                                    f"requests outside written policy)."[:160], "body": body},
            "change_summary": f"offline analyst: {words} failures need a procedure; propose skill {name}"}


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
