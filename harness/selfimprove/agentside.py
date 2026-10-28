"""`selfimprove self ...`: the commands the agent itself runs (the self-improving-with-mlflow skill).

    status   what the agent may know about itself: version, last gate, live failure rate
    note     the user corrected the agent: record it on that trace and ask a person to confirm
    recall   the confirmed lessons (the pack in production's instructions) that fit a request
    propose  a change to itself: an AGENTS.md candidate for the gate, or a skill change for a person

The agent is the thing being improved, so the guardrails live here, not in the skill's text:
a note is `self_reported` feedback (never `quality`, so nothing reads it as a verdict) plus a
`self_report` item in the human queue; only a person accepting it creates a label. Every command
returns one small JSON-able dict. A refusal is {"ok": False, "refused": "<why>"}, not an error.
Design: docs/design/self-improving-with-mlflow.md.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
from pathlib import Path

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from selfimprove import config, instructions, lessons, store, traces

AUTHOR = "openclaw-self"
SESSION = "mlflow.trace.session"
_MAX_TEXT = 500


def _refuse(why: str, **extra) -> dict:
    return {"ok": False, "refused": why, **extra}


def _clip(text, n: int = _MAX_TEXT) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------------------
def _last_gate() -> dict | None:
    """The newest finished `gate` job. The inner comparison run has the same si.job tag but no outcome."""
    runs = mlflow.search_runs(filter_string="tags.`si.job` = 'gate'", order_by=["start_time DESC"],
                              max_results=20, output_format="list")
    run = next((r for r in runs if r.data.tags.get("si.outcome") not in (None, "nothing_to_do")), None)
    if run is None:
        return None
    return {"candidate": run.data.params.get("candidate"), "result": run.data.tags["si.outcome"],
            "summary": _clip(run.data.tags.get("si.message"), 300), "at": run.info.start_time}


def _open_candidates(production: int) -> list[dict]:
    versions = mlflow.MlflowClient().search_prompt_versions(config.settings.prompt_name)
    return [{"version": int(v.version), "author": (v.tags or {}).get("author")}
            for v in sorted(versions, key=lambda v: int(v.version))
            if int(v.version) > production and (v.tags or {}).get("status") == "candidate"]


def synced() -> list[dict]:
    """`self sync` history, oldest first: which version was written to the live workspace, when."""
    path = config.settings.state_dir / "self_sync.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _live_rate(version: int, since: int | None) -> tuple[float, int]:
    """Harness traffic carries si_prompt_version. Deployed (connector) traffic doesn't (design D5),
    so it's credited to whichever version `self sync` last wrote before the trace started."""
    from selfimprove import signals
    from selfimprove.workflows.ship import live_failure_rate

    rate, judged = live_failure_rate(version, since_ms=since)
    log = synced()
    if judged or not log or log[-1]["version"] != version:
        return rate, judged
    trs = traces.production(since_ms=log[-1]["written_at"])
    must = signals.must_pass()
    seen = [t for t in trs if traces.has_verdict(t, must)]
    return (sum(traces.failed(t, must) for t in seen) / len(seen) if seen else 0.0), len(seen)


def status() -> dict:
    if not instructions.exists():
        return _refuse("no instructions registered yet (a person runs `selfimprove init`)")
    prod = instructions.load("production")
    tags = prod.tags or {}
    version = int(prod.version)
    since = int(tags.get("promoted_at", 0)) or None
    rate, judged = _live_rate(version, since)
    log = synced()
    reports = [i for i in store.queue_items("open") if i["kind"] == "self_report"]
    holdout_qs = store.holdout_questions()
    confirmed = [{"trace_id": i["trace_id"], "interpretation": _clip(i.get("interpretation"), 200)}
                 for i in store.queue_items(status=None)
                 if i["kind"] == "self_report" and i.get("status") in ("accepted", "edited")
                 and not store.is_holdout(store.question_of(i.get("record") or {}), holdout_qs)][-10:]
    return {
        "ok": True,
        "prompt": {"name": config.settings.prompt_name, "production": version,
                   "replaced": int(tags["replaced_version"]) if tags.get("replaced_version") else None,
                   "promoted_at": since, "promoted_by": tags.get("promoted_by") or tags.get("author"),
                   "promoted_because": _clip(tags.get("promoted_because"), 200) or None},
        "last_gate": _last_gate(),
        "live": {"failure_rate": round(rate, 4), "judged_traces": judged},
        "open_candidates": _open_candidates(version),
        "pending_self_reports": len(reports),
        "confirmed_reports": confirmed,   # your notes a person confirmed: evidence `propose` accepts
        "lessons": len(lessons.parse(prod.template)),
        "synced": ({"version": log[-1]["version"], "at": log[-1]["written_at"]} if log else None),
    }


# --------------------------------------------------------------------------------------------
# note
# --------------------------------------------------------------------------------------------
def _session_filter(session: str) -> str:
    return f"metadata.`{SESSION}` = '{session.replace(chr(39), '')}'"


def _resolve_trace(trace: str, session: str | None):
    """A production trace by id, or "previous": the newest one in this session (or recent window).

    The turn the agent is answering now is still running, so its trace isn't exported yet: the
    newest exported production trace is the answer the user is correcting.
    """
    s = config.settings
    if trace != "previous":
        hits = traces.production(f"trace.request_id = '{trace.replace(chr(39), '')}'", max_results=1)
        return hits[0] if hits else None
    since = int((time.time() - s.self_note_window_min * 60) * 1000)
    hits = traces.production(_session_filter(session) if session else None, since_ms=since, max_results=1)
    return hits[0] if hits else None


def _reports_in_session(session: str) -> int:
    day_ago = time.time() - 24 * 3600
    return sum(1 for i in store.queue_items(status=None)
               if i["kind"] == "self_report" and (i.get("session") or "") == session
               and i.get("created", 0) >= day_ago)


def note(trace: str, user_said: str, interpretation: str, expected: str | None = None,
         session: str | None = None) -> dict:
    """Record that the user corrected the answer in `trace` ("previous" or a trace id).

    Writes `self_reported` feedback on the trace and a `self_report` queue item. `expected`, if the
    user stated the right answer, goes only into the queue item: a person decides whether it
    becomes an expectation (a judge's or an agent's reading is never one).
    """
    s = config.settings
    if not user_said.strip() or not interpretation.strip():
        return _refuse("give the user's exact words (--user-said) and what you got wrong (--interpretation)")
    t = _resolve_trace(trace, session)
    if t is None:
        return _refuse("no production answer found to attach this to" + (
            "" if trace != "previous" else f" (looked {s.self_note_window_min:g} minutes back)"))
    trace_id = t.info.trace_id
    if traces.human_feedback(t) is not None:
        return _refuse("already labeled: a person has graded that answer", trace_id=trace_id)
    if traces.feedback(t, traces.SELF_REPORTED) is not None or any(
            i["kind"] == "self_report" and i.get("trace_id") == trace_id for i in store.queue_items(status=None)):
        return _refuse("already reported: a note for that answer is waiting for a person", trace_id=trace_id)
    session = session or t.info.trace_metadata.get(SESSION) or ""
    if _reports_in_session(session) >= s.self_note_max_per_session:
        return _refuse(f"note limit for this session ({s.self_note_max_per_session} a day); "
                       "your owner can review the conversation instead", trace_id=trace_id)

    user_said, interpretation = _clip(user_said), _clip(interpretation)
    mlflow.log_feedback(trace_id=trace_id, name=traces.SELF_REPORTED, value=False, rationale=interpretation,
                        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=AUTHOR),
                        metadata={"user_said": user_said})
    record = store.record(traces.question_of(t), {traces.EXPECTED: _clip(expected, 2000) if expected else None},
                          source="self_report", author=AUTHOR, source_trace=trace_id)
    item_id = store.enqueue("self_report", f"user said {user_said!r}; agent: {interpretation}",
                            trace_id=trace_id, record=record, user_said=user_said,
                            interpretation=interpretation, session=session)
    return {"ok": True, "trace_id": trace_id, "queue_item": item_id,
            "next": "a person confirms or rejects it (`selfimprove queue accept|reject " + item_id + "`)"}


# --------------------------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------------------------
def _holdout_traces() -> set[str]:
    return {r["tags"].get("source_trace") for r in store.records()
            if r["tags"].get("split") == "holdout" and r["tags"].get("source_trace")}


def recall(request: str, k: int = 3) -> dict:
    """The lessons in production's pack that fit `request`. Never reads traces or the dataset rows.

    A lesson whose evidence has since moved into the holdout is dropped here as well as at build
    time, so the pack can't leak a held-out case even if it was built before the split changed.
    """
    if not request.strip():
        return _refuse("give the user's request, in their words")
    if not instructions.exists():
        return _refuse("no instructions registered yet (a person runs `selfimprove init`)")
    prod = instructions.load("production")
    pack = lessons.parse(prod.template)
    hidden = _holdout_traces() if pack else set()
    pack = [les for les in pack if not hidden.intersection(les.get("traces") or [])]
    hits = lessons.match(request, pack, k=max(1, min(k, 5)))
    return {"ok": True, "source": f"lessons@{prod.version}",
            "lessons": [{"id": h["id"], "lesson": _clip(h["lesson"], 300), "evidence": int(h.get("evidence") or 0),
                         "confirmed_by": "human"} for h in hits]}


# --------------------------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------------------------
_SKILL = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _evidence(ids: list[str]) -> tuple[list, dict | None]:
    """The evidence traces, or a refusal: each must be production traffic a person marked failed."""
    s = config.settings
    ids = list(dict.fromkeys(i.strip() for i in ids if i.strip()))
    if len(ids) < s.self_propose_min_evidence:
        return [], _refuse(f"evidence needs at least {s.self_propose_min_evidence} confirmed failures "
                           "(one correction is not a pattern)")
    found, holdout_qs = [], store.holdout_questions()
    for tid in ids:
        t = _resolve_trace(tid, None)
        if t is None:
            return [], _refuse(f"no production answer with id {tid}")
        if store.is_holdout(traces.question_of(t), holdout_qs):
            return [], _refuse("evidence includes held-out cases: drop the proposal")
        fb = traces.human_feedback(t)
        if fb is None or not traces.is_fail(fb.value):
            return [], _refuse(f"evidence has no confirmed failure: nobody has marked {tid} wrong yet")
        found.append(t)
    return found, None


def _growth_refusal(new: str, old: str) -> dict | None:
    s = config.settings
    growth = len(new) / max(len(old), 1)
    if growth > s.max_prompt_growth:
        return _refuse(f"too long: {growth:.1f}x the current length (limit {s.max_prompt_growth:g}x). "
                       "Shorten it only if the extra text was padding")
    return None


def propose(target: str, file: str, why: str, evidence: list[str]) -> dict:
    """Submit a change to the agent itself. It never promotes or applies anything.

    AGENTS.md becomes a prompt candidate (author openclaw-self) that waits for the gate. A skill
    change goes to the human queue: skills, tools and code never ship automatically (rule 7).
    """
    if not why.strip():
        return _refuse("say why (--why): what the evidence shows and what the change fixes")
    path = Path(file).expanduser()
    if path.is_dir():
        path = path / "SKILL.md"
    if not path.is_file():
        return _refuse(f"no file at {file}")
    text = path.read_text()
    if not text.strip():
        return _refuse("the proposed file is empty")
    found, refusal = _evidence(evidence)
    if refusal:
        return refusal
    trace_ids = [t.info.trace_id for t in found]
    why = _clip(why)
    if target == "AGENTS.md":
        return _propose_instructions(text, why, trace_ids)
    if target.startswith("skill:") and _SKILL.match(target[6:]):
        return _propose_skill(target[6:], text, why, trace_ids)
    return _refuse("target is AGENTS.md or skill:<name> (lower-case letters, digits and dashes)")


def _propose_instructions(text: str, why: str, trace_ids: list[str]) -> dict:
    if not instructions.exists():
        return _refuse("no instructions registered yet (a person runs `selfimprove init`)")
    if lessons.block(text):
        return _refuse("lessons are rebuilt by your owner (`self lessons`): propose AGENTS.md without them")
    prod = instructions.load("production")
    production = int(prod.version)
    if any(c["author"] == AUTHOR for c in _open_candidates(production)):
        return _refuse("open proposal exists: one of yours is already waiting for the gate")
    candidate = lessons.join(text, prod.template)
    if candidate.strip() == prod.template.strip():
        return _refuse("no change: that is the production AGENTS.md")
    refusal = _growth_refusal(candidate, prod.template)
    if refusal:
        return refusal
    version = instructions.register_candidate(candidate, f"proposed by the agent: {why}", author=AUTHOR,
                                              base_version=production, evidence=",".join(trace_ids))
    out = {"ok": True, "target": "AGENTS.md", "candidate_version": version,
           "next": "the gate decides (`selfimprove gate`); you can't promote it"}
    if config.settings.self_propose_gate:
        import contextlib
        import sys

        from selfimprove.workflows.ship import run_gate

        with contextlib.redirect_stdout(sys.stderr):  # stdout is one JSON object
            result = run_gate(version)
        out["gate"] = {"result": result.outcome, "summary": _clip(result.message, 300)}
    return out


def _skill_file(name: str) -> Path | None:
    ws = config.settings.self_workspace
    return ws / "skills" / name / "SKILL.md" if ws else None


def _propose_skill(name: str, text: str, why: str, trace_ids: list[str]) -> dict:
    if any(i["kind"] == "skill_proposal" and i.get("skill") == name for i in store.queue_items("open")):
        return _refuse(f"open proposal exists: a change to skill {name} is already waiting for a person")
    current_file = _skill_file(name)
    current = current_file.read_text() if current_file and current_file.is_file() else ""
    if current and text.strip() == current.strip():
        return _refuse(f"no change: that is the current skill {name}")
    if current:
        refusal = _growth_refusal(text, current)
        if refusal:
            return refusal
    diff = "".join(difflib.unified_diff(current.splitlines(keepends=True), text.splitlines(keepends=True),
                                        fromfile=f"skills/{name}/SKILL.md", tofile=f"proposed/{name}/SKILL.md"))
    item_id = store.enqueue("skill_proposal", f"skill {name}: {why}", trace_id=trace_ids[0], skill=name, why=why,
                            dedupe=hashlib.sha1(f"{name}\n{text}".encode()).hexdigest()[:12],
                            evidence=trace_ids, proposed=text[:50000], diff=diff[:50000])
    folder = config.settings.state_dir / "proposals"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{item_id}.diff").write_text(diff or text)
    return {"ok": True, "target": f"skill:{name}", "queue_item": item_id,
            "next": f"a person reviews it (`selfimprove queue accept|reject {item_id}`); skills never ship on their own"}
