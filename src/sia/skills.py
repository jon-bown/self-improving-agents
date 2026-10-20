"""Skills as an update target: MLflow decides *whether*, OpenClaw's Skill Workshop governs *how*.

A prompt edit changes every answer. A skill (skills/<name>/SKILL.md) is a procedure the agent
loads only for matching questions -- the next step up the update-target scale, and OpenClaw's own
unit of learned behavior. Two systems each do what they are good at:

    OpenClaw Skill Workshop                    MLflow
    -----------------------                    ------
    propose-create / propose-update            register the same SKILL.md as a skill version
      -> pending proposal, security scan         (prompt `acme-skill-<name>`, not promoted)
                                               gate: production vs production + this skill
    apply   (re-scan, rollback metadata)  <--  pass: then move the skill's @production alias
    reject  (reason = the gate's reasons) <--  fail

The MLflow version is what gets evaluated and what gets served, so what shipped is exactly what
was measured. The Workshop proposal id is stored on the MLflow version (and vice versa in the
proposal's evidence), so either side can be traced to the other.

If OpenClaw isn't installed (CI, the offline rehearsal), proposals are registered in MLflow only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sia import config
from sia.agent import _OPENCLAW_LOCK, load_agents_md, openclaw_env, production_skills, load_skill, register_skill
from sia.workspace import build_workspace


@dataclass
class SkillProposal:
    name: str
    version: int                      # MLflow skill version (registered, not promoted)
    proposal_id: str | None           # OpenClaw Skill Workshop proposal, if OpenClaw is available
    scan: str = "n/a"                 # Workshop security scan state: clean | warn | critical | n/a
    is_update: bool = False


def skill_md(name: str, description: str, body: str) -> str:
    return f'---\nname: "{name}"\ndescription: "{description.replace(chr(34), chr(39))}"\n---\n\n{body.strip()}\n'


def _proposal_md(name: str, description: str, body: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return (f'---\nname: "{name}"\ndescription: "{description.replace(chr(34), chr(39))}"\n'
            f'status: proposal\nversion: "v1"\ndate: "{now}"\n---\n\n{body.strip()}\n')


def workshop_available() -> bool:
    return shutil.which(config.OPENCLAW_BIN) is not None and (config.OPENCLAW_STATE_DIR / "openclaw.json").exists()


def staging_workspace() -> Path:
    """The production agent, rendered where the Workshop can see it. Re-rendered before every
    Workshop call so update proposals bind to the skill content that is actually serving."""
    skills = {n: load_skill(n, v).template for n, v in production_skills().items()}
    return build_workspace(load_agents_md("production").template, "skill-staging", skills=skills)


def _workshop(*args: str) -> dict:
    ws = staging_workspace()
    cmd = [config.OPENCLAW_BIN, "skills", "workshop", *args, "--agent", "main", "--json"]
    with _OPENCLAW_LOCK:
        proc = subprocess.run(cmd, env=openclaw_env(ws), capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"openclaw skills workshop {args[0]} failed: {(proc.stderr or proc.stdout)[-1500:]}")
    out = json.loads(proc.stdout, strict=False)
    return out.get("record") or out.get("proposal") or out


def propose(name: str, description: str, body: str, *, goal: str, evidence: str, author: str,
            level: int) -> SkillProposal:
    """File the skill with the Skill Workshop (pending + scanned) and register it in MLflow."""
    is_update = name in production_skills()
    proposal_id, scan = None, "n/a"
    if workshop_available():
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "PROPOSAL.md"
            f.write_text(_proposal_md(name, description, body))
            args = (["propose-update", name, "--proposal", str(f)] if is_update else
                    ["propose-create", "--name", name, "--description", description[:160], "--proposal", str(f)])
            rec = _workshop(*args, "--goal", goal[:500], "--evidence", evidence[:1500])
        proposal_id, scan = rec.get("id"), (rec.get("scan") or {}).get("state", "unknown")
    version = register_skill(
        name, skill_md(name, description, body),
        commit_message=f"{'Update' if is_update else 'New'} skill {name}: {goal}"[:900],
        author=author, level=level, workshop_proposal=proposal_id or "none", workshop_scan=scan,
    )
    return SkillProposal(name, version, proposal_id, scan, is_update)


def apply(p: SkillProposal) -> tuple[bool, str]:
    """Workshop apply (re-runs the security scanner, records rollback metadata). Call only after
    the MLflow gate passed; promotion of the MLflow version follows only if this succeeds."""
    if not p.proposal_id:
        return True, "no Workshop proposal (OpenClaw unavailable); MLflow gate only"
    try:
        rec = _workshop("apply", p.proposal_id)
    except RuntimeError as e:
        return False, str(e)
    return rec.get("status") == "applied", f"Workshop status: {rec.get('status')}"


def reject(p: SkillProposal, reason: str) -> None:
    if p.proposal_id:
        try:
            _workshop("reject", p.proposal_id, "--reason", reason[:500])
        except RuntimeError:
            pass  # already closed (e.g. went stale); the MLflow version simply never gets promoted


def pending() -> list[dict]:
    """Pending Workshop proposals -- including ones OpenClaw's own self-learning created."""
    if not workshop_available():
        return []
    ws = staging_workspace()
    with _OPENCLAW_LOCK:
        proc = subprocess.run([config.OPENCLAW_BIN, "skills", "workshop", "list", "--agent", "main", "--json"],
                              env=openclaw_env(ws), capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return []
    data = json.loads(proc.stdout, strict=False)
    return [p for p in data.get("proposals", []) if p.get("status") == "pending"]
