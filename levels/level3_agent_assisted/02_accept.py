"""Level 3 · Accept a reviewed proposal: register the candidate, grow the verifiers, run the gate.

Accepting a proposal does three things, and only the last one can ship anything:
  1. registers the candidate, without promoting it:
       AGENTS.md target -> a new AGENTS.md version (author=analyst)
       skill target     -> an OpenClaw Skill Workshop proposal (pending, security-scanned)
                           + the same SKILL.md as a new MLflow skill version
  2. registers each non-noise theme criterion as a Guidelines judge in MLflow's scorer registry
     (theme_<slug>), so every future gate checks it -- the test suite grows with the evidence,
  3. adds the representative failing traces to the regression dataset and runs the gate:
     production vs production-with-this-change.

With --promote (a person pressing the button) a passing candidate ships. For a skill that means
Workshop `apply` first (OpenClaw re-scans and records rollback metadata) and only then moving the
skill's @production alias in MLflow. A failing skill is `reject`ed in the Workshop with the
gate's reasons. Without --promote a passing skill stays pending; re-run with --promote to ship.

    python levels/level3_agent_assisted/02_accept.py proposals/<ts> [--promote]
"""

import argparse
import json
from pathlib import Path

import mlflow

from sia import config, data, gate, signals, skills, tracking
from sia.agent import AgentSpec, register_candidate

parser = argparse.ArgumentParser()
parser.add_argument("proposal_dir")
parser.add_argument("--promote", action="store_true")
args = parser.parse_args()

experiment_id = tracking.setup()
d = Path(args.proposal_dir)
themes = [t for t in json.loads((d / "themes.json").read_text()) if not t["is_reviewer_noise"]]
state_file = d / "accepted.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {}
evidence = ", ".join(i for t in themes for i in t["trace_ids"][:2])

# ---- 1. register the candidate --------------------------------------------------------------
if state:
    print(f"Already accepted: {state['candidate']}; gate {'passed' if state['gate_passed'] else 'failed'}")
elif (d / "skill.json").exists():
    sk = json.loads((d / "skill.json").read_text())
    p = skills.propose(sk["name"], sk["description"], sk["body"], goal=f"Level 3 proposal {d.name}",
                       evidence=evidence, author="analyst", level=3)
    state = {"kind": "skill", "skill": sk["name"], "version": p.version, "proposal_id": p.proposal_id,
             "scan": p.scan, "candidate": f"skill {sk['name']} v{p.version}"}
    where = f"Workshop proposal {p.proposal_id} (scan: {p.scan})" if p.proposal_id else "MLflow only (no OpenClaw)"
    print(f"Registered skill {sk['name']} v{p.version} -> {where}")
else:
    version = register_candidate(
        (d / "AGENTS.md").read_text(),
        commit_message=f"Analyst proposal {d.name}: " + "; ".join(t["slug"] for t in themes),
        author="analyst", level=3, proposal=d.name,
    )
    state = {"kind": "agents_md", "version": version, "candidate": f"AGENTS.md v{version}"}
    print(f"Registered candidate AGENTS.md v{version}")

spec = (AgentSpec("production", ((state["skill"], state["version"]),)) if state["kind"] == "skill"
        else AgentSpec(state["version"]))

# ---- 2 + 3. grow the verifiers and the dataset, then gate --------------------------------
if "gate_passed" not in state:
    if config.HAS_LLM:
        for t in themes:
            signals.register_theme(t["slug"], t["criterion"])
            print(f"  + scorer theme_{t['slug']}: {t['criterion']}")
    dataset = data.get_or_create_dataset(experiment_id)
    records = [data.trace_to_record(trace, f"theme:{t['slug']}")
               for t in themes for i in t["trace_ids"] if (trace := mlflow.get_trace(i)) is not None]
    records = [r for r in data.dedupe(records) if r["expectations"].get("expected_response")]
    if records:
        dataset.merge_records(records)
    print(f"Dataset: {data.growth_report(dataset)}")

    decision = gate.compare(spec, "production", train=data.rows(dataset), holdout=data.holdout_rows())
    print("\n" + decision.table())
    print(f"\nGate: {'PASS' if decision.passed else 'FAIL'}", *(f"\n  - {r}" for r in decision.reasons))
    state |= {"gate_passed": decision.passed, "gate_run": decision.run_id, "reasons": decision.reasons}
    state_file.write_text(json.dumps(state, indent=2))
    if not decision.passed and state["kind"] == "skill":
        skills.reject(skills.SkillProposal(state["skill"], state["version"], state["proposal_id"]),
                      "MLflow gate failed: " + "; ".join(decision.reasons))
        print(f"Workshop proposal {state['proposal_id']} rejected with the gate's reasons.")

# ---- ship (a person decides) ---------------------------------------------------------------
if state["gate_passed"] and args.promote and not state.get("promoted"):
    reason = f"Level 3 proposal {d.name}, approved by a human; gate {state['gate_run']}"
    if state["kind"] == "skill":
        ok, msg = skills.apply(skills.SkillProposal(state["skill"], state["version"], state["proposal_id"]))
        print(f"Skill Workshop apply: {msg}")
        if not ok:
            raise SystemExit("Not promoted: the Workshop refused to apply (see scan findings).")
    receipt = gate.promote(spec, reason=reason)
    state |= {"promoted": True, "receipt": receipt}
    state_file.write_text(json.dumps(state, indent=2))
    print(f"Shipped {state['candidate']} to production.")
elif state["gate_passed"] and not state.get("promoted"):
    print(f"\nPassed. Re-run with --promote to ship {state['candidate']} (you are the approver at this level).")
