"""Level 3 · An agent reads the traces and proposes; a person approves.

With six labels a person can read everything. With ten reviewers and a week of traffic the
bottleneck is *reading*. Two readers work together here:

  1. MLflow issue discovery (`discover_issues`) clusters the failing traces into named issues
     with root causes and severities, and links each issue back to its traces in MLflow.
  2. An LLM analyst turns those issues into judge criteria and ONE proposed change. It picks
     the update target: AGENTS.md (behavior for every answer) or a skill (a procedure the agent
     loads only for matching questions -- OpenClaw's own unit of learned behavior).

Everything lands in proposals/<timestamp>/ for review:

    PROPOSAL.md    issues, themes, criteria, and the diff (AGENTS.md or the new/updated SKILL.md)
    AGENTS.md      proposed instructions (unchanged when the target is a skill)
    skill.json     the proposed skill, when the target is a skill
    themes.json    criteria that become registered LLM-judge scorers if accepted
    issues.json    the MLflow issues, with root causes and linked traces

Nothing is registered or promoted. Review the folder, edit anything, then run 02_accept.py.

    python levels/level3_agent_assisted/01_propose.py [--discovery mlflow|analyst] [--target auto|agents_md|skill]

Prefer a coding agent? `mlflow agent setup --agent claude` installs MLflow's trace-analysis
skills into Claude Code; ask it to do the same job against this experiment.
"""

import argparse
import datetime as dt
import difflib
import json

from sia import config, improvers, tracking
from sia.agent import load_agents_md, load_skill, production_skills
from sia.scenario import REPO_ROOT
from sia.skills import skill_md

parser = argparse.ArgumentParser()
parser.add_argument("--since-hours", type=float, default=24 * 7)
parser.add_argument("--discovery", choices=["mlflow", "analyst"], default="mlflow",
                    help="mlflow: discover_issues finds the themes; analyst: the LLM clusters on its own")
parser.add_argument("--target", choices=["auto", "agents_md", "skill"], default="auto")
args = parser.parse_args()

tracking.setup()
prod = load_agents_md("production")
skills = {n: load_skill(n, v).template for n, v in production_skills().items()}
traces = tracking.agent_traces(f"trace.timestamp_ms > {tracking.since_ms(args.since_hours)} "
                               f"AND tags.prompt_version = '{prod.version}'")
report = improvers.failure_report(traces)
print(f"{len(traces)} traces on production v{prod.version} (+ {len(skills)} skills); "
      f"{len(report)} have failing verdicts")
if not report:
    raise SystemExit("Nothing to fix. Generate traffic and signal first (Levels 0-1).")

target = None if args.target == "auto" else args.target
issues, discovery_run = [], None
if config.HAS_LLM and args.discovery == "mlflow":
    failing = [t for t in traces if t.info.trace_id in {r["trace_id"] for r in report}]
    found = improvers.discover(failing)
    issues, discovery_run = found["issues"], found["run_id"]
    print(f"MLflow issue discovery: {len(issues)} issues (run {discovery_run})")
    for i in issues:
        print(f"  [{i['severity']}] {i['name']} ({i['trace_count']} traces)")
if config.HAS_LLM:
    proposal = improvers.analyze(report, prod.template, skills=skills, issues=issues or None, target=target)
else:
    print("No LLM available: using the rule-based offline analyst (no issue discovery).")
    proposal = improvers.offline_analyze(report, prod.template, skills=skills, target=target)

out = REPO_ROOT / "proposals" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True)
(out / "AGENTS.md").write_text(proposal["agents_md"])
(out / "themes.json").write_text(json.dumps(proposal["themes"], indent=2))
(out / "issues.json").write_text(json.dumps({"discovery_run": discovery_run, "issues": issues}, indent=2))
(out / "failure_report.json").write_text(json.dumps(report, indent=1))

if proposal["update_target"] == "skill":
    sk = proposal["skill"]
    (out / "skill.json").write_text(json.dumps(sk, indent=2))
    before = skills.get(sk["name"], "")
    after = skill_md(sk["name"], sk["description"], sk["body"])
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        f"skills/{sk['name']}/SKILL.md (production)" if before else "/dev/null",
                                        f"skills/{sk['name']}/SKILL.md (proposed)"))
    headline = f"# Proposal: {'update' if before else 'new'} skill `{sk['name']}`\n"
else:
    diff = "".join(difflib.unified_diff(prod.template.splitlines(True), proposal["agents_md"].splitlines(True),
                                        f"AGENTS.md v{prod.version}", "AGENTS.md proposed"))
    headline = f"# Proposal against AGENTS.md v{prod.version}\n"

md = [headline, f"{proposal['change_summary']}\n",
      f"Evidence: {len(report)} failing traces from the last {args.since_hours:g}h.\n"]
if issues:
    md.append(f"## Issues discovered by MLflow (run `{discovery_run}`)\n")
    for i in issues:
        md.append(f"- **{i['name']}** [{i['severity']}] -- {i['description']}  \n"
                  f"  root cause: {'; '.join(i['root_causes']) or 'n/a'}  \n  traces: {', '.join(i['trace_ids'][:3])}")
md.append("\n## Themes\n")
for t in proposal["themes"]:
    flag = " _(reviewer noise: no scorer will be created)_" if t["is_reviewer_noise"] else ""
    md.append(f"- **{t['slug']}** ({t['count']} traces){flag}: {t['summary']}  \n"
              f"  criterion: _{t['criterion']}_  \n  e.g. {', '.join(t['trace_ids'][:3])}")
md += ["\n## Diff\n", f"```diff\n{diff}```\n",
       "## Reviewer checklist\n",
       "- [ ] The themes match what the rationales actually say",
       "- [ ] Each criterion asks for the behavior reviewers want, not a proxy",
       "- [ ] The update target fits: global behavior -> AGENTS.md, one kind of question -> a skill",
       "- [ ] The change is the smallest one that addresses the themes"]
(out / "PROPOSAL.md").write_text("\n".join(md))

print(f"\nTarget: {proposal['update_target']}" + (f" ({proposal['skill']['name']})" if proposal["update_target"] == "skill" else ""))
print(f"{proposal['change_summary']}")
print(f"\nWrote {out.relative_to(REPO_ROOT)}/  -- review it, then:")
print(f"  python levels/level3_agent_assisted/02_accept.py {out.relative_to(REPO_ROOT)} [--promote]")
