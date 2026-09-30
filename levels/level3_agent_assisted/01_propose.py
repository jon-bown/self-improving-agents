"""Level 3 · An agent reads the traces and proposes; a person approves.

With six labels a person can read everything. With ten reviewers and a week of traffic the
bottleneck is *reading*. Here an LLM trace analyst reads every failing trace (human, judge, and
code verdicts) and writes a proposal to proposals/<timestamp>/:

    PROPOSAL.md   themes, representative trace ids, criteria, and the AGENTS.md diff
    AGENTS.md     the proposed instructions
    themes.json   criteria that become registered LLM-judge scorers if accepted

Nothing is registered or promoted. Review the folder, edit anything, then run 02_accept.py.

    python levels/level3_agent_assisted/01_propose.py [--since-hours 168]

Prefer a coding agent? `mlflow agent setup --agent claude` installs MLflow's trace-analysis
skills into Claude Code; ask it to do the same job against this experiment.
"""

import argparse
import datetime as dt
import difflib
import json

from sia import config, improvers, tracking
from sia.agent import load_agents_md
from sia.scenario import REPO_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--since-hours", type=float, default=24 * 7)
args = parser.parse_args()

tracking.setup()
prod = load_agents_md("production")
traces = tracking.agent_traces(f"trace.timestamp_ms > {tracking.since_ms(args.since_hours)} "
                               f"AND tags.prompt_version = '{prod.version}'")
report = improvers.failure_report(traces)
print(f"{len(traces)} traces on production v{prod.version}; {len(report)} have failing verdicts")
if not report:
    raise SystemExit("Nothing to fix. Generate traffic and signal first (Levels 0-1).")

proposal = (improvers.analyze if config.HAS_LLM else improvers.offline_analyze)(report, prod.template)

out = REPO_ROOT / "proposals" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True)
(out / "AGENTS.md").write_text(proposal["agents_md"])
(out / "themes.json").write_text(json.dumps(proposal["themes"], indent=2))
(out / "failure_report.json").write_text(json.dumps(report, indent=1))
diff = "".join(difflib.unified_diff(prod.template.splitlines(True), proposal["agents_md"].splitlines(True),
                                    f"AGENTS.md v{prod.version}", "AGENTS.md proposed"))
md = [f"# Proposal against AGENTS.md v{prod.version}\n", f"{proposal['change_summary']}\n",
      f"Evidence: {len(report)} failing traces from the last {args.since_hours:g}h.\n", "## Themes\n"]
for t in proposal["themes"]:
    flag = " _(reviewer noise: no scorer will be created)_" if t["is_reviewer_noise"] else ""
    md.append(f"- **{t['slug']}** ({t['count']} traces){flag}: {t['summary']}  \n"
              f"  criterion: _{t['criterion']}_  \n  e.g. {', '.join(t['trace_ids'][:3])}")
md += ["\n## Diff\n", f"```diff\n{diff}```\n",
       "## Reviewer checklist\n",
       "- [ ] The themes match what the rationales actually say",
       "- [ ] Each criterion asks for the behavior reviewers want, not a proxy",
       "- [ ] The AGENTS.md change is the smallest one that addresses the themes"]
(out / "PROPOSAL.md").write_text("\n".join(md))

print("\n".join(md[:4 + len(proposal["themes"])]))
print(f"\nWrote {out.relative_to(REPO_ROOT)}/  -- review it, then:")
print(f"  python levels/level3_agent_assisted/02_accept.py {out.relative_to(REPO_ROOT)}")
