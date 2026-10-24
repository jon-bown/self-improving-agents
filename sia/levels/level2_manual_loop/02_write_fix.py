"""Level 2 · Update (by hand): register a human-written AGENTS.md as a new version.

A person read the rationales and wrote scenario/fixes/level2_manual_AGENTS.md. It:
  * points the agent at knowledge/*.md as the source of truth (fixes stale facts),
  * asks for the answer in the first sentence (fixes the buried answer),
  * says to escalate instead of guessing (fixes the confident guesses).

It is registered as a new version but NOT promoted: production still serves the old one
until the measurement in 03 says otherwise. The commit message names the evidence.

    python levels/level2_manual_loop/02_write_fix.py [--file path/to/AGENTS.md]
"""

import argparse
import difflib

from sia import data, tracking
from sia.agent import load_agents_md, register_candidate
from sia.scenario import SCENARIO_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=str(SCENARIO_DIR / "fixes" / "level2_manual_AGENTS.md"))
args = parser.parse_args()

tracking.setup()
new_text = open(args.file).read()
prod = load_agents_md("production")
evidence = ", ".join(t.info.trace_id for t in data.labeled_failures()[:5])

version = register_candidate(
    new_text,
    commit_message=f"Lead with the answer; use knowledge/ as source of truth; escalate instead of guessing. "
                   f"Evidence: {evidence}",
    author="human", level=2,
)
print("".join(difflib.unified_diff(prod.template.splitlines(True), new_text.splitlines(True),
                                   f"AGENTS.md v{prod.version} (production)", f"AGENTS.md v{version} (candidate)")))
print(f"\nRegistered candidate v{version}. Production still -> v{prod.version}.")
print(f"Next: 03_measure_and_promote.py --candidate {version}")
