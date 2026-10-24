#!/usr/bin/env bash
# Walk through every level with the offline backend: no API key, no OpenClaw, no server.
# Everything goes into a throwaway directory, so it never touches your real MLflow data.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
TMP=$(mktemp -d)
cd "$TMP"   # MLflow writes run artifacts under ./mlruns; keep them out of the repo
export MLFLOW_TRACKING_URI="sqlite:///$TMP/mlflow.db" SIA_BACKEND=offline SIA_NO_LLM=1 \
       SIA_WORKSPACES_DIR="$TMP/ws" SIA_OPENCLAW_STATE_DIR="$TMP/openclaw"
quiet() { "$PY" "$@" 2>&1 | grep -vE "INFO|WARNING|Evaluating|Run (ID|name)|MLflow run|open the|Metrics and|✨|To view|^\s*$" || true; }
step() { echo; echo "━━━ $*"; }

step "Level 0: setup + traffic";          quiet "$ROOT/levels/level0_instrument/01_setup.py"
                                          quiet "$ROOT/levels/level0_instrument/02_first_traces.py" --repeat 2 | tail -1
step "Level 1: signal";                   quiet "$ROOT/levels/level1_signal/01_human_feedback.py" | tail -5
                                          quiet "$ROOT/levels/level1_signal/02_automated_signal.py" | tail -5
step "Level 2: manual loop";              quiet "$ROOT/levels/level2_manual_loop/01_build_dataset.py" | head -2
                                          quiet "$ROOT/levels/level2_manual_loop/02_write_fix.py" | tail -2
                                          quiet "$ROOT/levels/level2_manual_loop/03_measure_and_promote.py" --promote
step "Drift: a new policy file appears";  quiet "$ROOT/levels/simulate_week.py" --drift --seed 1
step "Level 3: analyst proposes";         quiet "$ROOT/levels/level3_agent_assisted/01_propose.py" | tail -3
P=$(ls -d "$ROOT"/proposals/* | tail -1)
                                          quiet "$ROOT/levels/level3_agent_assisted/02_accept.py" "$P" --promote
step "Level 4: optimizer (offline stand-in)"; quiet "$ROOT/levels/level4_optimizer/02_optimize.py"
                                          quiet "$ROOT/levels/level4_optimizer/03_gate.py" | tail -8
echo "(offline, a FAIL here is expected: v3 already scores 100%, so there is nothing to improve)"
step "Regression suite (pytest)";         "$PY" -m pytest -p mlflow.pytest.plugin "$ROOT/tests/agent" -q -p no:warnings 2>&1 | tail -2
step "Level 5: autonomous loop from scratch"
export MLFLOW_TRACKING_URI="sqlite:///$TMP/loop.db"
quiet "$ROOT/levels/level0_instrument/01_setup.py" > /dev/null
quiet "$ROOT/levels/level5_autonomous/02_run_loop.py" --simulate 16 --drift --cycles 4 --improver analyst
rm -rf "$P"; rmdir "$ROOT/proposals" 2>/dev/null || true
echo; echo "Rehearsal data in $TMP"
