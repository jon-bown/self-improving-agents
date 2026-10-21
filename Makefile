# Self-improving agent with OpenClaw + MLflow.  `make help` lists targets.
PY      ?= .venv/bin/python
PORT    ?= 5000
L       := levels

.PHONY: help install server connect-openclaw connect-hermes setup level0 level1 level2 level3 level4 level5 week drift test rehearse reset

help:           ## list targets
	@grep -E '^[a-z0-9-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:        ## create .venv and install the package (needs uv)
	uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[dev]"

server:         ## run the MLflow tracking server (sqlite backend; enables online scoring jobs)
	$(PY) -m mlflow server --backend-store-uri sqlite:///mlflow.db --port $(PORT)

connect-openclaw: ## trace your own OpenClaw (its active config) to the MLflow server
	$(PY) connectors/connect.py openclaw

connect-hermes: ## trace your own Hermes Agent ($$HERMES_HOME or ~/.hermes) to the MLflow server
	$(PY) connectors/connect.py hermes

setup:          ## Level 0 setup: register AGENTS.md v1, isolated OpenClaw profile
	$(PY) $(L)/level0_instrument/01_setup.py

level0: setup   ## instrument + first week of traffic
	$(PY) $(L)/level0_instrument/02_first_traces.py --repeat 2

level1:         ## signal: human feedback, code + judge scorers, signal quality
	$(PY) $(L)/level1_signal/01_human_feedback.py
	$(PY) $(L)/level1_signal/02_automated_signal.py
	$(PY) $(L)/level1_signal/03_signal_quality.py

level2:         ## manual loop: dataset -> human fix -> measure -> promote
	$(PY) $(L)/level2_manual_loop/01_build_dataset.py
	$(PY) $(L)/level2_manual_loop/02_write_fix.py
	$(PY) $(L)/level2_manual_loop/03_measure_and_promote.py --promote

week:           ## another week of production traffic, half of it labeled
	$(PY) $(L)/simulate_week.py

drift:          ## a week of traffic that includes the new support_process questions
	$(PY) $(L)/simulate_week.py --drift

level3:         ## agent-assisted: analyst proposes (review proposals/ then run 02_accept.py)
	$(PY) $(L)/level3_agent_assisted/01_propose.py

level4:         ## optimizer: align judge -> GEPA -> gate
	$(PY) $(L)/level4_optimizer/01_align_judge.py
	SIA_BACKEND=direct $(PY) $(L)/level4_optimizer/02_optimize.py
	$(PY) $(L)/level4_optimizer/03_gate.py

level5:         ## autonomous loop with simulated traffic and drift
	$(PY) $(L)/level5_autonomous/02_run_loop.py --simulate 16 --drift --cycles 4

test:           ## the regression gate (dataset + holdout) as pytest
	$(PY) -m pytest -p mlflow.pytest.plugin tests/agent -q

rehearse:       ## run every level offline (no API key, no OpenClaw) into a throwaway store
	./scripts/rehearse_offline.sh

reset:          ## delete local MLflow data, workspaces, proposals, OpenClaw demo state
	rm -rf mlflow.db mlartifacts mlruns .workspaces proposals .openclaw-state
