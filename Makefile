# Self-improving agent with OpenClaw + MLflow.  `make help` lists targets.
PY      ?= .venv/bin/python
PORT    ?= 5000
L       := levels

.PHONY: help install ollama hermes notebooks sdk-notebooks notebooks-ui sdk-notebooks-ui sdk-gateway-stop diagrams server setup level0 level1 level2 level3 level4 level5 week drift test rehearse reset

help:           ## list targets
	@grep -E '^[a-z0-9]+:.*##' Makefile | sed 's/:.*##/\t/'

install:        ## create .venv and install the package (needs uv)
	uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[dev]"

ollama:         ## check the local Ollama server and pull the configured models
	@$(PY) -c "from sia import config, llm; import sys; \
	ms = {llm.split(u)[1] for u in (config.AGENT_MODEL, config.JUDGE_MODEL) if llm.split(u)[0] == 'ollama'}; \
	sys.exit('Ollama not reachable at ' + llm.OLLAMA_BASE + ': install from ollama.com and run ollama serve') if llm.ollama_models() is None else print(' '.join(ms))" \
	| xargs -n1 ollama pull

hermes:         ## install the Hermes Agent runtime and build its 64K-context Ollama model
	uv pip install -p .venv -e ".[dev,hermes]"
	SIA_BACKEND=hermes $(PY) -c "from sia import config, llm; p, m = llm.split(config.AGENT_MODEL); \
	print('Hermes model:', llm.ensure_ollama_ctx_variant(m, llm.HERMES_NUM_CTX) if p == 'ollama' else config.AGENT_MODEL)"

notebooks:      ## open the walkthrough notebooks in JupyterLab
	uv pip install -p .venv -e ".[dev,notebooks]"
	$(PY) -m jupyter lab notebooks/

sdk-notebooks:  ## OpenClaw (openclaw-sdk) + Hermes from Python, MLflow on top: notebooks/sdk
	uv pip install -p .venv -e ".[dev,notebooks,hermes,openclaw-sdk]"
	$(PY) -m jupyter lab notebooks/sdk/

notebooks-ui:   ## MLflow UI for what the notebooks created (notebooks/.mlflow), on port 5001
	$(PY) -m mlflow ui --backend-store-uri sqlite:///notebooks/.mlflow/mlflow.db --port 5001

sdk-notebooks-ui: ## MLflow UI for the notebooks/sdk series, on port 5002
	$(PY) -m mlflow ui --backend-store-uri sqlite:///notebooks/.mlflow/sdk.db --port 5002

sdk-gateway-stop: ## stop the OpenClaw gateway the notebooks/sdk series started (port 19556)
	$(PY) -c "from sia import notebook as nb; from sia.openclaw_client import Gateway; \
	Gateway(nb.NB_STORE / 'sdk' / 'gateway', nb.NB_STORE / 'sdk' / 'openclaw').stop()"

server:         ## run the MLflow tracking server (sqlite backend; enables online scoring jobs)
	$(PY) -m mlflow server --backend-store-uri sqlite:///mlflow.db --port $(PORT)

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

diagrams:       ## rebuild docs/architecture (mmd, svg, png) and README diagram blocks
	$(PY) scripts/render_diagrams.py

rehearse:       ## run every level offline (no API key, no OpenClaw) into a throwaway store
	./scripts/rehearse_offline.sh

reset:          ## delete local MLflow data, workspaces, proposals, OpenClaw demo state
	rm -rf mlflow.db mlartifacts mlruns .workspaces proposals .openclaw-state
