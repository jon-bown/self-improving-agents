# selfimprove harness.  `make help` lists targets.
# One venv for the whole repo, at the root: `make install` here runs the root's `make setup`.
VENV ?= ../.venv
PY   ?= $(VENV)/bin/python
SI   ?= $(VENV)/bin/selfimprove

.PHONY: help install doctor init traffic review loop cycle report test regression notebooks ui reset session-server session-notebooks session-precompute

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:        ## set up the repo (root `make setup`): venv, kernel, OpenClaw, connector, .env
	$(MAKE) -C .. setup

doctor:         ## show the configuration; check that your agent loads and answers
	$(SI) doctor --smoke "How long can I wait to request a refund?"

init:           ## instructions v1 -> production, golden cases -> dataset
	$(SI) init

traffic:        ## a week of example production traffic
	$(SI) replay examples/questions.txt --repeat 2

review:         ## simulated reviewers label the new traffic (or: selfimprove label)
	$(PY) -m examples.reviewer

loop:           ## the nightly jobs, by hand
	$(SI) score && $(SI) monitor && $(SI) align && $(SI) triage && $(SI) approve

cycle:          ## the whole loop once
	$(SI) cycle

report:         ## baseline numbers and the human queue
	$(SI) report

test:           ## the offline loop test (no keys)
	$(PY) -m pytest tests -q

regression:     ## every dataset row as an @mlflow.test case (calls your agent and judges)
	SI_REGRESSION=1 $(PY) -m pytest -p mlflow.pytest.plugin tests/regression -n 4 -q

notebooks:      ## JupyterLab on the walkthrough notebooks
	$(VENV)/bin/jupyter lab notebooks

ui:             ## MLflow UI on the harness store (port 5003; the notebooks: STORE=.selfimprove/notebooks)
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(or $(STORE),.selfimprove)/mlflow.db --port 5003

reset:          ## delete the harness's local store, workspaces and queue
	rm -rf .selfimprove

session-server: ## the ODSC session's precomputed store on http://127.0.0.1:19560 (leave running)
	mkdir -p session/store
	$(VENV)/bin/mlflow server --host 127.0.0.1 --port 19560 \
	  --backend-store-uri sqlite:///session/store/mlflow.db --artifacts-destination session/store/artifacts

session-notebooks: ## JupyterLab on the session notebooks (needs session-server)
	$(VENV)/bin/jupyter lab session/notebooks

session-precompute: ## re-run the session's four loops into session/store (needs session-server; ~2 h)
	$(PY) -m session.precompute
