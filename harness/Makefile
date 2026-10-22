# selfimprove harness.  `make help` lists targets.
PY  ?= .venv/bin/python
SI  ?= .venv/bin/selfimprove

.PHONY: help install doctor init traffic review loop cycle report test regression notebooks ui reset

help:           ## list targets
	@grep -E '^[a-z]+:.*##' Makefile | sed 's/:.*##/\t/'

install:        ## create .venv, install the harness and the notebook kernel
	uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[dev,notebooks]"
	$(PY) -m ipykernel install --prefix .venv --name selfimprove --display-name "selfimprove (.venv)"
	@test -f .env || cp .env.example .env

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
	.venv/bin/jupyter lab notebooks

ui:             ## MLflow UI on the harness store (port 5003; the notebooks: STORE=.selfimprove/notebooks)
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(or $(STORE),.selfimprove)/mlflow.db --port 5003

reset:          ## delete the harness's local store, workspaces and queue
	rm -rf .selfimprove
