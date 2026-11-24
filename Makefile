# Self-improving agents with MLflow & OpenClaw.  `make help` lists targets.
#
# Start here:  make setup        then, for the OpenClaw + MLflow demo:
#              make demo-server  (terminal 1, leave running)
#              make demo         (terminal 2)
VENV      := .venv
DEMO_SH   := connectors/openclaw-mlflow/examples/demo.sh
DEMO_PORT ?= 5050
Q         ?= Do you have an uptime SLA?

.PHONY: help setup check notebooks test demo-server demo demo-ask demo-reset clean

help:          ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

setup:         ## install everything: .venv, Jupyter kernel, OpenClaw, the MLflow connector, .env files
	@scripts/setup.sh

check:         ## check the setup without changing anything
	@scripts/setup.sh check

notebooks:     ## JupyterLab on the whole repo (blog/, harness/notebooks, harness/session/notebooks)
	$(VENV)/bin/jupyter lab

test:          ## the offline test suites (no API key, no OpenClaw): harness, SIA, connector
	cd harness && ../$(VENV)/bin/python -m pytest tests -q
	cd sia && ../$(VENV)/bin/python -m pytest -q
	cd connectors/openclaw-mlflow && npm test --silent

demo-server:   ## the connector demo's own MLflow server on http://127.0.0.1:5050 (leave it running)
	@PORT=$(DEMO_PORT) $(DEMO_SH) server

demo:          ## OpenClaw profile + connector -> MLflow server, then ask one question and show its trace
	@test -f "$${DEMO:-$$HOME/odsc-5min}/openclaw/openclaw.json" || PORT=$(DEMO_PORT) $(DEMO_SH) setup
	@PORT=$(DEMO_PORT) $(DEMO_SH) ask "$(Q)"
	@PORT=$(DEMO_PORT) $(DEMO_SH) traces
	@echo; echo "MLflow: http://127.0.0.1:$(DEMO_PORT)  (experiment openclaw-demo -> Traces; prompt openclaw-agents-md)"
	@echo "Next: make demo-ask Q=\"...\", or the full walkthrough in connectors/openclaw-mlflow/examples/live-demo.md"

demo-ask:      ## ask the demo agent a question: make demo-ask Q="How do I reset my password?"
	@PORT=$(DEMO_PORT) $(DEMO_SH) ask "$(Q)"

demo-reset:    ## delete the demo's profile and store (stop demo-server first)
	@PORT=$(DEMO_PORT) $(DEMO_SH) reset

clean:         ## remove .venv (OpenClaw and the kernel live there too) and the connector build
	rm -rf $(VENV) connectors/openclaw-mlflow/node_modules connectors/openclaw-mlflow/dist connectors/openclaw-mlflow/*.tgz
