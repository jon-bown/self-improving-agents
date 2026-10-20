# Walkthrough notebooks

> **New: [`sdk/`](sdk/README.md). MLflow on top of OpenClaw and Hermes, from Python.** OpenClaw is driven
> through its gateway with the `openclaw-sdk` client, and Hermes in-process through `AIAgent`. Tracing, the
> Prompt Registry, evaluation, feedback, and a gated improvement loop are all built up in the notebook cells.
> `make sdk-notebooks`.

A guided tour of the code behind the tutorial: the same library functions the `levels/` scripts use, one idea
at a time, with the outputs saved so you can read them before running anything.

| Notebook | What it shows |
|---|---|
| [00 · Tour](00_tour.ipynb) | configuration, providers and model naming, the scenario, the Prompt Registry |
| [01 · Agents and traces](01_agents_and_traces.ipynb) | one question on the offline stand-in, **OpenClaw**, and **Hermes Agent**; the shared trace shape; how each runtime's spans are captured |
| [02 · Signal](02_signal.ipynb) | reviewers (and why they disagree), code checks, tool-level checks, LLM judges, signal as searchable assessments |
| [03 · Store, update, gate](03_store_update_gate.ipynb) | failures → dataset → candidate → gate → promote → rollback, for `AGENTS.md` and for a **skill** (OpenClaw Skill Workshop) |
| [04 · The loop runs itself](04_autonomous_loop.ipynb) | Level 5: drift arrives and the loop proposes, gates, and ships (or refuses) on its own |

## Running them

```bash
make notebooks        # installs the notebook extra and opens JupyterLab (kernel: "self-improving-agent (.venv)")
make notebooks-ui     # MLflow UI for everything the notebooks created, on http://127.0.0.1:5001
```

* **Defaults.** The agent runs on the offline runtime unless a cell asks for OpenClaw or Hermes, against a local
  MLflow store in `notebooks/.mlflow/`.
  * Judges and the analyst use Claude when `ANTHROPIC_API_KEY` is set: Haiku 5.5 as the agent, Opus 5.5 as the
    judge.
  * Notebooks 03 and 04 pass `llm=False` so their outputs are deterministic.
* **Live cells.** Cells that need OpenClaw, Hermes, or an LLM judge check `nb.available_backends()` /
  `config.HAS_LLM` first and skip with a reason. On Claude a live answer takes seconds; on a local Ollama model,
  budget 1–3 minutes.
* **Start over.** Notebook 00 calls `nb.setup(fresh=True)`, which resets the shared store. Notebook 04 uses its
  own database (`loop.db`), because the Prompt Registry is global to a store and the loop must start from v1.
* **Changing settings.** Pass them to `nb.setup()` in the first cell (`backend=`, `provider=`, `store="server"`)
  and restart the kernel; `sia.config` reads the environment once.

Helpers live in `src/sia/notebook.py`: `setup`, `available_backends`, `span_tree` / `show`, `ensure_production`,
`verdicts`.
