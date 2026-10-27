# Session: four loops of measurable improvement on OpenClaw

Worked examples for the ODSC West 2026 session. An OpenClaw support agent for a fictional
product, Acme Cloud, ships with two flaws in its `AGENTS.md`. Two more arrive later, when other
teams edit `AGENTS.md` and ship it straight to production without the gate (`changes/`). Four
loops of the self-improvement framework find the four failures and fix them one at a time. Every
loop ends with a measured before and after, and each one hands more of the work to the harness:

| Loop | Failure | Signal | Store | Update | Promotes |
|---|---|---|---|---|---|
| [1 · Manual](notebooks/01_manual.ipynb) | **Skips the lookup**: answers from stale quick facts | you label; `read_knowledge` checks the connector's TOOL spans | you | you write v2 | you |
| [2 · Assisted](notebooks/02_assisted.ipynb) | **Improvises** when the docs are silent, instead of escalating | + `grounded_in_docs`: a judge on what the agent actually read | triage + approve | the LLM analyst | you |
| [3 · Optimizer](notebooks/03_optimizer.ipynb) | **Unsafe account help** after the support team's edit: asks for passwords, "resets" 2FA | + a must-pass check; the judge aligned to your labels | triage + approve | an optimizer (MetaPrompt, then GEPA) | you |
| [4 · Autonomous](notebooks/04_autonomous.ipynb) | **Bad escalation** after the growth team's edit: discounts, roadmap dates | + `no_commercial_promises`; a 50% label sample | `cycle` | `cycle` | the gate, with `monitor` |

The agent is real OpenClaw (`openclaw agent --local`, Claude Haiku 5.5) in an isolated profile,
and the [MLflow connector](../../connectors/openclaw-mlflow/README.md) traces every run. The
connector is what makes loops 1 and 2 possible: whether the agent read a policy file, and what
that file said, are only in OpenClaw's own trace (its `read` TOOL spans), which the harness finds
by session key.

## Present it

`store/` holds a complete precomputed run, so nothing calls OpenClaw or a model during the talk.
From `harness/`:

```bash
make session-server
```

Leave it running, then open the notebooks with `make session-notebooks`, or the
MLflow UI at http://127.0.0.1:19560: `Traces` (filter `tags.session_loop = '1'`), `Prompts`
(`acme-agents-md`, v1 to v12, including the rejected candidates), `Datasets`, and the `gate` and `cycle` runs.

The notebooks only read the store, so their cells run in seconds and in any order within a
notebook.

## Run it yourself

[RUNBOOK.md](RUNBOOK.md) has every step as a `selfimprove` command. `python -m session.precompute`
runs all four loops end to end into `store/` (about two hours: each OpenClaw call is about 20 s,
and the gate runs every dataset row twice). It needs `ANTHROPIC_API_KEY` in `harness/.env`,
`openclaw` on `PATH`, Node 22+, and the session server.

```bash
.venv/bin/python -m session.precompute --fresh    # with the server stopped: empty the store
make session-server                               # in another terminal
.venv/bin/python -m session.precompute            # or --loops 3 4
```

Models are LLM-driven, so a fresh run won't reproduce the exact numbers, and a gate can fail
on a run (that is the gate doing its job). Re-run that loop with `--loops N`.

## Files

```
agent/AGENTS.md        v1: the instructions with the first two flaws
changes/               the edits other teams ship without the gate (loops 3 and 4)
knowledge/             the policies the agent should answer from (fixed; not improved)
fixes/v2_manual.md     loop 1's hand-written fix
traffic/loopN.txt      each loop's production questions
cases.yaml             the answer key the simulated reviewers grade with (the agent never sees it)
golden.yaml            eight holdout cases, two per failure; only the gate sees them
criteria/loopN.yaml    the verifier, growing one loop at a time
reviewer.py            simulated reviewers (in the room, that's you)
precompute.py          runs the four loops into store/
session.env            harness settings: OpenClaw, the connector, the session store
notebooks/             one notebook per loop, reading store/ (with "Under the hood" MLflow cells)
build_notebooks.py     writes notebooks/: edit this, then `.venv/bin/python session/build_notebooks.py`
store/                 the precomputed MLflow store and results/loopN.json
```

`.state/` (not committed) holds the isolated OpenClaw profile, the harness's checkpoints and
queue, and the per-version workspaces.
