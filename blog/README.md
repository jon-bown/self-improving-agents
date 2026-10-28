# Blog code

The code from the *Self-Improvement & MLflow* series on ODSC's blog, one folder per post. Each
folder has the snippets as files and a notebook, `partN.ipynb`, that has them in article order.

| | post | files |
|---|---|---|
| [`part1/`](part1/part1.ipynb) | [Self-Improvement and Self-Evolving with MLflow](https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/) | trace the agent, log feedback, write scorers, build a regression dataset, write a regression test |
| [`part2/`](part2/part2.ipynb) | [Self-Improvement & MLflow: Part 2](https://opendatascience.com/self-improvement-mlflow-part-2/) | version the prompt, judge traces in place, compare v1 and v2, run the dataset as tests, promote by alias, CI job |
| [`part3/`](part3/part3.ipynb) | [Self-Improvement & MLflow: Automate via CI/CD](https://opendatascience.com/self-improvement-mlflow-part-3/) | baseline, scheduled ingestion with checkpoints, a trace-aware judge, judge alignment, automated approval, improving a skill, the CI workflow |

In `part3/`, the `loop/` and `tests/regression/` paths match the ones `self-improvement.yml` runs.
That workflow sits in `part3/`, not `.github/workflows/`, so it doesn't run in this repo.

The snippets show the MLflow calls, not a complete agent. Names such as `agent`, `run_agent`
and the helpers marked *our helper* are yours to supply, and each file's docstring lists the
ones it expects. For the full loop, runnable end to end, see [`harness/`](../harness/README.md).
