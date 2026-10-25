# The self-improving-agent skill

[`self-improving-agent/`](self-improving-agent/SKILL.md) is an [agent skill](https://agentskills.io)
that brings everything in this repo to a coding agent: the framework from the blog series, the
`selfimprove` harness and its workflows, the OpenClaw connector, and what the talk's four loops
taught. Give it to Claude Code (or any agent that reads skills) and ask it to make your agent
improve itself, build a verifier from your traces, trace OpenClaw into MLflow, schedule the loop,
or work out why the gate rejected a candidate.

It decides *what* to do and in which order. It finds the first gap (tracing, update target, human
signal, verifier), uses the harness when it's installed or plain MLflow when it isn't, moves up
one level of automation at a time, and holds to the rules: the proposer never approves, the
holdout stays hidden, a judge's verdict never becomes an expectation, and promotion is an alias
move with a record. It also covers the case where the agent is asked to improve its own
instructions.

## Install

For Claude Code, everywhere:

```bash
mkdir -p ~/.claude/skills && cp -r skills/self-improving-agent ~/.claude/skills/
```

or for one project, into that project's `.claude/skills/`. Any agent that reads skill folders
(OpenClaw, Hermes, Codex, Gemini CLI) can use the same folder. It pairs with MLflow's own skills
for the mechanics of tracing, scorers and trace analysis (`mlflow agent setup`, or this repo's
`.agents/skills/`).

## What's in it

| File | Covers |
|---|---|
| `SKILL.md` | where the user is, which path fits, the harness commands, levels of automation, what to do when the gate rejects, the rules |
| `references/framework.md` | the framework: self-improvement vs self-evolution, the loop, the baseline numbers |
| `references/verifiers.md` | the criteria file: code checks, tool-span checks, grounded judges, must-pass checks, judge alignment |
| `references/workflows.md` | `cycle`, GitHub Actions, auto-promotion and rollback, ungated outside edits, interrupted runs |
| `references/adapters.md` | bringing an agent (`python`, `http`, `command`, `litellm`, `openclaw`), Hermes, isolation |
| `references/openclaw.md` | OpenClaw with the harness, and the MLflow connector |
| `references/guardrails.md` | the failures the rules prevent, and red flags in a candidate diff |
| `references/troubleshooting.md` | install, MLflow 3.17 and long-run problems, with fixes |
| `references/mlflow-recipes.md` | every step in raw MLflow, without the harness |

## Tests

`self-improving-agent/evals/` holds test cases: realistic requests, each with the checks a good
answer passes (for example, the gate rejecting a candidate for length, an OpenClaw plugin that
sends no traces, and turning on auto-promote). With Claude Code, run them with and without the
skill from that folder:

```bash
claude plugin eval . --allow-tools Write,Edit,Bash
```

These are real, billed model runs.
