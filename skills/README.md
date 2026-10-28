# Skills

Two [agent skills](https://agentskills.io), for the two sides of the loop:

| skill | who loads it | what it's for |
|---|---|---|
| [`self-improving-agent/`](self-improving-agent/SKILL.md) | a coding agent (Claude Code, Codex, ...) | setting up and running the loop *on* an agent: tracing, verifiers, the harness, the gate, automation |
| [`self-improving-with-mlflow/`](self-improving-with-mlflow/SKILL.md) | OpenClaw itself, at runtime | the agent taking part in its *own* loop: its standing, user corrections, confirmed lessons, proposals |

## The self-improving-agent skill

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

### Install

For Claude Code, everywhere:

```bash
mkdir -p ~/.claude/skills && cp -r skills/self-improving-agent ~/.claude/skills/
```

or for one project, into that project's `.claude/skills/`. Any agent that reads skill folders
(OpenClaw, Hermes, Codex, Gemini CLI) can use the same folder. It pairs with MLflow's own skills
for the mechanics of tracing, scorers and trace analysis (`mlflow agent setup`, or this repo's
`.agents/skills/`).

### What's in it

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

### Tests

`self-improving-agent/evals/` holds test cases: realistic requests, each with the checks a good
answer passes (for example, the gate rejecting a candidate for length, an OpenClaw plugin that
sends no traces, and turning on auto-promote). With Claude Code, run them with and without the
skill from that folder:

```bash
claude plugin eval . --allow-tools Write,Edit,Bash
```

These are real, billed model runs.

## The self-improving-with-mlflow skill

[`self-improving-with-mlflow/`](self-improving-with-mlflow/SKILL.md) is for the agent being
improved. OpenClaw learns on its own by default (Skill Workshop, memory), and every one of those
changes applies at once, with no verifier. This skill puts MLflow's gate in front of that
learning: the agent can see which version it is and why, pass a user's correction to its owner,
recall the lessons a person confirmed, and propose changes to itself. It can't approve any of it.

Each capability is one `selfimprove self` command that prints JSON. The guardrails are in those
commands, not in the skill's prose, because the agent is the thing being improved:

| command | the agent uses it to | what stops it going wrong |
|---|---|---|
| `self status` | answer "which version are you, and why did you change?" | read-only; no holdout scores or rationales |
| `self note` | report that the user corrected it | writes `self_reported`, never `quality`; only a person's `queue accept` makes it a label |
| `self recall` | check confirmed lessons before a task | reads the lessons pack at `production`, which passed the gate; drops anything tied to the holdout |
| `self propose` | suggest an AGENTS.md or skill change | needs 2+ human-confirmed failures, none held out; length-checked; a candidate for the gate, or a queue item for a person |

Its owner runs the other side: `self lessons` (build the lessons pack as a gated candidate),
`self sync` (write production into the live workspace) and `self setup` (an isolated OpenClaw
profile with `read` + `exec` only, exec allowlisted to a wrapper that runs just those four
commands, and Skill Workshop off). See
[`references/operator.md`](self-improving-with-mlflow/references/operator.md) and the design,
[`docs/design/self-improving-with-mlflow.md`](../docs/design/self-improving-with-mlflow.md).

### Install

```bash
cd harness && make install
selfimprove self setup --workspace ~/my-agent-live
```

`self setup` copies the skill into `<workspace>/skills/`. To add it to an OpenClaw you already
run, copy the folder (without `evals/`) into its workspace's `skills/`, put `selfimprove` on its
exec path, and turn Skill Workshop off; the operator guide lists the settings.

### What's in it

| File | Covers |
|---|---|
| `SKILL.md` | when to recall, note, check status and propose; what goes to memory instead; the rules |
| `references/commands.md` | each command's flags, output and refusals |
| `references/operator.md` | for the owner: deploying, confirming notes, lessons, sync, settings |

### Tests

`harness/tests/test_agentside.py` and `test_agentside_loop.py` cover every refusal offline.
`self-improving-with-mlflow/evals/` holds skill evals: a user correcting the agent, "just remember
this", a refused proposal, and "why did you change?". Run them the same way as above.
