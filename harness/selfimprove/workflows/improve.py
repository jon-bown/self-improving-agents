"""Write a candidate (Part 3, sections 4 and 7). The update step; it never promotes.

SI_IMPROVER picks who writes it:
    analyst     an LLM reads the failure report and rewrites the instructions (Part 2, section 6)
    gepa        GepaPromptOptimizer: evolves variants by reflecting on scorer rationales; thorough
    metaprompt  MetaPromptOptimizer: restructures the prompt with a metaprompt; cheap, little data
    command     a coding agent of your choice edits the file (Claude Code, OpenClaw, Codex, ...)
    from_file   a person wrote it (the manual loop, Part 2 section 3): `improve --from-file v2.md`

Every improver sees the train split and production failures, never the holdout. The candidate is
registered as a new prompt version, tagged with who wrote it and from which run.
"""

from __future__ import annotations

from selfimprove import config, improvers, instructions, lessons, signals, store, traces
from selfimprove.workflows import Result, finish, job
from selfimprove.workflows.triage import judge_trusted


def improve(improver: str | None = None, budget: int = 60, from_file: str | None = None,
            message: str | None = None) -> Result:
    s = config.settings
    kind = "human" if from_file else (improver or s.improver).lower()
    with job("improve", improver=kind) as run:
        prod = instructions.load("production")
        if from_file:
            text = open(from_file).read()
            if text.strip() == prod.template.strip():
                return finish(Result("improve", "no_change", f"{from_file} matches production"))
            version = instructions.register_candidate(text, message or f"hand-written: {from_file}", author="human",
                                                      source_run=run.info.run_id, base_version=prod.version)
            return finish(Result("improve", "candidate", f"v{version} from {from_file}",
                                 {"candidate_version": version}, {"version": version}))
        holdout_qs = store.holdout_questions()
        live = [t for t in traces.production(f"tags.si_prompt_version = '{prod.version}'")
                if not store.is_holdout(traces.question_of(t), holdout_qs)]
        report = improvers.failure_report(live)
        if not judge_trusted():  # while the judge disagrees with people, only human failures count
            report = [r for r in report if r["human_failed"]]
        train = store.rows("train")
        if not report and not train:
            return finish(Result("improve", "no_signal", "no failures and no dataset rows to learn from"))
        if kind in ("analyst", "gepa", "metaprompt") and not s.has_llm:
            return finish(Result("improve", "skipped", f"SI_IMPROVER={kind} needs credentials for {s.judge_model}"))

        if kind == "analyst":
            proposal = improvers.analyze(report, prod.template)
            text, why = proposal["instructions"], proposal["change_summary"]
        elif kind in ("gepa", "metaprompt"):
            text, why = improvers.optimize(train, signals.live() + signals.reference(), kind=kind, budget=budget)
        elif kind == "command":
            text, why = improvers.coding_agent(report, prod.template, label=run.info.run_id[:8])
        else:
            raise ValueError(f"SI_IMPROVER={kind!r}: choose analyst, gepa, metaprompt or command")
        text = lessons.keep(text, prod.template)  # lessons change only through `self lessons`

        if text.strip() == prod.template.strip():
            return finish(Result("improve", "no_change", f"{kind} proposed no change"))
        version = instructions.register_candidate(
            text, f"{kind}: {why}", author=f"improver:{kind}", source_run=run.info.run_id,
            base_version=prod.version, failures_seen=len(report))
        return finish(Result("improve", "candidate", f"v{version} from {kind} ({len(report)} failures, "
                             f"{len(train)} train rows): {why[:200]}",
                             {"candidate_version": version, "failures_seen": len(report)},
                             {"version": version}))
