"""An example agent to bring: a support agent for a fictional product, Acme Cloud.

It is wired in with two lines of .env:

    SI_AGENT=python
    SI_AGENT_TARGET=examples.support_agent:answer

`answer()` is all the harness needs. The instructions it receives are whatever version of
examples/instructions.md the harness is testing. The policies live in the workspace's
knowledge/ folder and are read with a `lookup_policy` tool, traced as a TOOL span so scorers can
check whether the agent looked before it answered (Part 3, section 3).

v1 of the instructions has three seeded problems: stale quick facts (30-day refunds), a
mandatory Terms-of-Service disclaimer up front, and "give your best guess".

With no model credentials (or SI_EXAMPLE_OFFLINE=1) it falls back to `answer_offline()`, a
rule-based stand-in that obeys a few instructions, so the whole loop can be rehearsed for free.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

TOPICS = ("billing", "account_access", "product_howto", "support_process")
HERE = Path(__file__).parent
MODEL = os.getenv("SI_AGENT_MODEL", "anthropic:/claude-haiku-5-5")

LOOKUP_TOOL = {"type": "function", "function": {
    "name": "lookup_policy",
    "description": "Read Acme Cloud's current policy for one topic. The policy is the source of truth.",
    "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(TOPICS)}},
                   "required": ["topic"]},
}}


def _knowledge(workspace: Path | None) -> Path:
    for folder in ((workspace / "knowledge") if workspace else None, HERE / "knowledge"):
        if folder is not None and folder.is_dir():
            return folder
    raise FileNotFoundError("no knowledge/ folder")


@mlflow.trace(name="lookup_policy", span_type=SpanType.TOOL)
def lookup_policy(topic: str, workspace: Path | None = None) -> str:
    path = _knowledge(workspace) / f"{topic}.md"
    if topic not in TOPICS or not path.is_file():
        raise ValueError(f"unknown topic {topic!r}; choose one of {TOPICS}")
    return path.read_text()


def answer(question: str, instructions: str, workspace: Path | None = None) -> str:
    from selfimprove import models

    if os.getenv("SI_EXAMPLE_OFFLINE") == "1" or not models.has_credentials(MODEL):
        return answer_offline(question, instructions, workspace)
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": question}]
    for _ in range(5):
        msg = models.chat(MODEL, messages, tools=[LOOKUP_TOOL], max_tokens=1500)
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            topic = json.loads(call.function.arguments or "{}").get("topic", "")
            try:
                result = lookup_policy(topic, workspace)
            except ValueError as e:
                result = str(e)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Sorry, I couldn't finish looking that up."


# --------------------------------------------------------------------------------------------
# Offline stand-in: no LLM, deterministic, for rehearsals and CI
# --------------------------------------------------------------------------------------------
def _case(question: str):
    import yaml

    cases = yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]
    return next((c for c in cases if c["question"].strip().lower() == question.strip().lower()), None)


def answer_offline(question: str, instructions: str, workspace: Path | None = None) -> str:
    """Reacts to a few instructions the way a small model tends to:

    - told to look up the policy before answering -> calls lookup_policy and answers correctly
      (only for topics it is told to look up, if the instructions name topics)
    - otherwise                                    -> answers from the quick facts, or guesses
    - a Terms-of-Service reminder up front          -> opens every answer with it
    """
    text = instructions.lower()
    case = _case(question)
    looks_up = re.search(r"(always|before answering|first)[^.\n]*lookup_policy|lookup_policy[^.\n]*(always|every|first)",
                         text) and "only call `lookup_policy` if" not in text
    named = [t for t in TOPICS if t in text]
    if case is not None and looks_up and (not named or case["topic"] in named):
        lookup_policy(case["topic"], workspace)
        body = case["expected_response"]
    elif re.search(r"refund", question, re.I) and "30 days" in text:
        body = "You can get a refund within 30 days of purchase."
    elif re.search(r"reset", question, re.I) and "24 hours" in text:
        body = "Password reset links are valid for 24 hours."
    elif "best guess" in text:
        body = "Yes, that should be available on your plan. It usually takes about 24 hours."
    else:
        body = "I don't have that information, so I'm escalating this to a human teammate."
    if "terms of service" in text and "double-check" in text:
        body = "Policies can vary by plan and region, so please double-check the Terms of Service. " + body
    return body
