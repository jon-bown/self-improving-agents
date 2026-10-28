"""Trace the agent

From Self-Improvement & MLflow, Part 1:
https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/

Not defined here: agent, policies and route_topic are your own agent's code.
"""

import mlflow
from mlflow.entities import SpanType

mlflow.set_experiment("support-agent")
mlflow.openai.autolog()          # or langchain, anthropic, pydantic_ai, ...


@mlflow.trace(span_type=SpanType.TOOL)
def lookup_refund_policy(plan: str) -> str:
    return policies[plan]            # a child span, typed so judges can find it


@mlflow.trace(span_type=SpanType.AGENT)
def run_agent(question: str) -> str:
    mlflow.update_current_trace(tags={"topic": route_topic(question)})
    return agent.invoke(question)
