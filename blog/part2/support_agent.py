"""Version the prompt and load it by alias

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: agent and route_topic are your own agent's code.
"""

import mlflow
from mlflow.entities import SpanType

mlflow.set_experiment("support-agent")
mlflow.openai.autolog()

v1 = mlflow.genai.register_prompt(
    name="support-system-prompt",
    template="You are a support agent for {{ product }}. Answer the customer's "
             "question using the policy tools available to you. "
             "Question: {{ question }}",
    commit_message="v1, initial prompt",
)

# The agent loads by alias, so promotion is a pointer move, not a code change.
mlflow.genai.set_prompt_alias(
    name="support-system-prompt", alias="production", version=v1.version
)


@mlflow.trace(span_type=SpanType.AGENT)
def run_agent(question: str) -> str:
    prompt = mlflow.genai.load_prompt(
        "prompts:/support-system-prompt@production"
    )
    mlflow.update_current_trace(tags={"topic": route_topic(question)})
    return agent.invoke(prompt.format(product="Acme", question=question))
