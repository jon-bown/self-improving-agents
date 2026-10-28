"""Compare v1 and v2 on the dataset

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: agent is your agent; dataset comes from regression_dataset.py;
cites_current_policy and leads_with_answer come from Part 1's scorers.py.
"""

from mlflow.genai.scorers import Correctness


def predict_with(version: int):
    prompt = mlflow.genai.load_prompt(f"prompts:/support-system-prompt/{version}")

    def predict(question: str) -> str:
        return agent.invoke(prompt.format(product="Acme", question=question))
    return predict


scorers = [cites_current_policy, leads_with_answer, Correctness()]

v1 = mlflow.genai.evaluate(
    data=dataset.df, predict_fn=predict_with(1), scorers=scorers
)
v2 = mlflow.genai.evaluate(
    data=dataset.df, predict_fn=predict_with(2), scorers=scorers
)

for name in ("cites_current_policy/mean", "leads_with_answer/mean"):
    print(name, v1.metrics[name], "->", v2.metrics[name])
