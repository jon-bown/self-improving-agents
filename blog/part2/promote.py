"""Promote by moving the alias

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: v2 comes from register_v2.py.
"""

mlflow.genai.set_prompt_alias(
    name="support-system-prompt", alias="production", version=v2.version
)
