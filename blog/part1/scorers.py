"""Turn the feedback into scorers

From Self-Improvement & MLflow, Part 1:
https://opendatascience.com/self-improvement-vs-self-evolution-mlflow-part-1/
"""

from mlflow.genai.scorers import Guidelines, scorer


@scorer
def cites_current_policy(outputs) -> bool:
    return "14 days" in str(outputs) and "30 days" not in str(outputs)


leads_with_answer = Guidelines(
    name="leads_with_answer",
    guidelines="The response must state the refund window in the first sentence. "
               "PASS if it does; FAIL if it opens with caveats.",
)
