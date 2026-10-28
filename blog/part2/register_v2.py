"""Register a candidate prompt

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/
"""

v2 = mlflow.genai.register_prompt(
    name="support-system-prompt",
    template="You are a support agent for {{ product }}. State the direct answer in "
             "your first sentence, then add any conditions. Refund policy, effective "
             "July 2026, is 14 days from purchase for all plans. "
             "Question: {{ question }}",
    commit_message="v2, lead with the answer; current refund window "
                   "(evidence: traces tr-7f3a9c, tr-b21e40)",
)
