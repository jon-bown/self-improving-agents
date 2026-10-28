"""Judge last week's traces in place

From Self-Improvement & MLflow, Part 2:
https://opendatascience.com/self-improvement-mlflow-part-2/

Not defined here: leads_with_answer comes from Part 1's scorers.py.
"""

import time

since = int((time.time() - 7 * 24 * 3600) * 1000)
recent = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {since}",
    return_type="list",
)

# Judge the week's traces in place. No predict_fn, so nothing is re-run.
mlflow.genai.evaluate(data=recent, scorers=[leads_with_answer])
