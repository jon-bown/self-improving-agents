"""7. Improve Other Components: judges for a skill

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/
"""

from typing import Literal
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

skill_loaded = make_judge(
    name="tracing-skill-invoked",
    instructions=(
        "Answer yes or no. Does {{ trace }} show the agent loading the "
        "instrumenting-with-mlflow-tracing skill before it edited any code?"
    ),
    model="openai:/gpt-4o-mini",
    feedback_value_type=Literal["yes", "no"],
)

@scorer
def dataset_created(trace):
    eval_experiment_id = trace.info.tags["eval_experiment_id"]   # set by run_claude, one fresh experiment per run
    found = MlflowClient().search_datasets(experiment_ids=[eval_experiment_id])
    return Feedback(
        value=bool(found),
        rationale="An evaluation dataset exists." if found
        else "The agent never created an evaluation dataset.",
    )
