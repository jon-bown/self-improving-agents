"""3. Amplify the Signal

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/
"""

from mlflow.entities import Feedback, SpanStatusCode, SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer, ScorerSamplingConfig

# export TYPESAFE_API_KEY=...
policy_judge = make_judge(
    name="grounded_in_policy",
    instructions=(
        "Answer yes or no. Does {{ outputs }} make only claims that the "
        "policy text in {{ inputs }} supports?"
    ),
    model="typesafe:/jev-latest",
    feedback_value_type=bool,
)

@scorer
def grounded_in_policy(trace):
    lookups = trace.search_spans(span_type=SpanType.TOOL, name="lookup_policy")
    if not lookups:
        return Feedback(value=False,
                        rationale="The agent answered without calling lookup_policy.")

    call = lookups[-1]
    if call.status.status_code == SpanStatusCode.ERROR:
        return Feedback(value=False,
                        rationale="lookup_policy failed, so the answer has no policy behind it.")
    if not (call.inputs or {}).get("topic") or not call.outputs:
        return Feedback(value=False,
                        rationale="lookup_policy was called without a topic or returned nothing.")

    return policy_judge(
        inputs={"question": trace.data.request, "policy": call.outputs},
        outputs=trace.data.response,
    )

registered = grounded_in_policy.register(name="grounded_in_policy")
registered.start(
    sampling_config=ScorerSamplingConfig(
        sample_rate=0.1,
        filter_string="tags.environment = 'production'",
    )
)

# Judge verdicts carry metadata["typesafe.probability"]
