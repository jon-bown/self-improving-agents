# MLflow recipes, without the harness

Each step of the loop in plain MLflow (3.17+). Names match the harness, so the two interoperate:
feedback `quality`, expectation `expected_response`, trace tags `si_source` and `si_prompt_version`.

## 1. Instructions in the Prompt Registry, loaded by alias

```python
import mlflow

v1 = mlflow.genai.register_prompt(name="my-agent-instructions", template=open("AGENTS.md").read(),
                                  commit_message="v1: initial instructions")
mlflow.genai.set_prompt_alias("my-agent-instructions", alias="production", version=v1.version)

prompt = mlflow.genai.load_prompt("prompts:/my-agent-instructions@production")  # in the agent
```

Use `prompt.template` as raw text when the instructions contain `{{ }}`.

## 2. One trace per request, tagged with the version

```python
from mlflow.entities import SpanType

@mlflow.trace(name="my_agent", span_type=SpanType.AGENT)
def run(question: str, ref: str = "production", source: str = "production") -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/my-agent-instructions@{ref}")  # links prompt <-> trace
    mlflow.update_current_trace(tags={"si_source": source, "si_prompt_version": str(prompt.version)})
    return my_agent(question, prompt.template)
```

Evaluation, optimizer and CI traffic must use `source="eval"` so it never comes back as signal.
Tool calls should be `SpanType.TOOL` spans; scorers find them with `trace.search_spans(span_type=...)`.

## 3. Human signal

```python
from mlflow.entities import AssessmentSource, AssessmentSourceType

me = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="jon")
mlflow.log_feedback(trace_id=tid, name="quality", value=False, rationale="Quoted the old policy.", source=me)
mlflow.log_expectation(trace_id=tid, name="expected_response", value="Within 14 days.", source=me)
```

Boolean feedback is stored as text: filter with `feedback.quality = 'false'`.

## 4. Automated signal on existing traces (no predict_fn: nothing re-runs)

```python
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines, scorer

@scorer
def no_stale_policy(outputs) -> bool:
    return "30 days" not in str(outputs)

leads = Guidelines(name="leads_with_answer", guidelines="The first sentence must answer the question.")
quality = make_judge(name="quality", feedback_value_type=bool,
                     instructions="Request: {{ inputs }}\nResponse: {{ outputs }}\nWould the owner accept this as-is?")

recent = mlflow.search_traces(filter_string="tags.si_source = 'production'", return_type="list")
mlflow.genai.evaluate(data=recent, scorers=[no_stale_policy, leads, quality])
```

A cheap first pass with a decision model (MLflow 3.17+, `TYPESAFE_API_KEY`). Jev returns a
yes/no with a probability and no rationale, and cannot read `{{ trace }}`, so check spans in code
first and pass the tool output in as inputs. Re-judge the uncertain band with a text judge:

```python
import json
from mlflow.entities import Feedback, SpanStatusCode, SpanType

jev = make_judge(name="grounded_in_policy", feedback_value_type=bool, model="typesafe:/jev-latest",
                 instructions="Answer yes or no. Does {{ outputs }} make only claims that the policy text in {{ inputs }} supports?")
text = make_judge(name="grounded_in_policy", feedback_value_type=bool, model="anthropic:/claude-opus-5-5",
                  instructions="Does {{ outputs }} make only claims that the policy text in {{ inputs }} supports?")

@scorer
def grounded_in_policy(trace):
    calls = trace.search_spans(span_type=SpanType.TOOL, name="lookup_policy")
    if not calls or calls[-1].status.status_code == SpanStatusCode.ERROR or not calls[-1].outputs:
        return Feedback(value=False, rationale="No successful lookup_policy call behind this answer.")
    args = dict(inputs={"question": trace.data.request, "policy": calls[-1].outputs}, outputs=trace.data.response)
    first = jev(**args)
    p = json.loads(first.metadata["typesafe.probability"])
    return text(**args) if 0.5 <= p <= 0.8 else first      # pick the band from your own labels
```

Server-side sampling instead (needs `mlflow server` with a database):

```python
from mlflow.genai.scorers import ScorerSamplingConfig
leads.register(name="leads_with_answer").start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.1, filter_string="tags.si_source = 'production'"))
```

## 5. The dataset, with provenance and a holdout

```python
from mlflow.genai.datasets import create_dataset, set_dataset_tags

ds = create_dataset(name="my-agent-regressions", experiment_id=[exp_id])
failing = mlflow.search_traces(
    filter_string="feedback.quality = 'false' AND expectation.expected_response IS NOT NULL "
                  "AND tags.si_source = 'production'", return_type="list")
ds.merge_records([{
    "inputs": {"question": question_of(t)},
    "expectations": {"expected_response": expected_of(t)},
    "tags": {"source": "human_fail", "source_trace": t.info.trace_id, "theme": theme,
             "split": "holdout" if stable_hash(question_of(t)) < 0.2 else "train"},
} for t in failing])
set_dataset_tags(ds.dataset_id, {"generated_by": "triage-agent", "tests_per_theme_cap": "3"})
```

Assign the split by a hash of the input so a case never changes split. Check every new record
against reviewer disagreement first (two humans, two verdicts: send to a person).

## 6. Measure, then gate

```python
def predict_with(version):
    def predict(question: str) -> str:
        return run(question, ref=str(version), source="eval")
    return predict

rows = ds.to_df()
train = [r for _, r in rows.iterrows() if r["tags"].get("split") != "holdout"]
holdout = [r for _, r in rows.iterrows() if r["tags"].get("split") == "holdout"]
base = mlflow.genai.evaluate(data=train, predict_fn=predict_with(1), scorers=scorers)
cand = mlflow.genai.evaluate(data=train, predict_fn=predict_with(2), scorers=scorers)
```

Promote only if must-pass scorers are at 100%, the candidate is not worse on train, and it beats
production on the holdout by the margin. Then:

```python
mlflow.genai.set_prompt_alias("my-agent-instructions", alias="previous", version=1)
mlflow.genai.set_prompt_alias("my-agent-instructions", alias="production", version=2)
mlflow.genai.set_prompt_version_tag("my-agent-instructions", 2, "replaced_version", "1")
```

In MLflow 3.16-3.17, `EvaluationResult.passed` also treats expectation columns as assertions;
check the scorers' own `<name>/value` columns instead.

## 7. A regression suite in CI

```python
import pytest

@mlflow.test
@pytest.mark.parametrize("row", train + holdout, ids=lambda r: r["inputs"]["question"][:40])
def test_regressions(row):
    result = mlflow.genai.evaluate(data=[row], predict_fn=predict_with("production"), scorers=scorers)
    assert not scorer_failures(result, scorers)
```

Run with `pytest -p mlflow.pytest.plugin`.

## 8. Align the judge

```python
from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer   # MemAlignOptimizer with embeddings

labeled = [t for t in mlflow.search_traces(filter_string="feedback.quality IS NOT NULL", return_type="list")]
fit, check = split_by_trace_id(labeled, 0.3)
before = agreement(quality, check)
if len(labeled) >= 10 and before < 0.75:
    aligned = quality.align(fit, optimizer=SIMBAAlignmentOptimizer(model="anthropic:/claude-opus-5-5"))
    if agreement(aligned, check) > before:
        aligned.register(name="quality")
```

## 9. Optimize the prompt

```python
from mlflow.genai.optimize import GepaPromptOptimizer   # MetaPromptOptimizer for little data

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_with(1),          # must load the prompt inside each call
    train_data=train,                    # never the holdout
    prompt_uris=["prompts:/my-agent-instructions/1"],
    optimizer=GepaPromptOptimizer(reflection_model="anthropic:/claude-opus-5-5", max_metric_calls=60),
    scorers=scorers,
)
candidate = result.optimized_prompts[0]   # registered as a new version; gate it like any other
```
