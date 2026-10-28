"""5. Automate the Alignment

From Self-Improvement & MLflow, Part 3: Automate via CI/CD:
https://opendatascience.com/self-improvement-mlflow-part-3/

Not defined here: quality_judge, experiment_id, AGREEMENT_THRESHOLD, split_traces and
agreement_with_humans are your own.
"""

from mlflow.genai.judges.optimizers import MemAlignOptimizer

paired = mlflow.search_traces(filter_string="feedback.quality IS NOT NULL",
                              return_type="list")
fit, check = split_traces(paired, check_fraction=0.3)   # our helper

before = agreement_with_humans(quality_judge, check)    # our helper
if len(paired) >= 10 and before < AGREEMENT_THRESHOLD:
    aligned = quality_judge.align(
        fit, MemAlignOptimizer(reflection_lm="openai:/gpt-5-mini"))
    if agreement_with_humans(aligned, check) > before:
        aligned.register(experiment_id=experiment_id)
# Can assign an alias here if you're confident in the loop
