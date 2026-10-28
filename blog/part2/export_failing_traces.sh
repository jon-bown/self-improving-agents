#!/usr/bin/env bash
# Export failing traces for a coding agent
#
# From Self-Improvement & MLflow, Part 2:
# https://opendatascience.com/self-improvement-mlflow-part-2/
mlflow traces search --experiment-id 1 \
  --filter-string "feedback.quality = 'Fail'" \
  --output json > /tmp/failing_week37.json
