"""Level 5 · Continuous signal: judges score production traffic server-side as it arrives.

Registered scorers with a sampling config are run by the MLflow server's job scheduler on new
traces (OSS: needs `mlflow server` with a database backend, which `make server` starts).
The loop no longer has to wait for someone to run Level 1 -- signal accumulates on its own.

Custom @scorer functions can't be registered on open-source MLflow (they would need to be
deserialized as code), so the deterministic checks run inside the loop instead.

    python levels/level5_autonomous/01_monitor.py [--sample-rate 1.0] [--stop]
"""

import argparse

from mlflow.genai.scorers import ScorerSamplingConfig, get_scorer

from sia import config, llm, signals, tracking

parser = argparse.ArgumentParser()
parser.add_argument("--sample-rate", type=float, default=1.0)
parser.add_argument("--stop", action="store_true")
args = parser.parse_args()

if not config.HAS_LLM:
    raise SystemExit(f"Online judges need an LLM: {llm.missing_key_hint(config.JUDGE_MODEL)} (see .env.example).")
tracking.setup()

try:
    quality = get_scorer(name="quality")          # aligned in Level 4, if you ran it
except Exception:
    quality = signals.quality_judge()

monitors = [quality] + signals.judge_scorers() + signals.theme_scorers()
sampling = ScorerSamplingConfig(sample_rate=args.sample_rate, filter_string="tags.source = 'production'")
for s in monitors:
    if args.stop:
        s.stop(name=s.name)
        print(f"stopped  {s.name}")
    else:
        s.register(name=s.name).start(name=s.name, sampling_config=sampling)
        print(f"started  {s.name}  (sample_rate={args.sample_rate})")
