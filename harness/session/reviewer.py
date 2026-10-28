"""Simulated reviewers for the session: support leads who know Acme's policies.

They grade production traces against the answer key in cases.yaml. In the room, you play this
part: label a few traces in the MLflow UI (or `selfimprove label`) and let these three do the
rest. A failure gets the correct answer attached as its expected response, which is what the
store step turns into a test.

    python -m session.reviewer            label every unlabeled production trace
    python -m session.reviewer --sample 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REVIEWERS = ("maria", "dev", "sam")


def cases() -> dict[str, dict]:
    return {c["question"].strip().lower(): c for c in yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]}


def verdict(case: dict, response: str) -> tuple[bool, str]:
    text = response.strip()
    for p in case.get("must_not_include", []):
        if re.search(p, text, re.IGNORECASE | re.MULTILINE):
            return False, f"Says /{p}/, which is wrong or against policy. Correct: {case['expected_response']}"
    for p in case.get("must_include", []):
        if not re.search(p, text, re.IGNORECASE):
            return False, f"Missing a key fact (/{p}/). Correct: {case['expected_response']}"
    return True, "Correct and within policy."


def review_all(sample: float = 1.0) -> dict[str, list[int]]:
    from selfimprove import feedback, traces

    key, counts = cases(), {r: [0, 0] for r in REVIEWERS}
    for t in feedback.unlabeled(limit=500):
        h = int(hashlib.sha1(t.info.trace_id.encode()).hexdigest()[:8], 16)
        if (h % 1000) / 1000 >= sample:
            continue
        case = key.get(traces.question_of(t).strip().lower())
        if case is None:
            continue
        reviewer = REVIEWERS[h % len(REVIEWERS)]
        ok, why = verdict(case, traces.response_of(t))
        feedback.label(t.info.trace_id, passed=ok, rationale=why, reviewer=reviewer,
                       expected_response=None if ok else case["expected_response"])
        counts[reviewer][0 if ok else 1] += 1
    return counts


if __name__ == "__main__":
    from selfimprove import traces

    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=float, default=1.0)
    args = p.parse_args()
    traces.setup()
    for name, (ok, bad) in review_all(args.sample).items():
        print(f"{name:<6} {ok:>3} pass {bad:>3} fail")
