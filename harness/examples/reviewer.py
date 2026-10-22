"""Simulated reviewers, for rehearsing the loop without labeling by hand.

In real use, you (or your reviewers) label traces with `selfimprove label`, the MLflow review
queue, or feedback.label(). Here three reviewers share the answer key in cases.yaml but grade with
slightly different rubrics, on purpose, because that is what real reviewer pools do (Part 2,
section 5): maria fails disclaimers, dev only checks the facts, sam also fails long answers.

    python -m examples.reviewer            label a sample of unlabeled production traces
    python -m examples.reviewer --all      label all of them
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REVIEWERS = ("maria", "dev", "sam")


def _cases() -> dict[str, dict]:
    return {c["question"].strip().lower(): c for c in yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]}


def verdict(reviewer: str, case: dict, response: str) -> tuple[bool, str]:
    for p in case.get("must_not_include", []):
        if re.search(p, response, re.IGNORECASE):
            return False, f"Says /{p}/, which is wrong or outdated. Correct: {case['expected_response']}"
    for p in case.get("must_include", []):
        if not re.search(p, response, re.IGNORECASE):
            return False, f"Missing a key fact (/{p}/). Correct: {case['expected_response']}"
    if reviewer == "maria" and re.search(r"terms of service|policies can vary", response, re.I):
        return False, "Buries the answer under a disclaimer."
    if reviewer == "sam" and len(response.split()) > 90:
        return False, "Too long for a support reply."
    return True, "Correct and to the point."


def review_all(sample: float = 1.0) -> dict[str, list[int]]:
    from selfimprove import feedback, traces

    cases, counts = _cases(), {r: [0, 0] for r in REVIEWERS}
    for t in feedback.unlabeled(limit=500):
        h = int(hashlib.sha1(t.info.trace_id.encode()).hexdigest()[:8], 16)
        if (h % 1000) / 1000 >= sample:
            continue
        case = cases.get(traces.question_of(t).strip().lower())
        if case is None:
            continue
        reviewer = REVIEWERS[h % len(REVIEWERS)]
        ok, why = verdict(reviewer, case, traces.response_of(t))
        feedback.label(t.info.trace_id, passed=ok, rationale=why, reviewer=reviewer,
                       expected_response=None if ok else case["expected_response"])
        counts[reviewer][0 if ok else 1] += 1
    return counts


if __name__ == "__main__":
    from selfimprove import traces

    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--sample", type=float, default=0.6)
    args = p.parse_args()
    traces.setup()
    for name, (ok, bad) in review_all(1.0 if args.all else args.sample).items():
        print(f"{name:<6} {ok:>3} pass {bad:>3} fail")
