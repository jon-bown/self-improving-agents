"""Store: the curated evaluation dataset, its holdout slice, checkpoints and the human queue.

The dataset is the regression suite (Part 1, section 2.4). Every record carries provenance tags:

    source        golden | human_fail | queue | ...   where the case came from
    source_trace  the production trace it reproduces
    theme         the failure theme it tests (capped at SI_TESTS_PER_THEME_CAP per theme)
    author        who proposed it (triage-agent, a reviewer, ...)
    split         train | holdout

The holdout slice is chosen by a hash of the input, so a case lands in the same split forever.
No optimizer, analyst or approver ever sees it; only the gate does (Part 3, sections 6-7).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
import uuid

import mlflow
import yaml
from mlflow.genai.datasets import create_dataset, get_dataset, set_dataset_tags

from selfimprove import config, traces


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------
def dataset():
    s = config.settings
    try:
        return get_dataset(name=s.dataset_name)
    except Exception:
        exp = mlflow.get_experiment_by_name(s.experiment)
        return create_dataset(name=s.dataset_name, experiment_id=[exp.experiment_id],
                              tags={"purpose": "regression", "holdout_fraction": str(s.holdout_fraction)})


def tag_dataset(**tags) -> None:
    set_dataset_tags(dataset().dataset_id, {k: str(v) for k, v in tags.items()})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def split_for(question: str) -> str:
    h = int(hashlib.sha1(_norm(question).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "holdout" if h < config.settings.holdout_fraction else "train"


def holdout_questions() -> set[str]:
    return {_norm(question_of(r)) for r in records() if r["tags"].get("split") == "holdout"}


def is_holdout(question: str, known: set[str] | None = None) -> bool:
    """True for a holdout case, or an input that would land in the holdout when added."""
    known = holdout_questions() if known is None else known
    return _norm(question) in known or split_for(question) == "holdout"


def question_of(record: dict) -> str:
    return str((record.get("inputs") or {}).get(config.settings.input_key, ""))


def record(question: str, expectations: dict, source: str, author: str, **tags) -> dict:
    return {
        "inputs": {config.settings.input_key: question},
        "expectations": {k: v for k, v in expectations.items() if v not in (None, "", [])},
        "tags": {"source": source, "author": author, "split": tags.pop("split", None) or split_for(question),
                 "added_at": str(traces.now_ms()), **{k: str(v) for k, v in tags.items() if v is not None}},
    }


def record_from_trace(trace, source: str, author: str, theme: str | None = None) -> dict:
    """A dataset row from a labeled trace: its input, and the expectations a person attached."""
    return record(traces.question_of(trace), traces.expectations(trace), source=source, author=author,
                  source_trace=trace.info.trace_id, theme=theme,
                  prompt_version=trace.info.tags.get("si_prompt_version"))


def records() -> list[dict]:
    df = dataset().to_df()
    if df.empty:
        return []
    return [{"inputs": r["inputs"], "expectations": r.get("expectations") or {}, "tags": r.get("tags") or {}}
            for _, r in df.iterrows()]


def rows(split: str | None = "train") -> list[dict]:
    """{inputs, expectations} dicts for evaluate() / optimize_prompts(). split=None for all."""
    return [{"inputs": r["inputs"], "expectations": r["expectations"]} for r in records()
            if split is None or r["tags"].get("split", "train") == split]


def merge(new: list[dict]) -> int:
    if new:
        dataset().merge_records(new)
    return len(new)


def near_duplicate(question: str, existing: list[dict], threshold: float = 0.9) -> str | None:
    """The existing question this one nearly repeats, or None."""
    q = _norm(question)
    for r in existing:
        other = _norm(question_of(r))
        if q == other or difflib.SequenceMatcher(None, q, other).ratio() >= threshold:
            return question_of(r)
    return None


def count_in_theme(theme: str, existing: list[dict] | None = None) -> int:
    return sum(1 for r in (existing if existing is not None else records()) if r["tags"].get("theme") == theme)


def summary() -> str:
    rs = records()
    by_source: dict[str, int] = {}
    for r in rs:
        by_source[r["tags"].get("source", "?")] = by_source.get(r["tags"].get("source", "?"), 0) + 1
    holdout = sum(1 for r in rs if r["tags"].get("split") == "holdout")
    return f"{len(rs)} records ({len(rs) - holdout} train, {holdout} holdout) by source {by_source}"


# --------------------------------------------------------------------------------------------
# Golden cases (hand-written; Part 2, section 2)
# --------------------------------------------------------------------------------------------
def golden() -> list[dict]:
    path = config.settings.golden_file
    if path is None or not path.is_file():
        return []
    out = []
    for case in yaml.safe_load(path.read_text()) or []:
        split = "holdout" if case.get("holdout") else ("train" if "holdout" in case else None)
        out.append(record(case["inputs"][config.settings.input_key], case.get("expectations") or {},
                          source="golden", author="human", split=split, case_id=case.get("id")))
    return out


def seed_golden() -> int:
    existing = records()
    return merge([r for r in golden() if not near_duplicate(question_of(r), existing, threshold=1.0)])


# --------------------------------------------------------------------------------------------
# Checkpoints: "what has this job already processed?" (Part 3, section 2)
# --------------------------------------------------------------------------------------------
def load_checkpoint(job: str) -> int:
    exp = mlflow.get_experiment_by_name(config.settings.experiment)
    return int(exp.tags.get(f"si.checkpoint.{job}", "0")) if exp else 0


def save_checkpoint(job: str, ms: int | None = None) -> None:
    mlflow.set_experiment_tag(f"si.checkpoint.{job}", str(ms or traces.now_ms()))


# --------------------------------------------------------------------------------------------
# The human queue: everything the automation refuses to decide on its own
# --------------------------------------------------------------------------------------------
QUEUE_KINDS = ("contradiction", "needs_expectation", "flagged_test", "judge_only_failure", "halt")


def _queue_file():
    p = config.settings.state_dir / "human_queue.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def queue_items(status: str | None = "open") -> list[dict]:
    p = _queue_file()
    if not p.is_file():
        return []
    items: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["id"]] = {**items.get(item["id"], {}), **item}  # later lines update earlier ones
    return [i for i in items.values() if status is None or i.get("status") == status]


def enqueue(kind: str, reason: str, trace_id: str | None = None, record: dict | None = None, **extra) -> str:
    """Send something to a person. Deduplicated on (kind, trace_id, question)."""
    assert kind in QUEUE_KINDS, kind
    key = (kind, trace_id, question_of(record) if record else None)
    for item in queue_items(status=None):
        if (item["kind"], item.get("trace_id"), question_of(item["record"]) if item.get("record") else None) == key:
            return item["id"]
    item = {"id": uuid.uuid4().hex[:8], "kind": kind, "reason": reason, "trace_id": trace_id, "record": record,
            "status": "open", "created": int(time.time()), **extra}
    with _queue_file().open("a") as f:
        f.write(json.dumps(item, default=str) + "\n")
    return item["id"]


def resolve(item_id: str, status: str, note: str = "", **extra) -> dict:
    """Close a queue item (accepted / rejected / edited). Accepted and edited records join the dataset."""
    item = next(i for i in queue_items(status=None) if i["id"] == item_id)
    update = {"id": item_id, "status": status, "note": note, "resolved": int(time.time()), **extra}
    with _queue_file().open("a") as f:
        f.write(json.dumps(update, default=str) + "\n")
    item.update(update)
    if status in ("accepted", "edited") and item.get("record"):
        rec = item["record"]
        rec["tags"] = {**rec.get("tags", {}), "source": "queue", "approved_by": "human",
                       "human_edited": str(status == "edited").lower()}
        merge([rec])
    return item
