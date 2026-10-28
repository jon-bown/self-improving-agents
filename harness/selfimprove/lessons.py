"""The lessons pack: confirmed lessons from past failures, versioned *inside* the instructions.

The pack is one delimited block in the registered prompt, so the gate scores instructions and
lessons together, promotion moves both, and a rollback takes both back (design D1-B, Q4). The
block is rendered into the agent's workspace as its own file, LESSONS.md, and AGENTS.md keeps a
one-line pointer in its place, so the instructions stay short and `self recall` can pick out the
few lessons that apply to a request.

    <!-- si:lessons begin ... -->
    ## Lessons
    - [L3fa2] Read knowledge/refunds.md before quoting a refund window. <!-- si:{"theme": ...} -->
    <!-- si:lessons end -->

Only `selfimprove self lessons` writes the block, from human-confirmed failures on the train
split. The metadata comment on each line (theme, evidence count, keywords, source traces) is
what `recall` matches on and what lets it drop a lesson whose evidence later moved to the holdout.
"""

from __future__ import annotations

import hashlib
import json
import re

BEGIN = "<!-- si:lessons begin (written by `selfimprove self lessons`; changes go through the gate) -->"
END = "<!-- si:lessons end -->"
FILENAME = "LESSONS.md"
POINTER = ("Lessons from past mistakes, each confirmed by a person, are in `LESSONS.md`. Check the ones "
           "that fit before answering questions about policies, numbers or procedures.")
HEADER = ("## Lessons\n\nEach lesson comes from answers a person marked wrong. Follow its pointer to the "
          "source rather than reciting it, and say so if one doesn't fit the case in front of you.\n")

_BLOCK = re.compile(re.escape(BEGIN.split(" (")[0]) + r".*?" + re.escape(END), re.DOTALL)
_LINE = re.compile(r"^- \[(?P<id>L[0-9a-f]+)\] (?P<text>.*?)(?: <!-- si:(?P<meta>\{.*\}) -->)?\s*$")
_STOP = set("""a an and are as at be been but by can could did do does for from had has have how i if in
is it its me my of on or our should so that the their them then there these this to was we were what
when where which who why will with would you your about any get got just like not no yes please
tell know need want""".split())


def lesson_id(theme: str) -> str:
    return "L" + hashlib.sha1(theme.encode()).hexdigest()[:5]


def tokens(text: str) -> list[str]:
    """Lower-case content words, with a crude plural fold (refunds -> refund)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", str(text).lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def keywords(texts: list[str], n: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for t in texts:
        for w in set(tokens(t)):
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


# --------------------------------------------------------------------------------------------
# The block inside a prompt template
# --------------------------------------------------------------------------------------------
def block(template: str) -> str | None:
    m = _BLOCK.search(template)
    return m.group(0) if m else None


def render(lessons: list[dict]) -> str:
    lines = [BEGIN, HEADER]
    for les in lessons:
        meta = json.dumps({k: les[k] for k in ("theme", "evidence", "keys", "traces") if k in les})
        lines.append(f"- [{les['id']}] {' '.join(les['lesson'].split())} <!-- si:{meta.replace('-->', '')} -->")
    lines.append(END)
    return "\n".join(lines)


def parse(template: str) -> list[dict]:
    b = block(template)
    if not b:
        return []
    out = []
    for line in b.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        meta = json.loads(m.group("meta")) if m.group("meta") else {}
        out.append({"id": m.group("id"), "lesson": m.group("text").strip(), **meta})
    return out


def replace(template: str, lessons: list[dict]) -> str:
    """The template with its lessons block set to `lessons` (removed when empty, appended when new)."""
    new = render(lessons) if lessons else ""
    if block(template):
        out = _BLOCK.sub(lambda _: new, template, count=1)
    else:
        out = template.rstrip("\n") + ("\n\n" + new if new else "")
    return re.sub(r"\n{3,}", "\n\n", out).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------
# Workspace form: AGENTS.md with a pointer, and LESSONS.md
# --------------------------------------------------------------------------------------------
def split(template: str) -> tuple[str, str | None]:
    """(instructions file text, LESSONS.md text or None) for writing into a workspace."""
    b = block(template)
    if not b:
        return template, None
    body = "\n".join(line for line in b.splitlines() if line not in (BEGIN, END)).strip()
    lessons = "\n".join(re.sub(r" <!-- si:\{.*\} -->\s*$", "", line) for line in body.splitlines())
    return template.replace(b, POINTER), lessons + "\n"


def join(instructions_text: str, production_template: str) -> str:
    """Put production's lessons block back into an instructions file written in workspace form.

    The agent edits the AGENTS.md it can see, which has the pointer in place of the block. Its
    proposal keeps production's lessons exactly: lessons only change through `self lessons`.
    """
    b = block(production_template)
    if not b:
        return instructions_text
    if POINTER in instructions_text:
        return instructions_text.replace(POINTER, b, 1)
    return instructions_text.rstrip("\n") + "\n\n" + b + "\n"


def keep(new_template: str, production_template: str) -> str:
    """`new_template` with production's lessons block, whatever the writer did to its copy."""
    return join(split(new_template)[0], production_template)


# --------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------
def match(request: str, lessons: list[dict], k: int = 3) -> list[dict]:
    """The lessons whose keywords overlap the request most, best first; none without overlap."""
    want = set(tokens(request))
    scored = []
    for les in lessons:
        have = set(les.get("keys") or []) | set(tokens(les["lesson"])) | set(tokens(les.get("theme", "")))
        overlap = len(want & have)
        if overlap:
            scored.append((overlap, int(les.get("evidence") or 0), les))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [les for _, _, les in scored[:k]]
