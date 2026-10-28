"""`selfimprove self recall | propose | lessons | sync | setup`, offline.

The agent can read what was learned and propose changes to itself, and every route to a change
ends at the gate or a person: lessons are a candidate prompt version, an AGENTS.md proposal is a
candidate the agent can't promote, a skill proposal waits in the queue, and the deployed agent's
exec allowlist reaches only the four agent-facing commands.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_agentside import cli  # noqa: F401 -- the same isolated CLI fixture, a fresh store per module

TRACES = """
import json
from selfimprove import traces
traces.setup()
print(json.dumps({traces.question_of(t): t.info.trace_id for t in traces.production()}))
"""

MERGE = """
import json, sys
from selfimprove import store, traces
traces.setup()
recs = []
for tid in sys.argv[2:]:
    t = traces.search(f"trace.request_id = '{tid}'")[0]
    rec = store.record(traces.question_of(t), traces.expectations(t), source="human_fail", author="triage-agent",
                       split=sys.argv[1], source_trace=tid, theme="refund_window",
                       reason="quality: quoted the old 30-day refund window")
    recs.append(rec)
print(store.merge(recs))
"""

TEMPLATE = """
import json, sys
from selfimprove import agents, instructions, lessons, traces
traces.setup()
t = instructions.load(sys.argv[1]).template
ws = agents.workspace_for(t)
print(json.dumps({"block": lessons.block(t) is not None, "parsed": lessons.parse(t),
                  "agents_md": (ws / "AGENTS.md").read_text() if (ws / "AGENTS.md").is_file()
                               else (ws / "instructions.md").read_text(),
                  "lessons_md": (ws / "LESSONS.md").read_text() if (ws / "LESSONS.md").is_file() else None}))
"""

REFUND = ["How long do I have to request a refund?", "Can I get a refund after 20 days?",
          "What is the refund window for annual plans?"]
HELD_OUT = "How many days for a refund?"   # store.split_for puts this one in the holdout


@pytest.fixture(scope="module")
def world(cli):  # noqa: F811
    cli("init")
    for q in [*REFUND, HELD_OUT, "Is shipping free?"]:
        cli("ask", q)
    ids = cli.python(TRACES)
    for q in [*REFUND, HELD_OUT]:
        cli("label", ids[q], "fail", "--why", "quoted 30 days; policy is 60",
            "--expected", "Refunds can be requested within 60 days.")
    cli.python(MERGE, "train", ids[REFUND[0]], ids[REFUND[1]])
    return ids


def test_recall_before_any_lessons(cli, world):  # noqa: F811
    out = cli.self_("recall", "How long do refunds take?")
    assert out == {"ok": True, "source": "lessons@1", "lessons": []}


def test_lessons_are_a_candidate_the_gate_decides(cli, world):  # noqa: F811
    out = cli("self", "lessons", "--no-llm")
    assert "[lessons] candidate: v2" in out
    v2 = cli.python(TEMPLATE, "2")
    assert v2["block"] and len(v2["parsed"]) == 1
    les = v2["parsed"][0]
    assert les["theme"] == "refund_window" and les["evidence"] == 2
    assert set(les["traces"]) == {world[REFUND[0]], world[REFUND[1]]}
    # The workspace the gate runs in: LESSONS.md beside AGENTS.md, a pointer in its place.
    assert v2["lessons_md"] and "## Lessons" in v2["lessons_md"] and "si:" not in v2["lessons_md"]
    assert "LESSONS.md" in v2["agents_md"] and "si:lessons" not in v2["agents_md"]

    # Not live until it's promoted: recall reads production.
    assert cli.self_("recall", "How long do refunds take?")["lessons"] == []
    assert cli.self_("status")["lessons"] == 0
    cli("promote", "2")
    hit = cli.self_("recall", "How long do refunds take?")
    assert hit["source"] == "lessons@2" and hit["lessons"][0]["evidence"] == 2
    assert hit["lessons"][0]["confirmed_by"] == "human"
    assert cli.self_("recall", "What's your phone number?")["lessons"] == []
    assert cli.self_("status")["lessons"] == 1
    assert "no_change" in cli("self", "lessons", "--no-llm")


def test_propose_refusals(cli, world):  # noqa: F811
    two = f"{world[REFUND[0]]},{world[REFUND[1]]}"
    text = "# Acme support\n\nRead knowledge/ before quoting any policy.\n"

    def propose(evidence: str, body: str = text, target: str = "AGENTS.md") -> dict:
        return cli.self_("propose", "--target", target, "--text", body, "--why", "refund window", "--evidence", evidence)

    assert "at least 2" in propose(world[REFUND[0]])["refused"]
    assert "no confirmed failure" in propose(f"{world[REFUND[0]]},{world['Is shipping free?']}")["refused"]
    assert "held-out" in propose(f"{world[REFUND[0]]},{world[HELD_OUT]}")["refused"]
    assert "too long" in propose(two, text * 400)["refused"]
    assert "lessons are rebuilt" in propose(two, text + "<!-- si:lessons begin -->\n<!-- si:lessons end -->\n")["refused"]
    assert "target is" in propose(two, target="SOUL.md")["refused"]


def test_propose_agents_md_waits_for_the_gate(cli, world):  # noqa: F811
    two = f"{world[REFUND[0]]},{world[REFUND[1]]}"
    seen = cli.python(TEMPLATE, "production")["agents_md"]   # what the agent sees: the pointer, no block
    edited = seen + "\nQuote refund windows from knowledge/ only.\n"
    out = cli.self_("propose", "--target", "AGENTS.md", "--text", edited, "--why", "two refund failures",
                    "--evidence", two)
    assert out["ok"] is True, out
    v = out["candidate_version"]
    cand = cli.python(TEMPLATE, str(v))
    assert cand["parsed"] == cli.python(TEMPLATE, "production")["parsed"]   # production's lessons, kept
    status = cli.self_("status")
    assert status["prompt"]["production"] == 2
    assert {"version": v, "author": "openclaw-self"} in status["open_candidates"]
    again = cli.self_("propose", "--target", "AGENTS.md", "--text", edited + "x\n", "--why", "again",
                      "--evidence", two)
    assert "open proposal exists" in again["refused"]


def test_skill_proposal_goes_to_a_person(cli, world, tmp_path):  # noqa: F811
    ws = tmp_path / "live"
    skill = ws / "skills" / "refunds" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: refunds\n---\nAnswer refund questions.\n")
    proposed = "---\nname: refunds\n---\nAnswer refund questions from knowledge/billing.md.\n"
    two = f"{world[REFUND[0]]},{world[REFUND[1]]}"
    args = ("propose", "--target", "skill:refunds", "--text", proposed, "--why", "cite the policy", "--evidence", two)
    out = cli.self_(*args, SI_SELF_WORKSPACE=str(ws))
    assert out["ok"] is True and out["target"] == "skill:refunds", out
    assert skill.read_text() != proposed                       # nothing applied yet
    assert "open proposal exists" in cli.self_(*args, SI_SELF_WORKSPACE=str(ws))["refused"]
    accepted = json.loads(cli("queue", "accept", out["queue_item"], SI_SELF_WORKSPACE=str(ws)))
    assert accepted["kind"] == "skill_proposal" and accepted["applied_to"] == str(skill)
    assert skill.read_text() == proposed


def test_sync_writes_the_live_workspace(cli, world, tmp_path):  # noqa: F811
    ws = tmp_path / "live"
    assert "[sync] synced" in cli("self", "sync", "--workspace", str(ws))
    assert "LESSONS.md" in (ws / "AGENTS.md").read_text() and "si:lessons" not in (ws / "AGENTS.md").read_text()
    assert "## Lessons" in (ws / "LESSONS.md").read_text()
    assert "[sync] current" in cli("self", "sync", "--workspace", str(ws))
    assert cli.self_("status")["synced"]["version"] == 2


def test_setup_allowlists_only_the_agent_commands(cli, world, tmp_path):  # noqa: F811
    ws, prof = tmp_path / "ws", tmp_path / "profile"
    out = cli("self", "setup", "--workspace", str(ws), "--profile", str(prof), OPENCLAW_BIN="openclaw-not-here")
    assert "[setup] ready" in out and "not on PATH" in out
    cfg = json.loads((prof / "openclaw.json").read_text())
    assert cfg["tools"]["allow"] == ["read", "exec"] and cfg["tools"]["exec"]["mode"] == "allowlist"
    assert cfg["skills"]["workshop"]["autonomous"]["mode"] == "off"
    assert cfg["gateway"]["port"] == 19556 and cfg["agents"]["defaults"]["workspace"] == str(ws)
    assert (ws / "skills" / "self-improving-with-mlflow" / "SKILL.md").is_file()
    assert not (ws / "skills" / "self-improving-with-mlflow" / "evals").exists()
    assert (ws / "AGENTS.md").is_file()

    wrapper = prof / "bin" / "selfimprove"
    for blocked in (["self", "gate"], ["self", "lessons"], ["promote", "3"], ["queue", "accept", "x"]):
        res = json.loads(subprocess.run([str(wrapper), *blocked], capture_output=True, text=True).stdout)
        assert res["ok"] is False and "not available" in res["refused"]
    res = subprocess.run([str(wrapper), "self", "status"], capture_output=True, text=True, timeout=300)
    assert json.loads(res.stdout.strip().splitlines()[-1])["prompt"]["production"] == 2, res.stderr[-2000:]


def test_recall_drops_lessons_whose_evidence_moved_to_the_holdout(cli, world):  # noqa: F811
    assert cli.self_("recall", "How long do refunds take?")["lessons"]
    cli.python(MERGE, "holdout", world[REFUND[0]])
    assert cli.self_("recall", "How long do refunds take?")["lessons"] == []
