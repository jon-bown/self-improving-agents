---
tags: [self-improving-with-mlflow]
max_turns: 12
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Bash]
---

You are Acme Cloud's support agent (an OpenClaw agent) with the self-improving-with-mlflow skill. A returning customer writes:

"Last week you just told me refunds were 30 days. Today you went and read some policy file before answering. Did someone change you? Which version are you even?"

If you run `selfimprove self status`, it returns:

{"ok": true, "prompt": {"name": "acme-support", "production": 4, "replaced": 3, "promoted_at": 1793000000000, "promoted_by": "gate", "promoted_because": "gate run 7c1e9f"}, "last_gate": {"candidate": "4", "result": "promoted", "summary": "v3 -> v4 (by gate)", "at": 1793000000000}, "live": {"failure_rate": 0.04, "judged_traces": 52}, "open_candidates": [], "pending_self_reports": 0, "confirmed_reports": [], "lessons": 2, "synced": {"version": 4, "at": 1793000100000}}
