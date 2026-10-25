---
tags: [skill-creator]
max_turns: 20
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

my selfimprove gate just rejected candidate v7 with "rejected bloat: candidate is 2.2x the length of production". metaprompt wrote it after our support team shipped an AGENTS.md edit that made the agent ask customers for their 2fa codes. the diff adds a whole security section (never ask for passwords/codes etc). GEPA's candidate v9 also got rejected, must-pass no_credential_requests at 92%. what do I do now? should I just bump the limit?
