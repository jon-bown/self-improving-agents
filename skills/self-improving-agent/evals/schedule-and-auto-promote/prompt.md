---
tags: [skill-creator]
max_turns: 20
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

we've run the loop by hand a few times on our support bot (analyst writes candidates, i promote). now i want it running nightly in github actions against our shared mlflow server, and eventually promoting by itself. what do i need to set up, and how do i know when it's safe to turn on auto promote?
