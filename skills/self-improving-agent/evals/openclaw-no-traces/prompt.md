---
tags: [skill-creator]
max_turns: 20
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

I installed the mlflow openclaw plugin into my openclaw (2026.10.5) and pointed it at my local mlflow with MLFLOW_TRACKING_URI=sqlite:///mlflow.db but nothing shows up in the UI after running openclaw agent --local. once it works I want my openclaw agent to go through the self improvement harness in this repo, with tool calls visible so I can check it actually reads the docs. how do I set that up?
