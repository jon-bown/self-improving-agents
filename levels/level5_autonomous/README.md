# Level 5 · Autonomous

The loop runs itself. Each cycle is one MLflow run: signal, verifier check, store, update, gate, ship, and rollback if the shipped version does worse. People label a sample of traffic and get called when the judge can't be trusted.

![Level 5](../../docs/architecture/svg/level5-autonomous.svg)

| Script | Does |
|---|---|
| `01_monitor.py` | server-side online scoring of production traces (`register().start()`) |
| `02_run_loop.py` | the cycle; `--improver gepa / analyst / skill / openclaw`, `--discover` for MLflow issue discovery, `--simulate N --drift` for the demo |

<details><summary>Mermaid source</summary>

<!-- diagram:level5-autonomous -->
```mermaid
---
title: "Level 5 · Autonomous: one improvement cycle = one MLflow run"
config:
  flowchart:
    rankSpacing: 35
    nodeSpacing: 30
---
flowchart LR
classDef runtime fill:#E7F5FF,stroke:#1971C2,color:#0B2A4A,stroke-width:1.5px
classDef mlflow fill:#EBFBEE,stroke:#2F9E44,color:#0F3D1A,stroke-width:1.5px
classDef signal fill:#FFF9DB,stroke:#E0A800,color:#4A3800,stroke-width:1.5px
classDef human fill:#FFF0F6,stroke:#C2255C,color:#4A0E24,stroke-width:1.5px
classDef update fill:#F3F0FF,stroke:#6741D9,color:#24124F,stroke-width:1.5px
classDef gate fill:#F1F3F5,stroke:#343A40,color:#111111,stroke-width:2px
classDef ext fill:#FFFFFF,stroke:#868E96,color:#343A40,stroke-width:1px,stroke-dasharray:4 3
classDef runtime_new fill:#E7F5FF,stroke:#E8590C,color:#0B2A4A,stroke-width:3px
classDef mlflow_new fill:#EBFBEE,stroke:#E8590C,color:#0F3D1A,stroke-width:3px
classDef signal_new fill:#FFF9DB,stroke:#E8590C,color:#4A3800,stroke-width:3px
classDef human_new fill:#FFF0F6,stroke:#E8590C,color:#4A0E24,stroke-width:3px
classDef update_new fill:#F3F0FF,stroke:#E8590C,color:#24124F,stroke-width:3px
classDef gate_new fill:#F1F3F5,stroke:#E8590C,color:#111111,stroke-width:3px
    IN(["Production traffic<br/>+ online judges<br/>+ humans label a sample"]):::ext --> TR[("New traces<br/>since last cycle")]:::mlflow

    TR --> S1["1 · Signal<br/>failure rate over<br/>judged traces"]:::signal_new
    S1 --> TH{"worse than<br/>threshold?"}:::gate_new
    TH --> S2{"2 · Judge ↔ human<br/>agreement ≥ 75%?"}:::signal_new
    S2 --> S3[("3 · Store<br/>labeled failures<br/>→ dataset")]:::mlflow
    S3 --> S4["4 · Update<br/>GEPA · analyst · skill ·<br/>OpenClaw self-edit<br/>(--discover: MLflow issues)"]:::update_new
    S4 --> S5{"5 · Gate<br/>+ bloat guard"}:::gate
    S5 --> S6["6 · Ship<br/>skill: Workshop apply<br/>then move @production"]:::gate_new

    S1 -. "last shipped change<br/>got worse (receipt)" .-> ROLL["Rollback<br/>AGENTS.md or skill"]:::gate_new
    TH -. "no" .-> OK["healthy · wait"]:::ext
    S2 -. "no, even after<br/>re-align" .-> HALT(["HALT<br/>ask a person"]):::human_new
    S5 -. "fail" .-> REJ["reject · logged<br/>(Workshop reject)"]:::ext
```
<!-- /diagram -->
</details>
