# Level 3 · Agent-assisted

MLflow finds the issues and an LLM analyst writes one change for a person to review. The change is either an
`AGENTS.md` edit or a **new OpenClaw skill** filed through the Skill Workshop. Accepted criteria become
registered judges, so the verifier set grows with the evidence.

![Level 3](../../docs/architecture/svg/level3-agent-assisted.svg)

| Script | Does |
|---|---|
| `01_propose.py` | MLflow `discover_issues` → LLM analyst → `proposals/<ts>/` (`--discovery analyst` skips MLflow; `--target agents_md / skill` forces the update target) |
| `02_accept.py <dir>` | registers the candidate (AGENTS.md version, or Workshop proposal + MLflow skill version) and the `theme_<slug>` judges, grows the dataset, runs the gate. `--promote` ships: for a skill, Workshop `apply` first, then the `@production` alias. A failing skill is Workshop-`reject`ed with the gate's reasons |

**Look at:** the issue-discovery run in MLflow and the issue assessments on the affected traces. In the isolated
OpenClaw profile, `openclaw skills workshop list` shows the proposal's status (pending, applied, or rejected).

<details><summary>Mermaid source</summary>

<!-- diagram:level3-agent-assisted -->
```mermaid
---
title: "Level 3 · Agent-assisted: MLflow finds the issues, an analyst proposes, a person approves"
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
    TR[("Failing traces<br/>verdicts + tool calls")]:::mlflow
    DI["MLflow issue discovery<br/>discover_issues()<br/>named issues · root causes"]:::signal_new
    AN["LLM analyst<br/>criteria + ONE change<br/>picks the update target"]:::update_new
    PROP["proposals/&lt;ts&gt;/<br/>reviewed by a person"]:::human

    subgraph ACCEPT["02_accept.py"]
        AMD["AGENTS.md<br/>new prompt version"]:::update
        SK["Skill<br/>OpenClaw Skill Workshop proposal<br/>(pending · scanned)<br/>+ MLflow skill version"]:::update_new
        TJ[("theme_&lt;slug&gt; judges<br/>+ dataset rows")]:::mlflow
    end

    G{"Gate<br/>production vs<br/>production + change"}:::gate
    APPLY["Workshop apply<br/>→ move @production"]:::gate_new
    REJ["Workshop reject<br/>with gate reasons"]:::ext

    TR --> DI --> AN --> PROP
    PROP -- "AGENTS.md" --> AMD
    PROP -- "skill" --> SK
    PROP --> TJ
    AMD & SK & TJ --> G
    G -- "pass + person" --> APPLY
    G -- "fail" --> REJ
```
<!-- /diagram -->
</details>
