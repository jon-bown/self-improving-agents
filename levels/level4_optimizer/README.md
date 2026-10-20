# Level 4 · Optimizer

Fix the verifier first: align the judge to the reviewers. Then let GEPA search for a better `AGENTS.md`, scored by that judge. The holdout set in the gate catches overfitting.

![Level 4](../../docs/architecture/svg/level4-optimizer.svg)

| Script | Does |
|---|---|
| `01_align_judge.py` | `judge.align()` (SIMBA, or MemAlign when embeddings are available); registers the aligned judge only if agreement improves |
| `02_optimize.py` | `optimize_prompts()` + `GepaPromptOptimizer`; run with `SIA_BACKEND=direct` |
| `03_gate.py` | diff + gate; `--promote` to ship |

<details><summary>Mermaid source</summary>

<!-- diagram:level4-optimizer -->
```mermaid
---
title: "Level 4 · Optimizer: fix the verifier first, then let GEPA write the candidate"
---
flowchart TB
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
    HL[("Human labels")]:::mlflow -- "01 · judge.align()<br/>SIMBA · MemAlign" --> AQ["Aligned quality judge<br/>registered only if agreement improves"]:::signal_new
    DS[("Dataset rows")]:::mlflow -- "02 · optimize_prompts()" --> GEPA["GEPA proposes AGENTS.md variants<br/>(SIA_BACKEND=direct)"]:::update_new
    GEPA --> RUN["Agent answers each row"]:::runtime
    RUN --> SC["Scorers: code · guidelines · themes<br/>+ aligned quality judge"]:::signal
    AQ --> SC
    SC -- "scores + rationales → reflection" --> GEPA
    GEPA -- "best candidate · holdout guards against overfitting" --> G{"03 · Gate"}:::gate
    G -- "pass" --> P(["Person reads the diff and promotes"]):::human
```
<!-- /diagram -->
</details>
