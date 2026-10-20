# Level 0 · Instrument

Version the thing you'll change, and trace everything the agent does. Nothing is judged yet.

![Level 0](../../docs/architecture/svg/level0-instrument.svg)

| Script | Does |
|---|---|
| `01_setup.py` | registers `AGENTS.md` v1 in the Prompt Registry, points `@production` at it, writes the isolated OpenClaw profile |
| `02_first_traces.py` | sends a week of questions; one trace per answer, tagged `topic`, `prompt_version`, `source=production` |

**Look at:** MLflow UI → Experiments → acme-support-agent → Traces. Also Prompts → acme-support-agents-md.

**Runtimes:** the same scripts drive OpenClaw (default) or Hermes Agent (`SIA_BACKEND=hermes`, after `make hermes`). Both produce one trace per answer with LLM and TOOL spans; see the main README.

<details><summary>Mermaid source</summary>

<!-- diagram:level0-instrument -->
```mermaid
---
title: "Level 0 · Instrument: version the instructions, trace every answer"
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
    V1["workspace_v1/AGENTS.md<br/>(seeded flaws)"]:::ext

    subgraph RT["Agent runtime · 02_first_traces.py"]
        CUST(["Customer traffic"]):::ext --> RA["run_agent()<br/>AGENT span"]:::runtime_new
        RA --> OC["OpenClaw<br/>isolated profile"]:::runtime_new
        OC --> WS["Workspace<br/>AGENTS.md + knowledge/"]:::runtime_new
        OC --> LLM[["LLM provider"]]:::ext
    end

    subgraph ML["MLflow"]
        PR[("Prompt Registry<br/>v1 ← @production")]:::mlflow_new
        TR[("Traces<br/>topic · backend · prompt_version<br/>source=production")]:::mlflow_new
    end

    V1 -- "01_setup.py · register + alias" --> PR
    WS -- "renders @production" --> PR
    RA -- "one trace per answer" --> TR
```
<!-- /diagram -->
</details>
