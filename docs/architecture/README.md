# Architecture diagrams

One diagram per level, plus three overview diagrams. The diagrams share one visual language, so they work as a
slide build-up:

| Color | Role |
|---|---|
| blue | agent runtime (harness, OpenClaw, workspace) |
| green | stored in MLflow (Prompt Registry, traces, datasets, scorer registry, runs) |
| yellow | automated signal (code scorers, LLM judges) |
| pink | people |
| purple | produces a candidate change |
| grey | the gate: decides and promotes |
| **orange outline** | **introduced at this level** |

| File | Use |
|---|---|
| `src/*.mmd` | the source; edit these |
| `mmd/*.mmd` | standalone Mermaid with styles inlined. Paste into [mermaid.live](https://mermaid.live), Notion, Confluence, or Slidev |
| `svg/*.svg` | vector, for slides and the web |
| `png/*.png` | 2x raster, for tools that want images |

After editing a source file, rebuild everything and refresh the copies embedded in the READMEs:

```bash
make diagrams
```

`make diagrams` needs Node.js and uses a local Chrome if one is installed. `python scripts/render_diagrams.py --no-images`
refreshes only the `.mmd` files and the markdown.

## System

<!-- diagram:00-system -->
```mermaid
---
title: "System architecture: OpenClaw agent + MLflow"
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
    subgraph SERVE["Serving · src/sia/agent.py"]
        CUST(["Customer"]):::ext --> RA["run_agent(AgentSpec)<br/>AGENT span + tags"]:::runtime
        RA --> RT["OpenClaw · Hermes Agent<br/>(or direct · offline)"]:::runtime
        RT --> WS["Workspace<br/>AGENTS.md · skills/ · knowledge/"]:::runtime
        RT --> LLM[["LLM provider<br/>provider:/model"]]:::ext
        RT -- "trajectory · callbacks" --> TJ["LLM + TOOL spans"]:::runtime
    end

    subgraph IMP["Improvement · levels/"]
        SIG["Signal<br/>reviewers · scorers · judges<br/>tool-level checks · issue discovery"]:::signal --> UPD["Update<br/>person · analyst · GEPA · OpenClaw"]:::update
        UPD --> GATE{"Gate"}:::gate
        UPD -- "skills" --> SW["OpenClaw<br/>Skill Workshop"]:::runtime
    end

    subgraph ML["MLflow tracking server"]
        PR[("Prompt Registry<br/>AGENTS.md + skill versions · @production")]:::mlflow
        TR[("Traces + assessments + issues")]:::mlflow
        SR[("Scorer registry")]:::mlflow
        DS[("Evaluation dataset")]:::mlflow
        RN[("Runs · evals · gates")]:::mlflow
    end

    WS -- "renders @production" --> PR
    RA -- "logs" --> TR
    TJ -- "spans" --> TR
    SIG -- "reads traces · writes assessments" --> TR
    SIG -- "uses judges" --> SR
    SIG -- "curates" --> DS
    UPD -- "registers new version" --> PR
    GATE -- "evaluates on" --> DS
    GATE -- "logs" --> RN
    GATE -- "moves alias" --> PR
```
<!-- /diagram -->

## The loop every level runs

<!-- diagram:01-loop -->
```mermaid
---
title: "The self-improvement loop (every level)"
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
    T[("Traces")]:::mlflow --> S["SIGNAL<br/>human feedback · expectations<br/>code scorers · LLM judges"]:::signal
    S --> ST[("STORE<br/>evaluation dataset<br/>grows from failures")]:::mlflow
    ST --> U["UPDATE<br/>new AGENTS.md version<br/>in the Prompt Registry"]:::update
    U --> G{"GATE<br/>must-pass · beats production<br/>no holdout regression"}:::gate
    G -- "pass" --> P["PROMOTE<br/>move @production alias"]:::gate
    G -- "fail" --> X["reject<br/>(kept as a version)"]:::ext
    P --> A["Agent serves<br/>new version"]:::runtime --> T
```
<!-- /diagram -->

## The levels

<!-- diagram:02-levels-overview -->
```mermaid
---
title: "Levels of self-improvement: who writes the change, who approves it"
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
    L0["<b>Level 0 · Instrument</b><br/>version AGENTS.md<br/>trace every answer"]:::runtime
    L1["<b>Level 1 · Signal</b><br/>reviewers · code scorers · judges<br/>check the verifiers"]:::signal
    L2["<b>Level 2 · Manual loop</b><br/>writes: person<br/>approves: person"]:::human
    L3["<b>Level 3 · Agent-assisted</b><br/>writes: LLM analyst<br/>approves: person"]:::update
    L4["<b>Level 4 · Optimizer</b><br/>writes: GEPA + aligned judge<br/>approves: person"]:::update
    L5["<b>Level 5 · Autonomous</b><br/>writes: GEPA · analyst · OpenClaw itself<br/>approves: the gate"]:::gate
    L0 --> L1 --> L2 --> L3 --> L4 --> L5
```
<!-- /diagram -->

## Level 0 · Instrument
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

## Level 1 · Signal
<!-- diagram:level1-signal -->
```mermaid
---
title: "Level 1 · Signal: people, code, and judges label the traces"
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
    TR[("Traces (production)<br/>incl. OpenClaw's own LLM + TOOL spans")]:::mlflow

    subgraph PEOPLE["01 · people"]
        RQ["Review queue<br/>label schemas (--live)"]:::human_new --> RV["Reviewers<br/>maria · dev · sam<br/>different rubrics"]:::human_new
    end

    subgraph AUTO["02 · evaluate(data=traces) · no predict_fn"]
        CS["Code scorers<br/>stale policy · disclaimer · length"]:::signal_new
        TS["Tool-level scorers<br/>read_right_policy · focused_reading"]:::signal_new
        GJ["Guidelines judges<br/>leads_with_answer · no_guessing"]:::signal_new
        QJ["quality judge<br/>make_judge → bool"]:::signal_new
    end

    AS[("Assessments per trace<br/>feedback: HUMAN · CODE · LLM_JUDGE<br/>expectations: answer + key facts")]:::mlflow_new
    AGR{"03 · Check the verifiers<br/>judge ↔ human agreement<br/>&lt; 85% → don't optimize on it"}:::signal_new

    TR --> RQ
    TR --> CS & TS & GJ & QJ
    RV -- "log_feedback<br/>log_expectation" --> AS
    CS & TS & GJ & QJ --> AS
    AS --> AGR
```
<!-- /diagram -->

## Level 2 · Manual loop
<!-- diagram:level2-manual-loop -->
```mermaid
---
title: "Level 2 · Manual loop: a person writes the fix and approves it"
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
    AS[("Labeled traces<br/>feedback.quality = 'false'<br/>+ expected_response")]:::mlflow
    DS[("Evaluation dataset<br/>dedupe by question")]:::mlflow_new
    HO["Holdout cases<br/>(never optimized on)"]:::ext

    P1(["Person reads<br/>rationales"]):::human_new
    V2["AGENTS.md v2<br/>register_prompt<br/>(not promoted)"]:::update_new

    subgraph GATE["Gate · sia/gate.py"]
        EV["evaluate() production vs v2<br/>same data · same scorers"]:::gate_new
        RULES{"must-pass 100%<br/>beats production<br/>holdout not worse"}:::gate_new
    end

    P2(["Person approves<br/>--promote"]):::human_new
    PR[("Prompt Registry<br/>@production → v2<br/>@previous → v1")]:::mlflow
    CI["pytest -p mlflow.pytest.plugin<br/>@mlflow.test per dataset row<br/>(CI on every PR)"]:::gate_new

    AS -- "merge_records" --> DS
    AS --> P1 --> V2
    DS & HO --> EV
    V2 --> EV --> RULES
    RULES -- "pass" --> P2 --> PR
    DS --> CI
```
<!-- /diagram -->

## Level 3 · Agent-assisted
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

## Level 4 · Optimizer
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

## Level 5 · Autonomous
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
