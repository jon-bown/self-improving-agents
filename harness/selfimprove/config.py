"""Settings, read once from the environment and an optional .env file (see .env.example).

The .env file is looked up at $SI_ENV_FILE, then ./.env. Relative paths in it (instructions,
criteria, golden cases, workspace, the SQLite store) are resolved against the folder that holds
the .env file, so the harness behaves the same from a notebook, the CLI, or CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# MLflow usage telemetry can deadlock on a lazy import when evaluate() runs scorers in threads.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _find_env() -> Path | None:
    explicit = os.getenv("SI_ENV_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path.cwd()
    for folder in (here, *here.parents):
        if (folder / ".env").is_file() and (folder / "selfimprove").is_dir():
            return folder / ".env"
    return (here / ".env") if (here / ".env").is_file() else None


ENV_FILE = _find_env()
ROOT = ENV_FILE.parent if ENV_FILE else Path.cwd()
if ENV_FILE:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _path(name: str, default: str = "") -> Path | None:
    value = _get(name, default)
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _float(name: str, default: float) -> float:
    return float(_get(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_get(name, str(default)))


def _tracking_uri() -> str:
    uri = _get("MLFLOW_TRACKING_URI", "sqlite:///.selfimprove/mlflow.db")
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        # sqlite:///relative.db -> absolute, so every process uses the same file
        db = ROOT / uri.removeprefix("sqlite:///")
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    return uri


@dataclass(frozen=True)
class Settings:
    # MLflow
    tracking_uri: str = field(default_factory=_tracking_uri)
    experiment: str = field(default_factory=lambda: _get("SI_EXPERIMENT", "my-agent"))

    # The agent
    agent: str = field(default_factory=lambda: _get("SI_AGENT", "python").lower())
    agent_target: str = field(default_factory=lambda: _get("SI_AGENT_TARGET"))
    agent_url: str = field(default_factory=lambda: _get("SI_AGENT_URL"))
    agent_command: str = field(default_factory=lambda: _get("SI_AGENT_COMMAND"))
    agent_model: str = field(default_factory=lambda: _get("SI_AGENT_MODEL", "anthropic:/claude-haiku-5-5"))
    agent_timeout: float = field(default_factory=lambda: _float("SI_AGENT_TIMEOUT", 600))
    agent_name: str = field(default_factory=lambda: _get("SI_AGENT_NAME", "my_agent"))
    input_key: str = field(default_factory=lambda: _get("SI_INPUT_KEY", "question"))
    openclaw_bin: str = field(default_factory=lambda: _get("OPENCLAW_BIN", "openclaw"))
    openclaw_mlflow: bool = field(default_factory=lambda: _get("SI_OPENCLAW_MLFLOW") == "1")

    # What improves
    prompt_name: str = field(default_factory=lambda: _get("SI_PROMPT_NAME", "my-agent-instructions"))
    instructions_file: Path | None = field(default_factory=lambda: _path("SI_INSTRUCTIONS_FILE"))
    workspace_dir: Path | None = field(default_factory=lambda: _path("SI_WORKSPACE_DIR"))
    production_filter: str = field(default_factory=lambda: _get("SI_PRODUCTION_FILTER"))
    dataset_name: str = field(default_factory=lambda: _get("SI_DATASET_NAME") or
                              f"{_get('SI_EXPERIMENT', 'my-agent')}-regressions")

    # Models
    judge_model: str = field(default_factory=lambda: _get("SI_JUDGE_MODEL") or
                             _get("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "anthropic:/claude-opus-5-5"))
    embedding_model: str = field(default_factory=lambda: _get("SI_EMBEDDING_MODEL"))
    # A cheap decision model as the first-pass judge (Part 3, section 3), e.g. typesafe:/jev-latest.
    # Its verdicts with a pass probability inside the band are re-judged by judge_model.
    fast_judge_model: str = field(default_factory=lambda: _get("SI_FAST_JUDGE_MODEL"))
    fast_judge_band: tuple[float, float] = field(default_factory=lambda: tuple(
        float(x) for x in _get("SI_FAST_JUDGE_BAND", "0.5,0.8").split(",")))
    no_llm: bool = field(default_factory=lambda: _get("SI_NO_LLM") == "1")

    # Verifier
    criteria_file: Path | None = field(default_factory=lambda: _path("SI_CRITERIA_FILE"))
    golden_file: Path | None = field(default_factory=lambda: _path("SI_GOLDEN_FILE"))

    # Update step
    improver: str = field(default_factory=lambda: _get("SI_IMPROVER", "analyst").lower())
    improver_command: str = field(default_factory=lambda: _get("SI_IMPROVER_COMMAND"))

    # Guardrails
    triage_threshold: int = field(default_factory=lambda: _int("SI_TRIAGE_THRESHOLD", 5))
    tests_per_theme_cap: int = field(default_factory=lambda: _int("SI_TESTS_PER_THEME_CAP", 3))
    holdout_fraction: float = field(default_factory=lambda: _float("SI_HOLDOUT_FRACTION", 0.2))
    agreement_threshold: float = field(default_factory=lambda: _float("SI_AGREEMENT_THRESHOLD", 0.75))
    min_alignment_traces: int = field(default_factory=lambda: _int("SI_MIN_ALIGNMENT_TRACES", 10))
    promotion_margin: float = field(default_factory=lambda: _float("SI_PROMOTION_MARGIN", 0.0))
    max_prompt_growth: float = field(default_factory=lambda: _float("SI_MAX_PROMPT_GROWTH", 2.0))
    rollback_margin: float = field(default_factory=lambda: _float("SI_ROLLBACK_MARGIN", 0.10))
    fail_threshold: float = field(default_factory=lambda: _float("SI_FAIL_THRESHOLD", 0.15))
    min_traces: int = field(default_factory=lambda: _int("SI_MIN_TRACES", 8))
    auto_promote: bool = field(default_factory=lambda: _get("SI_AUTO_PROMOTE", "false").lower() == "true")
    # `selfimprove self note`: reports per conversation per day, and how far back "previous" looks
    self_note_max_per_session: int = field(default_factory=lambda: _int("SI_SELF_NOTE_MAX_PER_SESSION", 5))
    self_note_window_min: float = field(default_factory=lambda: _float("SI_SELF_NOTE_WINDOW_MIN", 30))
    # `selfimprove self propose`: confirmed failures a proposal must cite, and whether it gates at once
    self_propose_min_evidence: int = field(default_factory=lambda: _int("SI_SELF_PROPOSE_MIN_EVIDENCE", 2))
    self_propose_gate: bool = field(default_factory=lambda: _get("SI_SELF_PROPOSE_GATE") == "1")
    # `selfimprove self lessons`: at most this many lessons (one per theme) in the pack
    lessons_max: int = field(default_factory=lambda: _int("SI_LESSONS_MAX", 8))
    # Deployed mode: the live OpenClaw workspace `self sync` writes, and its isolated profile
    self_workspace: Path | None = field(default_factory=lambda: _path("SI_SELF_WORKSPACE"))

    # Local state: checkpoints, the human queue, improver workspaces
    state_dir: Path = field(default_factory=lambda: _path("SI_STATE_DIR", ".selfimprove"))

    @property
    def has_llm(self) -> bool:
        from selfimprove import models

        return not self.no_llm and models.has_credentials(self.judge_model)

    @property
    def has_fast_judge(self) -> bool:
        from selfimprove import models

        return bool(self.fast_judge_model) and not self.no_llm and models.has_credentials(self.fast_judge_model)

    def describe(self) -> str:
        target = {"python": self.agent_target, "http": self.agent_url, "command": self.agent_command,
                  "litellm": self.agent_model, "openclaw": self.agent_model}.get(self.agent, "")
        return "\n".join([
            f"env file      {ENV_FILE or '(none)'}",
            f"tracking      {self.tracking_uri}   experiment {self.experiment}",
            f"agent         {self.agent}: {target}",
            f"instructions  prompts:/{self.prompt_name}  (seed {self.instructions_file})",
            f"judge         {self.judge_model}  {'(credentials found)' if self.has_llm else '(no LLM: code checks only)'}",
            f"first pass    {self.fast_judge_model or '-'}" + (
                f"  (re-judge p(pass) in {self.fast_judge_band[0]:.2f}-{self.fast_judge_band[1]:.2f})"
                if self.has_fast_judge else ("  (no credentials: off)" if self.fast_judge_model else "")),
            f"improver      {self.improver}   auto-promote {self.auto_promote}",
        ])


settings = Settings()
os.environ.setdefault("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", settings.judge_model)


def reload() -> Settings:
    """Re-read the environment and the .env file (notebooks change settings between cells)."""
    global settings
    if ENV_FILE:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
    settings = Settings()
    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = settings.judge_model
    return settings
