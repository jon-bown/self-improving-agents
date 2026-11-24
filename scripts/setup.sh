#!/usr/bin/env bash
# One-step setup for the whole repo. Safe to re-run; each step skips work that's already done.
#
#   scripts/setup.sh          (or: make setup)   install everything, then run the checks
#   scripts/setup.sh check    (or: make check)   only the checks
#
# What it sets up, all inside this checkout:
#   .venv/                    one Python environment for harness/, sia/ and blog/ (uv workspace)
#   Jupyter kernel            "Python (self-improving-agents)", registered inside .venv
#   .venv/bin/openclaw        OpenClaw, installed into the venv with npm (no global install)
#   connectors/openclaw-mlflow/*.tgz   the MLflow tracing plugin for OpenClaw, built
#   harness/.env, sia/.env    copied from .env.example if missing
#
# Environment overrides:
#   UV_PYTHON=3.12            use a specific Python (any version >= 3.10 works). By default uv
#                             picks one, native to the machine (arm64 on Apple Silicon).
#   OPENCLAW_VERSION=2026.10.5  install this OpenClaw instead of the latest
#   SKIP_OPENCLAW=1           skip OpenClaw and the connector (Python + notebooks only)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
CONNECTOR="$ROOT/connectors/openclaw-mlflow"
KERNEL_NAME="self-improving-agents"
KERNEL_DISPLAY="Python (self-improving-agents)"
OPENCLAW_MIN="2026.10"
NODE_MIN=22

if [[ -t 1 ]]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B= G= Y= R= N=; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n%ssetup: %s%s\n' "$R" "$*" "$N" >&2; exit 1; }

apple_silicon() { [[ "$(uname -s)" == Darwin && "$(sysctl -n hw.optional.arm64 2>/dev/null)" == 1 ]]; }
openclaw_version() { "$1" --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9]+(\.[0-9]+)?' | head -1; }
at_least() {  # at_least FOUND MIN: dotted versions, numerically
  awk -v a="$1" -v b="$2" 'BEGIN { n = split(a, x, "."); m = split(b, y, ".");
    for (i = 1; i <= (n > m ? n : m); i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 } exit 0 }'
}
skip_openclaw() { [[ "${SKIP_OPENCLAW:-0}" == 1 ]]; }

# --------------------------------------------------------------------------------------------
check() {
  local bad=0 line
  row() { printf '    %-22s %s\n' "$1" "$2"; }
  pass() { row "$1" "${G}$2${N}"; }
  warn() { row "$1" "${Y}$2${N}"; }
  fail() { row "$1" "${R}$2${N}"; bad=1; }

  step "Checking the setup"
  if [[ -x "$VENV/bin/python" ]]; then
    line="$("$VENV/bin/python" -c 'import platform, sys; print(platform.python_version(), platform.machine())')"
    if apple_silicon && [[ "$line" != *arm64 ]]; then
      fail "python" "$line: an Intel Python under Rosetta. Run: rm -rf .venv && make setup"
    else
      pass "python" "$line ($VENV)"
    fi
    if line="$("$VENV/bin/python" -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null)"; then
      pass "mlflow" "$line"
    else
      fail "mlflow" "does not import. Run: make setup"
    fi
    if [[ -f "$VENV/share/jupyter/kernels/$KERNEL_NAME/kernel.json" ]]; then
      pass "jupyter kernel" "$KERNEL_DISPLAY"
    else
      fail "jupyter kernel" "not registered. Run: make setup"
    fi
  else
    fail "python" "no .venv yet. Run: make setup"
  fi

  if skip_openclaw; then
    warn "openclaw" "skipped (SKIP_OPENCLAW=1)"
  else
    if command -v node >/dev/null; then
      line="$(node --version)"
      if at_least "${line#v}" "$NODE_MIN"; then pass "node" "$line"; else fail "node" "$line (need $NODE_MIN+)"; fi
    else
      fail "node" "not installed (need $NODE_MIN+)"
    fi
    if [[ -x "$VENV/bin/openclaw" ]] && line="$(openclaw_version "$VENV/bin/openclaw")" && [[ -n "$line" ]]; then
      if at_least "$line" "$OPENCLAW_MIN"; then pass "openclaw" "$line (.venv/bin/openclaw)"
      else fail "openclaw" "$line (need $OPENCLAW_MIN+). Run: make setup"; fi
    else
      fail "openclaw" "not installed. Run: make setup"
    fi
    line="$(ls "$CONNECTOR"/*.tgz 2>/dev/null | head -1 || true)"
    if [[ -n "$line" ]]; then pass "mlflow connector" "built (${line#"$ROOT"/})"
    else fail "mlflow connector" "not built. Run: make setup"; fi
  fi

  local env key
  for env in harness/.env sia/.env; do
    if [[ ! -f "$ROOT/$env" ]]; then
      fail "$env" "missing. Run: make setup"
    elif key="$(grep -E '^(ANTHROPIC|OPENAI|GEMINI|MISTRAL|GROQ)_API_KEY=.+' "$ROOT/$env" | head -1 | cut -d= -f1)" && [[ -n "$key" ]]; then
      pass "$env" "$key set"
    else
      warn "$env" "no API key yet: offline modes only (add one for live runs)"
    fi
  done

  echo
  if [[ $bad == 0 ]]; then
    printf '%sReady.%s Next steps are in README.md ("After setup").\n' "$G" "$N"
  else
    printf '%sSome checks failed; see above.%s\n' "$R" "$N"
  fi
  return $bad
}

# --------------------------------------------------------------------------------------------
setup() {
  step "Checking prerequisites"
  command -v uv >/dev/null || die "uv is not installed. Install it, open a new terminal, and run make setup again:
    curl -LsSf https://astral.sh/uv/install.sh | sh        (or: brew install uv)"
  info "uv $(uv --version | awk '{print $2}')"
  if ! skip_openclaw; then
    command -v node >/dev/null && command -v npm >/dev/null || die "Node.js $NODE_MIN+ is required for OpenClaw and the connector.
    Install it (brew install node, or nvm install --lts), then run make setup again.
    For Python and the notebooks only: SKIP_OPENCLAW=1 make setup"
    at_least "$(node --version | tr -d v)" "$NODE_MIN" \
      || die "Node.js $NODE_MIN+ is required; found $(node --version). Upgrade it (brew upgrade node, or nvm install --lts)."
    info "node $(node --version)"
  fi

  # One environment for every component. On Apple Silicon, ask for an arm64 Python explicitly:
  # otherwise uv may pick an Intel (x86_64) one if it's installed, and that runs under Rosetta
  # (slowly, and with broken cryptography wheels). A venv built on the wrong architecture is
  # replaced.
  step "Python environment (.venv)"
  local request=()
  if [[ -z "${UV_PYTHON:-}" ]] && apple_silicon; then request=(--python cpython-macos-aarch64); fi
  (cd "$ROOT" && uv sync ${request[@]+"${request[@]}"})
  info "$("$VENV/bin/python" -c 'import platform; print("Python", platform.python_version(), platform.machine())')"

  step "Jupyter kernel"
  "$VENV/bin/python" -m ipykernel install --sys-prefix --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" >/dev/null
  info "\"$KERNEL_DISPLAY\": every notebook in the repo uses it (make notebooks)"

  if ! skip_openclaw; then
    # Into the venv, not a global npm prefix: no sudo, no PATH changes, and `rm -rf .venv` removes it.
    step "OpenClaw (.venv/bin/openclaw)"
    local found=""
    [[ -x "$VENV/bin/openclaw" ]] && found="$(openclaw_version "$VENV/bin/openclaw" || true)"
    if [[ -n "$found" && -z "${OPENCLAW_VERSION:-}" ]] && at_least "$found" "$OPENCLAW_MIN"; then
      info "OpenClaw $found already installed"
    else
      info "installing openclaw@${OPENCLAW_VERSION:-latest} (a minute or two)"
      npm install --global --prefix "$VENV" --no-audit --no-fund --loglevel=error \
        "openclaw@${OPENCLAW_VERSION:-latest}" >/dev/null
      found="$(openclaw_version "$VENV/bin/openclaw" || true)"
      [[ -n "$found" ]] || die "OpenClaw installed but .venv/bin/openclaw --version fails"
      at_least "$found" "$OPENCLAW_MIN" || die "OpenClaw $found is older than $OPENCLAW_MIN"
      info "OpenClaw $found"
    fi

    step "MLflow connector for OpenClaw (connectors/openclaw-mlflow)"
    local tgz
    tgz="$(ls "$CONNECTOR"/*.tgz 2>/dev/null | head -1 || true)"
    if [[ -n "$tgz" && -z "$(find "$CONNECTOR/src" "$CONNECTOR/package.json" "$CONNECTOR/openclaw.plugin.json" -newer "$tgz" 2>/dev/null)" ]]; then
      info "already built: ${tgz#"$ROOT"/}"
    else
      "$CONNECTOR/install.sh" --build-only | sed 's/^/    /'
    fi
    info "each demo installs it into its own OpenClaw profile (make demo, SI_OPENCLAW_MLFLOW=1)"
  fi

  step "Configuration"
  local dir
  for dir in harness sia; do
    if [[ -f "$ROOT/$dir/.env" ]]; then
      info "$dir/.env exists; leaving it alone"
    else
      cp "$ROOT/$dir/.env.example" "$ROOT/$dir/.env"
      info "$dir/.env created from $dir/.env.example: add your API key there"
    fi
  done

  check
}

case "${1:-setup}" in
  setup) setup ;;
  check) check ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
