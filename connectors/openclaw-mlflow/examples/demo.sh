#!/usr/bin/env bash
# A five-minute demo: ask OpenClaw a question, see the trace in MLflow, update its AGENTS.md in
# MLflow, ask again. Run from anywhere; everything lives in $DEMO (default ~/odsc-5min), with its
# own OpenClaw profile and MLflow server, so ~/.openclaw and other servers are never touched.
#
#   ./demo.sh server                 start the MLflow server (port 5050); leave it running
#   ./demo.sh setup                  profile + workspace, install the connector, AGENTS.md v1 -> MLflow
#   ./demo.sh ask "question"         run whatever @production points at, then ask OpenClaw
#
#   ./demo.sh push FILE "message"    register FILE as a new version and point @production at it
#   ./demo.sh use N                  point @production at version N (moving it back is a rollback)
#   ./demo.sh pull                   copy @production into the workspace without asking anything
#   ./demo.sh reset                  delete $DEMO to start over (stop the server first)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DEMO="${DEMO:-$HOME/odsc-5min}"
PORT="${PORT:-5050}"
PY="$REPO/harness/.venv/bin/python"
AGENTS_MD=("$PY" "$HERE/agents_md.py")

export MLFLOW_TRACKING_URI="http://127.0.0.1:$PORT"
export OPENCLAW_STATE_DIR="$DEMO/openclaw" OPENCLAW_CONFIG_PATH="$DEMO/openclaw/openclaw.json"
export MLFLOW_DISABLE_AGENT_HINT=1

die() { echo "demo.sh: $*" >&2; exit 1; }
server_up() { curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null; }
need_server() { server_up || die "no MLflow server on port $PORT: run './demo.sh server' in another terminal"; }
[[ -x "$PY" ]] || die "the harness venv is missing: run 'make install' in $REPO/harness"

cmd="${1:-help}"; shift || true
case "$cmd" in
  server)
    mkdir -p "$DEMO"
    exec "$REPO/harness/.venv/bin/mlflow" server --host 127.0.0.1 --port "$PORT" \
      --backend-store-uri "sqlite:///$DEMO/mlflow.db" --artifacts-destination "$DEMO/artifacts"
    ;;

  setup)
    need_server
    mkdir -p "$DEMO/openclaw" "$DEMO/workspace"
    touch "$DEMO/.odsc-demo"   # marks the folder as this demo's, for `reset`
    cp -R "$REPO/harness/session/knowledge" "$DEMO/workspace/"
    cat > "$OPENCLAW_CONFIG_PATH" <<EOF
{
  "agents": {"defaults": {"workspace": "$DEMO/workspace",
                          "model": {"primary": "anthropic/claude-haiku-5-5"},
                          "skipBootstrap": true}},
  "tools": {"allow": ["read"]},
  "gateway": {"mode": "local", "port": 19557, "bind": "loopback"}
}
EOF
    "$HERE/../install.sh" --tracking-uri "$MLFLOW_TRACKING_URI" --experiment openclaw-demo
    "${AGENTS_MD[@]}" push "$REPO/harness/session/fixes/v2_manual.md" -m "v1: what the agent runs today"
    "${AGENTS_MD[@]}" pull "$DEMO/workspace" >/dev/null
    exp_id="$("$PY" -c 'import mlflow; print(mlflow.get_experiment_by_name("openclaw-demo").experiment_id)')"
    echo
    echo "ready. In MLflow (experiment openclaw-demo):"
    echo "  traces:  $MLFLOW_TRACKING_URI/#/experiments/$exp_id/traces"
    echo "  prompt:  $MLFLOW_TRACKING_URI/#/experiments/$exp_id/prompts  (openclaw-agents-md)"
    echo "then: ./demo.sh ask \"Do you have an uptime SLA?\""
    ;;

  ask)
    [[ $# -ge 1 ]] || die 'usage: ./demo.sh ask "question"'
    need_server
    [[ -f "$OPENCLAW_CONFIG_PATH" ]] || die "not set up yet: run './demo.sh setup'"
    if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$REPO/harness/.env" ]]; then
      ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' "$REPO/harness/.env" | cut -d= -f2- | tr -d '"'"'")"
      export ANTHROPIC_API_KEY
    fi
    unset ANTHROPIC_BASE_URL
    "${AGENTS_MD[@]}" pull "$DEMO/workspace" | sed 's/.*<- /running /'
    echo
    openclaw agent --local --agent main --session-id "demo-$(date +%s)" -m "$*" 2>/dev/null | grep -v '^\['
    ;;

  push)
    [[ $# -ge 1 ]] || die 'usage: ./demo.sh push FILE "message"'
    need_server
    "${AGENTS_MD[@]}" push "$1" -m "${2:-new version}"
    ;;

  use)
    [[ $# -eq 1 ]] || die "usage: ./demo.sh use N"
    need_server
    "${AGENTS_MD[@]}" use "$1"
    ;;

  pull)
    need_server
    "${AGENTS_MD[@]}" pull "$DEMO/workspace"
    ;;

  reset)
    server_up && die "stop the MLflow server first (Ctrl-C in its terminal)"
    [[ -f "$DEMO/.odsc-demo" ]] || die "$DEMO isn't this demo's folder; not deleting it"
    rm -rf "$DEMO" && echo "removed $DEMO"
    ;;

  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
