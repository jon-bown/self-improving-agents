#!/usr/bin/env bash
# A five-minute demo: ask OpenClaw a question, see the trace in MLflow, update its AGENTS.md in
# MLflow, ask again. Run from anywhere; everything lives in $DEMO (default ~/odsc-5min), with its
# own OpenClaw profile and MLflow server, so ~/.openclaw and other servers are never touched.
#
#   ./demo.sh server                 start the MLflow server (port 5050); leave it running
#   ./demo.sh setup                  profile + workspace, install the connector, AGENTS.md v1 -> MLflow
#   ./demo.sh ask "question"         run whatever @production points at, then ask OpenClaw
#   ./demo.sh traffic                ask a handful of everyday questions, so there's history to analyze
#   ./demo.sh detect                 MLflow's automatic issue detection over the traces: named Issues,
#                                    each with a root cause and the traces it affects
#   ./demo.sh score [--all]          score new traces (--all: every trace) with read_knowledge and
#                                    grounded_in_docs, a judge that sees the files the agent read
#   ./demo.sh improve                OpenClaw rewrites its own AGENTS.md from the open Issue (or, without
#                                    one, the traces score failed)
#                                    (a candidate version; @production doesn't move)
#   ./demo.sh gate [N]               test the candidate on the failing questions + a holdout one;
#                                    promote it automatically only if every check passes
#
#   More MLflow, on the same traces (seconds each, unless it says otherwise):
#   ./demo.sh traces ["FILTER"]      search traces (any MLflow filter); latency, tokens and verdicts
#                                    per AGENTS.md version
#   ./demo.sh prompts [A B]          every AGENTS.md version: aliases, tags, traces it made; diff A B
#   ./demo.sh feedback yes|no "why"  your verdict on the newest trace (--trace ID for another)
#   ./demo.sh expect "answer"        your expected answer for the newest trace
#   ./demo.sh judges                 MLflow's built-in LLM judges on the newest trace (~20 s)
#   ./demo.sh gateway                an MLflow AI Gateway endpoint for the judge model (key kept on
#                                    the server; calls traced with their tokens)
#   ./demo.sh monitor on|off|status  the server scores every new trace by itself, about once a minute
#   ./demo.sh chat                   a three-turn conversation, judged as one session (~2 min)
#   ./demo.sh dataset                an evaluation dataset: golden cases + your expected answers
#   ./demo.sh eval [N]               run version N (default @production) on the dataset (~2 min);
#                                    compare eval runs in the UI
#
#   ./demo.sh push FILE "message"    register FILE as a new version and point @production at it
#                                    (agents/v2.md is the fix, if you'd rather not type it in the UI)
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
FEATURES=("$PY" "$HERE/mlflow_features.py")

export MLFLOW_TRACKING_URI="http://127.0.0.1:$PORT"
export OPENCLAW_STATE_DIR="$DEMO/openclaw" OPENCLAW_CONFIG_PATH="$DEMO/openclaw/openclaw.json"
export MLFLOW_DISABLE_AGENT_HINT=1

die() { echo "demo.sh: $*" >&2; exit 1; }
ours() {  # $DEMO is this demo's: marked, or holding only what `server` makes (an older server didn't mark it)
  [[ -f "$DEMO/.odsc-demo" ]] && return
  [[ -z "$(ls -A "$DEMO" 2>/dev/null | grep -vxE 'mlflow\.db|artifacts')" ]]
}
claim() {  # create and mark $DEMO, refusing a folder that's someone else's
  [[ ! -e "$DEMO" ]] || ours || die "$DEMO already exists and isn't this demo's; set DEMO= to use another folder"
  mkdir -p "$DEMO" && touch "$DEMO/.odsc-demo"   # the marker `reset` looks for
}
server_up() { curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null; }
need_server() { server_up || die "no MLflow server on port $PORT: run './demo.sh server' in another terminal"; }
load_key() {  # the Anthropic key from harness/.env, never printed; for OpenClaw and the judge
  if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$REPO/harness/.env" ]]; then
    ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' "$REPO/harness/.env" | cut -d= -f2- | tr -d '"'"'")"
    export ANTHROPIC_API_KEY
  fi
  unset ANTHROPIC_BASE_URL
}
awake() {  # keep a Mac from idle-sleeping mid-turn: a sleeping Mac makes OpenClaw's turn time out
  if command -v caffeinate >/dev/null; then caffeinate -i "$@"; else "$@"; fi
}
[[ -x "$PY" ]] || die "the harness venv is missing: run 'make install' in $REPO/harness"

cmd="${1:-help}"; shift || true
case "$cmd" in
  server)
    claim
    exec "$REPO/harness/.venv/bin/mlflow" server --host 127.0.0.1 --port "$PORT" \
      --backend-store-uri "sqlite:///$DEMO/mlflow.db" --artifacts-destination "$DEMO/artifacts"
    ;;

  setup)
    need_server
    claim
    mkdir -p "$DEMO/openclaw" "$DEMO/workspace"
    # The policies, minus support_process.md: its "escalate anything not covered" rule would
    # fight v1 and hide the failure. The fix (agents/v2.md) adds that rule to AGENTS.md instead.
    mkdir -p "$DEMO/workspace/knowledge"
    cp "$REPO/harness/session/knowledge/"{billing,account_access,product_howto}.md "$DEMO/workspace/knowledge/"
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
    "${AGENTS_MD[@]}" push "$HERE/agents/v1.md" -m "v1: what the agent runs today"
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
    load_key
    "${AGENTS_MD[@]}" pull "$DEMO/workspace" | sed 's/.*<- /running /'
    echo
    since=$(( $(date +%s) * 1000 ))
    awake openclaw agent --local --agent main --session-id "demo-$(date +%s)" -m "$*" 2>/dev/null | grep -v '^\['
    "${FEATURES[@]}" link --since "$since"   # the trace -> the AGENTS.md version that answered
    ;;

  traffic)
    need_server
    [[ -f "$OPENCLAW_CONFIG_PATH" ]] || die "not set up yet: run './demo.sh setup'"
    load_key
    "${AGENTS_MD[@]}" pull "$DEMO/workspace" | sed 's/.*<- /running /'
    # Three the policies cover, three they don't. Not the refund window: that's the gate's holdout.
    since=$(( $(date +%s) * 1000 ))
    for q in "How do I reset my password?" "What's the API rate limit?" "Can I export my data to CSV?" \
             "Where is my data hosted?" "How many seats does the Pro plan include?" "Do you have an uptime SLA?"; do
      echo "   asking: $q"
      awake openclaw agent --local --agent main --session-id "demo-$(date +%s)" -m "$q" >/dev/null 2>&1 \
        || echo "   (that one failed; carrying on)"
    done
    "${FEATURES[@]}" link --since "$since"
    echo "done. Next: ./demo.sh detect"
    ;;

  score)
    need_server
    load_key
    awake "$PY" "$HERE/self_improvement_demo.py" --experiment openclaw-demo --score-only "$@"
    ;;

  detect|improve|gate)
    need_server
    load_key
    awake "$PY" "$HERE/self_improve_loop.py" --demo "$DEMO" "$cmd" "$@"
    ;;

  traces|prompts|feedback|expect|dataset)
    need_server
    "${FEATURES[@]}" "$cmd" "$@"
    ;;

  judges|gateway|monitor)
    need_server
    load_key
    "${FEATURES[@]}" "$cmd" "$@"
    ;;

  chat|eval)
    need_server
    [[ -f "$OPENCLAW_CONFIG_PATH" ]] || die "not set up yet: run './demo.sh setup'"
    load_key
    awake "${FEATURES[@]}" --demo "$DEMO" "$cmd" "$@"
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
    [[ -e "$DEMO" ]] || { echo "nothing to remove at $DEMO"; exit 0; }
    ours || die "$DEMO isn't this demo's folder; not deleting it"
    rm -rf "$DEMO" && echo "removed $DEMO"
    ;;

  *)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"   # the header above
    ;;
esac
