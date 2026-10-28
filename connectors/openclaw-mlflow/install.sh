#!/usr/bin/env bash
# Build the MLflow tracing plugin for OpenClaw from this folder and install it into an
# OpenClaw profile, configured to export traces to an MLflow tracking server.
#
#   ./install.sh --tracking-uri http://localhost:5070 --experiment my-agent
#   ./install.sh --tracking-uri http://localhost:5070 --experiment-id 1
#   ./install.sh --build-only            # just produce the .tgz
#
# The profile is whichever one the openclaw CLI would use: ~/.openclaw by default, or the one
# selected by OPENCLAW_STATE_DIR / OPENCLAW_CONFIG_PATH (the harness passes its isolated profile).
# Restart a running gateway afterwards; `openclaw agent --local` runs pick it up immediately.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ID="mlflow-openclaw"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"

tracking_uri="${MLFLOW_TRACKING_URI:-}"
experiment_id="${MLFLOW_EXPERIMENT_ID:-}"
experiment_name=""
build_only=0
run_tests=0

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "install.sh: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracking-uri) tracking_uri="$2"; shift 2 ;;
    --experiment-id) experiment_id="$2"; shift 2 ;;
    --experiment) experiment_name="$2"; shift 2 ;;
    --build-only) build_only=1; shift ;;
    --test) run_tests=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

command -v node >/dev/null || die "node is not on PATH (Node 22+ is required)"
command -v npm >/dev/null || die "npm is not on PATH"
node -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' \
  || die "Node 22+ is required (found $(node --version))"

# --- build -----------------------------------------------------------------------------------
echo "==> building $PLUGIN_ID"
cd "$HERE"
npm ci --no-audit --no-fund --loglevel=error
npm run --silent build
[[ $run_tests == 1 ]] && npm test --silent
rm -f ./*.tgz
tgz="$HERE/$(npm pack --silent 2>/dev/null | tail -1)"
echo "    $tgz"
[[ $build_only == 1 ]] && exit 0

# --- configure -------------------------------------------------------------------------------
command -v "$OPENCLAW_BIN" >/dev/null || die "$OPENCLAW_BIN is not on PATH (npm i -g openclaw)"
[[ -n "$tracking_uri" ]] || die "--tracking-uri is required (or set MLFLOW_TRACKING_URI)"
case "$tracking_uri" in
  http://*|https://*|databricks*) ;;
  *) die "the plugin exports over HTTP, so --tracking-uri must be an MLflow server
    (http://..., https://... or databricks), not '$tracking_uri'. Start one with:
      mlflow server --port 5070" ;;
esac

if [[ -z "$experiment_id" ]]; then
  [[ -n "$experiment_name" ]] || die "pass --experiment NAME or --experiment-id ID"
  # Look the experiment up by name and create it if it does not exist yet.
  experiment_id="$(TRACKING_URI="$tracking_uri" EXPERIMENT="$experiment_name" node --input-type=module -e '
    const base = process.env.TRACKING_URI.replace(/\/+$/, "") + "/api/2.0/mlflow/experiments";
    const name = process.env.EXPERIMENT;
    const got = await fetch(`${base}/get-by-name?experiment_name=${encodeURIComponent(name)}`);
    if (got.ok) {
      console.log((await got.json()).experiment.experiment_id);
    } else if (got.status === 404) {
      const made = await fetch(`${base}/create`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }),
      });
      if (!made.ok) throw new Error(`create ${name}: HTTP ${made.status} ${await made.text()}`);
      console.log((await made.json()).experiment_id);
    } else {
      throw new Error(`get-by-name ${name}: HTTP ${got.status} ${await got.text()}`);
    }
  ')" || die "could not reach MLflow at $tracking_uri"
fi

echo "==> installing into OpenClaw profile ${OPENCLAW_STATE_DIR:-~/.openclaw}"
# --force: a local archive has no ClawHub/npm provenance, so OpenClaw asks for confirmation.
# --accept-capabilities: records consent to the hooks the plugin declares.
"$OPENCLAW_BIN" plugins install --force --accept-capabilities "$tgz" 2>&1 \
  | grep -v "^Extracting\|^Installing to" || true

allow="$("$OPENCLAW_BIN" config get plugins.allow --json 2>/dev/null || true)"
batch="$(ALLOW="$allow" TRACKING_URI="$tracking_uri" EXPERIMENT_ID="$experiment_id" node -e '
  const id = process.argv[1];
  const entry = `plugins.entries.${id}`;
  const ops = [
    { path: `${entry}.enabled`, value: true },
    { path: `${entry}.config.trackingUri`, value: process.env.TRACKING_URI },
    { path: `${entry}.config.experimentId`, value: process.env.EXPERIMENT_ID },
    // OpenClaw 2026.10+ blocks llm_input/llm_output/agent_end for non-bundled plugins without
    // this, which leaves traces with no root or LLM spans (in practice, no traces at all).
    { path: `${entry}.hooks.allowConversationAccess`, value: true },
  ];
  // An existing allowlist would keep the plugin from loading. Never create one: that would
  // disable every other plugin in the profile.
  let allow; try { allow = JSON.parse(process.env.ALLOW); } catch {}
  if (Array.isArray(allow) && !allow.includes(id)) ops.push({ path: "plugins.allow", value: [...allow, id] });
  console.log(JSON.stringify(ops));
' "$PLUGIN_ID")"
"$OPENCLAW_BIN" config set --batch-json "$batch" >/dev/null

echo "==> done: traces from each OpenClaw run now go to $tracking_uri (experiment $experiment_id)"
echo "    Restart the gateway if one is running from this profile."
echo "    \`openclaw mlflow status\` shows the settings (it takes a while to start)."
