// OpenClaw -> MLflow tracing plugin. Each agent run becomes one MLflow trace (sent over OTLP/HTTP),
// with a span per model request and per tool call. See README.md for setup.

import { MlflowExporter } from "./src/otlp.ts";
import { RunTracker } from "./src/tracker.ts";

const PLUGIN_ID = "mlflow-otel";
const VERSION = "0.1.0";

interface PluginConfig {
  enabled?: boolean;
  trackingUri?: string;
  experimentId?: string;
  experimentName?: string;
  headers?: Record<string, string>;
  captureContent?: boolean;
  captureSystemPrompt?: boolean;
  maxContentChars?: number;
  serviceName?: string;
}

const envFlag = (name: string) => {
  const v = process.env[name];
  return v === undefined ? undefined : !/^(0|false|no|off)$/i.test(v);
};

// OpenClaw registers a plugin more than once per process (a "full" registry plus discovery
// registries), and the agent's hooks may run from any of them. Keep one tracker per process so a
// run's events always land in the same place, and install the exit handler once.
interface Shared { key: string; tracker: RunTracker; exporter: MlflowExporter; log: Logger }
type Logger = { info: (m: string) => void; warn: (m: string) => void; debug: (m: string) => void };
const SHARED = Symbol.for("openclaw.mlflow-otel");
const g = globalThis as { [SHARED]?: Shared };

function shared(cfg: PluginConfig, log: Logger): Shared | undefined {
  const trackingUri = (cfg.trackingUri ?? process.env.MLFLOW_TRACKING_URI ?? "http://127.0.0.1:5000").replace(/\/+$/, "");
  if (!/^https?:\/\//.test(trackingUri)) {
    log.warn(`${PLUGIN_ID}: the MLflow tracking URI must be an http(s) server URL, got "${trackingUri}"; tracing is off`);
    return undefined;
  }
  const target = {
    trackingUri,
    experimentId: cfg.experimentId ?? process.env.MLFLOW_EXPERIMENT_ID,
    experimentName: cfg.experimentName ?? process.env.MLFLOW_EXPERIMENT_NAME ?? "openclaw",
    headers: cfg.headers,
  };
  const options = {
    captureContent: cfg.captureContent ?? envFlag("MLFLOW_OTEL_CAPTURE_CONTENT") ?? true,
    captureSystemPrompt: cfg.captureSystemPrompt ?? envFlag("MLFLOW_OTEL_CAPTURE_SYSTEM_PROMPT") ?? false,
    maxContentChars: cfg.maxContentChars ?? 20_000,
  };
  const key = JSON.stringify([target, options, cfg.serviceName]);
  if (g[SHARED]?.key === key) return g[SHARED];

  const exporter = new MlflowExporter(
    target,
    { "service.name": cfg.serviceName ?? "openclaw", "service.version": VERSION },
    { name: PLUGIN_ID, version: VERSION },
    log,
  );
  const tracker = new RunTracker({
    ...options,
    export: (spans) => exporter.export(spans),
    staleRunMs: 15 * 60_000,
    endGraceMs: 250,
    logger: log,
  });
  const first = !g[SHARED];
  g[SHARED] = { key, tracker, exporter, log };
  if (first) {
    // Gateway mode: export runs that never ended. Unref'd, so it never keeps a CLI process alive.
    setInterval(() => void g[SHARED]?.tracker.sweep(), 60_000).unref();
    // `openclaw agent --local` calls process.exit() ~500ms after the run, which can cut off an
    // async export. Send whatever is still open or in flight synchronously.
    process.once("exit", () => {
      const s = g[SHARED];
      if (s) s.exporter.exportPendingSync(s.tracker.drainSync());
    });
  }
  return g[SHARED];
}

export default {
  id: PLUGIN_ID,
  name: "MLflow Tracing (OTel)",
  description: "Export OpenClaw agent runs, model calls and tool calls to an MLflow server as traces.",
  register(api: any) {
    const cfg: PluginConfig = api.pluginConfig ?? {};
    if (cfg.enabled === false || envFlag("MLFLOW_OTEL_DISABLED")) return;
    const log: Logger = {
      info: (m) => api.logger?.info?.(m),
      warn: (m) => api.logger?.warn?.(m),
      debug: (m) => api.logger?.debug?.(m),
    };
    if (!shared(cfg, log)) return;

    // Hook handlers never throw into the agent and never return a value (which could alter its behavior).
    const on = (hook: string, fn: (t: RunTracker, event: any, ctx: any) => unknown) =>
      api.on(hook, (event: any, ctx: any) => {
        const s = g[SHARED];
        if (!s) return undefined;
        try {
          const r = fn(s.tracker, event, ctx);
          if (r instanceof Promise) r.catch((err) => s.log.warn(`${PLUGIN_ID}: ${hook} failed: ${String(err)}`));
        } catch (err) {
          s.log.warn(`${PLUGIN_ID}: ${hook} failed: ${String(err)}`);
        }
        return undefined;
      });

    // llm_input / llm_output / agent_end carry conversation content: they need
    // plugins.entries.mlflow-otel.hooks.allowConversationAccess = true.
    on("llm_input", (t, e, c) => t.onLlmInput(e, c));
    on("llm_output", (t, e, c) => t.onLlmOutput(e, c));
    on("model_call_started", (t, e, c) => t.onModelCallStarted(e, c));
    on("model_call_ended", (t, e, c) => t.onModelCallEnded(e, c));
    on("before_tool_call", (t, e, c) => t.onBeforeToolCall(e, c));
    on("after_tool_call", (t, e, c) => t.onAfterToolCall(e, c));
    on("agent_end", (t, e, c) => t.onAgentEnd(e, c));
  },
};
