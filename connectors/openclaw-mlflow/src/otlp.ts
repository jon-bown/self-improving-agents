// Minimal OTLP/HTTP JSON exporter for MLflow. No dependencies: MLflow's /v1/traces endpoint accepts
// `application/json`, and an agent run is small enough to send as one batch when it ends.

import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";

// Runs in a child process: reads {target, body} on stdin, resolves the experiment, POSTs the body.
const SYNC_EXPORT_SCRIPT = `
let raw = ""; for await (const c of process.stdin) raw += c;
const { target, body } = JSON.parse(raw);
const h = { ...target.headers, "content-type": "application/json" };
let id = target.experimentId;
if (!id) {
  const base = target.trackingUri + "/api/2.0/mlflow/experiments";
  const name = target.experimentName ?? "Default";
  let r = await fetch(base + "/get-by-name?experiment_name=" + encodeURIComponent(name), { headers: h });
  if (r.ok) id = (await r.json()).experiment.experiment_id;
  else { r = await fetch(base + "/create", { method: "POST", headers: h, body: JSON.stringify({ name }) }); id = (await r.json()).experiment_id; }
}
const r = await fetch(target.trackingUri + "/v1/traces", { method: "POST", headers: { ...h, "x-mlflow-experiment-id": String(id) }, body });
if (!r.ok) { console.error("HTTP " + r.status + " " + (await r.text()).slice(0, 300)); process.exit(1); }
`;

export type AttrValue = string | number | boolean;

export interface SpanRecord {
  traceId: string;        // 32 hex chars
  spanId: string;         // 16 hex chars
  parentSpanId?: string;
  name: string;
  startNs: bigint;
  endNs?: bigint;
  attributes: Record<string, AttrValue>;
  status?: { code: 1 | 2; message?: string };   // 1 = OK, 2 = ERROR
}

export interface Logger {
  info?: (msg: string) => void;
  warn?: (msg: string) => void;
  debug?: (msg: string) => void;
}

export const newTraceId = () => randomBytes(16).toString("hex");
export const newSpanId = () => randomBytes(8).toString("hex");
export const nowNs = () => BigInt(Math.round((performance.timeOrigin + performance.now()) * 1e6));

function otlpValue(v: AttrValue) {
  if (typeof v === "boolean") return { boolValue: v };
  if (typeof v === "number") return Number.isInteger(v) ? { intValue: String(v) } : { doubleValue: v };
  return { stringValue: v };
}

const otlpAttrs = (attrs: Record<string, AttrValue>) =>
  Object.entries(attrs)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([key, v]) => ({ key, value: otlpValue(v) }));

export function toOtlpJson(spans: SpanRecord[], resource: Record<string, AttrValue>, scope: { name: string; version: string }) {
  return {
    resourceSpans: [{
      resource: { attributes: otlpAttrs(resource) },
      scopeSpans: [{
        scope,
        spans: spans.map((s) => ({
          traceId: s.traceId,
          spanId: s.spanId,
          ...(s.parentSpanId ? { parentSpanId: s.parentSpanId } : {}),
          name: s.name,
          kind: 1,  // INTERNAL
          startTimeUnixNano: String(s.startNs),
          endTimeUnixNano: String(s.endNs ?? nowNs()),
          attributes: otlpAttrs(s.attributes),
          status: s.status ?? { code: 0 },
        })),
      }],
    }],
  };
}

export interface MlflowTarget {
  trackingUri: string;
  experimentId?: string;
  experimentName?: string;
  headers?: Record<string, string>;
}

/** Sends batches of finished spans to `${trackingUri}/v1/traces`. Resolves the experiment lazily. */
export class MlflowExporter {
  private experimentId?: Promise<string>;
  private target: MlflowTarget;
  private resource: Record<string, AttrValue>;
  private scope: { name: string; version: string };
  private logger: Logger;

  constructor(target: MlflowTarget, resource: Record<string, AttrValue>, scope: { name: string; version: string }, logger: Logger = {}) {
    this.target = { ...target, trackingUri: target.trackingUri.replace(/\/+$/, "") };
    this.resource = resource;
    this.scope = scope;
    this.logger = logger;
  }

  /** The OTLP header must be a numeric experiment ID, so a name is looked up (and created if missing). */
  resolveExperimentId(): Promise<string> {
    if (this.target.experimentId) return Promise.resolve(this.target.experimentId);
    this.experimentId ??= this.lookupExperiment(this.target.experimentName ?? "Default").catch((err) => {
      this.experimentId = undefined;  // retry on the next export
      throw err;
    });
    return this.experimentId;
  }

  private async lookupExperiment(name: string): Promise<string> {
    const base = `${this.target.trackingUri}/api/2.0/mlflow/experiments`;
    const headers = { ...this.target.headers, "content-type": "application/json" };
    const got = await fetch(`${base}/get-by-name?experiment_name=${encodeURIComponent(name)}`, { headers, signal: AbortSignal.timeout(5000) });
    if (got.ok) return (await got.json()).experiment.experiment_id;
    const created = await fetch(`${base}/create`, {
      method: "POST", headers, body: JSON.stringify({ name }), signal: AbortSignal.timeout(5000),
    });
    if (!created.ok) throw new Error(`could not get or create MLflow experiment "${name}": HTTP ${created.status}`);
    return (await created.json()).experiment_id;
  }

  /** Request bodies whose async export hasn't finished; `exportPendingSync` sends them at process exit. */
  private pending = new Set<string>();

  async export(spans: SpanRecord[]): Promise<boolean> {
    if (spans.length === 0) return true;
    const body = JSON.stringify(toOtlpJson(spans, this.resource, this.scope));
    this.pending.add(body);
    try {
      return await this.post(body);
    } finally {
      this.pending.delete(body);
    }
  }

  /**
   * One-shot CLIs (`openclaw agent --local`) call process.exit() right after the run, before an async
   * fetch can finish. From a synchronous `exit` handler, send what's left with a blocking child process.
   */
  exportPendingSync(spans: SpanRecord[] = []): void {
    const bodies = [...this.pending];
    if (spans.length) bodies.push(JSON.stringify(toOtlpJson(spans, this.resource, this.scope)));
    this.pending.clear();
    for (const body of bodies) {
      const res = spawnSync(process.execPath, ["--input-type=module", "-e", SYNC_EXPORT_SCRIPT], {
        input: JSON.stringify({ target: this.target, body }), timeout: 8000, encoding: "utf8",
        env: { PATH: process.env.PATH ?? "" },
      });
      if (res.status !== 0) this.logger.warn?.(`mlflow-otel: export at exit failed: ${(res.stderr || res.error?.message || "").slice(0, 300)}`);
    }
  }

  private async post(body: string): Promise<boolean> {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        const res = await fetch(`${this.target.trackingUri}/v1/traces`, {
          method: "POST",
          headers: { ...this.target.headers, "content-type": "application/json", "x-mlflow-experiment-id": await this.resolveExperimentId() },
          body,
          signal: AbortSignal.timeout(10_000),
        });
        if (res.ok) return true;
        this.logger.warn?.(`mlflow-otel: export failed with HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
        if (res.status < 500) return false;   // a 4xx won't succeed on retry
      } catch (err) {
        this.logger.warn?.(`mlflow-otel: export to ${this.target.trackingUri} failed (attempt ${attempt}): ${String(err)}`);
      }
    }
    return false;
  }
}
