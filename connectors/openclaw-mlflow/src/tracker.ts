// Turns OpenClaw's typed plugin hooks into one MLflow trace per agent run:
//
//   openclaw_agent (AGENT)          llm_input ... agent_end    conversation in/out, token usage, session
//   ├─ chat <model> (CHAT_MODEL)    model_call_started/ended   one per provider request
//   ├─ <tool> (TOOL)                before/after_tool_call     arguments, result, error
//   └─ ...
//
// Spans are buffered per run and sent as a single OTLP batch when the run ends, so MLflow sees each
// trace complete (root span included) in one request.

import os from "node:os";
import { type AttrValue, type Logger, type SpanRecord, newSpanId, newTraceId, nowNs } from "./otlp.ts";

export interface TrackerOptions {
  export: (spans: SpanRecord[]) => Promise<boolean>;
  captureContent: boolean;
  captureSystemPrompt: boolean;
  maxContentChars: number;
  staleRunMs: number;
  /** How long to wait after agent_end for llm_output, which OpenClaw emits just after it. */
  endGraceMs?: number;
  logger?: Logger;
}

interface RunState {
  runId: string;
  sessionKey?: string;
  root: SpanRecord;
  children: SpanRecord[];
  modelCalls: Map<string, SpanRecord>;
  tools: Map<string, SpanRecord>;
  pendingRequestInputs?: string;   // the messages from llm_input, given to the next model call
  lastModelCall?: SpanRecord;
  usage: { input_tokens: number; output_tokens: number; cache_read_input_tokens: number; cache_creation_input_tokens: number };
  hasUsage: boolean;
  lastActivity: number;
  endTimer?: ReturnType<typeof setTimeout>;
  onFinished?: (exported: Promise<boolean>) => void;
}

// Loose views of the hook payloads; see PluginHook*Event in openclaw's plugin SDK types.
type Ctx = Record<string, any>;
type Ev = Record<string, any>;

const J = (v: unknown) => {
  try { return JSON.stringify(v) ?? "null"; } catch { return JSON.stringify(String(v)); }
};

/**
 * Extra trace tags from MLFLOW_OTEL_TRACE_TAGS (a JSON object), read when each run starts. A harness
 * that shells out to `openclaw agent --local` can use it to link the run to its own trace, e.g.
 * {"parent_trace_id": "tr-...", "prompt_version": "3"}. (Process-wide, so less useful for a Gateway.)
 */
function envTraceTags(): Record<string, string> {
  try {
    const parsed = JSON.parse(process.env.MLFLOW_OTEL_TRACE_TAGS ?? "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? Object.fromEntries(Object.entries(parsed).map(([k, v]) => [k, String(v)]))
      : {};
  } catch {
    return {};
  }
}

/** OpenClaw prefixes user turns with a timestamp, e.g. "[Tue 2026-10-20 16:03 EDT] ". */
export const stripEnvelope = (text: string) => text.replace(/^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}[^\]]*\]\s*/, "");

export class RunTracker {
  private runs = new Map<string, RunState>();
  private latestRunBySession = new Map<string, string>();
  private opts: TrackerOptions;
  private log: Logger;
  private inflight = new Set<Promise<unknown>>();

  constructor(opts: TrackerOptions) {
    this.opts = opts;
    this.log = opts.logger ?? {};
  }

  // ---------------------------------------------------------------------------------------- helpers
  private clip(s: string): string {
    const max = this.opts.maxContentChars;
    return s.length > max ? `${s.slice(0, max)}… [truncated ${s.length - max} chars]` : s;
  }

  private content(v: unknown): string | undefined {
    return this.opts.captureContent ? this.clip(J(v)) : undefined;
  }

  private setContent(span: SpanRecord, key: "mlflow.spanInputs" | "mlflow.spanOutputs", v: unknown) {
    const c = this.content(v);
    if (c !== undefined) span.attributes[key] = c;
  }

  private runIdOf(event: Ev, ctx: Ctx): string | undefined {
    return event?.runId ?? ctx?.runId ?? (ctx?.sessionKey && this.latestRunBySession.get(ctx.sessionKey));
  }

  /** For events that end something: never start a new run (that would be an orphan trace). */
  private getRun(event: Ev, ctx: Ctx): RunState | undefined {
    const runId = this.runIdOf(event, ctx);
    const run = runId ? this.runs.get(runId) : undefined;
    if (run) run.lastActivity = Date.now();
    return run;
  }

  private ensureRun(event: Ev, ctx: Ctx): RunState | undefined {
    const runId = this.runIdOf(event, ctx);
    if (!runId) return undefined;
    let run = this.runs.get(runId);
    if (!run) {
      const sessionKey: string | undefined = ctx?.sessionKey ?? event?.sessionKey ?? undefined;
      const session = sessionKey ?? ctx?.sessionId ?? event?.sessionId;
      const attributes: Record<string, AttrValue> = {
        "mlflow.spanType": "AGENT",
        "mlflow.traceTag.mlflow.traceName": "openclaw_agent",
        "mlflow.message.format": "openai",
        "gen_ai.operation.name": "invoke_agent",
        "openclaw.run_id": runId,
        "user.id": String(ctx?.senderId ?? os.userInfo().username),
      };
      if (session) attributes["session.id"] = String(session);
      for (const [key, tag] of [["agentId", "agent_id"], ["channel", "channel"], ["trigger", "trigger"], ["messageProvider", "message_provider"]]) {
        if (ctx?.[key]) attributes[`mlflow.traceTag.openclaw.${tag}`] = String(ctx[key]);
      }
      if (ctx?.sessionId) attributes["openclaw.session_id"] = String(ctx.sessionId);
      for (const [k, v] of Object.entries(envTraceTags())) attributes[`mlflow.traceTag.${k}`] = v;
      run = {
        runId,
        sessionKey,
        root: { traceId: newTraceId(), spanId: newSpanId(), name: "openclaw_agent", startNs: nowNs(), attributes },
        children: [],
        modelCalls: new Map(),
        tools: new Map(),
        usage: { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
        hasUsage: false,
        lastActivity: Date.now(),
      };
      this.runs.set(runId, run);
      if (sessionKey) this.latestRunBySession.set(sessionKey, runId);
    }
    run.lastActivity = Date.now();
    return run;
  }

  private child(run: RunState, name: string, spanType: string, attributes: Record<string, AttrValue>, startNs = nowNs()): SpanRecord {
    const span: SpanRecord = {
      traceId: run.root.traceId, spanId: newSpanId(), parentSpanId: run.root.spanId,
      name, startNs, attributes: { "mlflow.spanType": spanType, ...attributes },
    };
    run.children.push(span);
    return span;
  }

  /** OpenClaw transcript messages -> OpenAI chat messages (what MLflow's chat view and previews read). */
  toChatMessages(messages: unknown[]): Record<string, unknown>[] {
    const out: Record<string, unknown>[] = [];
    for (const m of messages ?? []) {
      if (!m || typeof m !== "object") continue;
      const msg = m as Record<string, any>;
      const parts = Array.isArray(msg.content) ? msg.content : [{ type: "text", text: msg.content ?? "" }];
      const text = parts.filter((p: any) => p?.type === "text" && typeof p.text === "string").map((p: any) => p.text).join("\n");
      if (msg.role === "toolResult" || msg.role === "tool") {
        out.push({ role: "tool", tool_call_id: msg.toolCallId ?? msg.tool_call_id, content: text });
      } else if (msg.role === "assistant") {
        const calls = parts.filter((p: any) => p?.type === "toolCall").map((p: any) => ({
          id: p.id, type: "function", function: { name: p.name, arguments: J(p.arguments ?? {}) },
        }));
        out.push({ role: "assistant", content: text, ...(calls.length ? { tool_calls: calls } : {}) });
      } else if (typeof msg.role === "string") {
        out.push({ role: msg.role, content: msg.role === "user" ? stripEnvelope(text) : text });
      }
    }
    return out;
  }

  private assistantText(texts: unknown): string {
    return Array.isArray(texts) ? texts.filter((t) => typeof t === "string" && t).join("\n") : "";
  }

  // ------------------------------------------------------------------------------------------ hooks
  onLlmInput(event: Ev, ctx: Ctx) {
    const run = this.ensureRun(event, ctx);
    if (!run) return;
    const messages = [
      ...(this.opts.captureSystemPrompt && event.systemPrompt ? [{ role: "system", content: event.systemPrompt }] : []),
      ...this.toChatMessages(event.historyMessages ?? []),
      { role: "user", content: stripEnvelope(String(event.prompt ?? "")) },
    ];
    const inputs = this.content({ messages });
    if (inputs !== undefined) {
      if (!run.root.attributes["mlflow.spanInputs"]) run.root.attributes["mlflow.spanInputs"] = inputs;
      run.pendingRequestInputs = inputs;
    }
    if (event.model) run.root.attributes["gen_ai.request.model"] = String(event.model);
    if (event.provider) run.root.attributes["gen_ai.provider.name"] = String(event.provider);
    if (Array.isArray(event.tools) && this.opts.captureContent) {
      run.root.attributes["openclaw.tools"] = this.clip(J(event.tools.map((t: any) => t?.name ?? t)));
    }
  }

  onModelCallStarted(event: Ev, ctx: Ctx) {
    const run = this.ensureRun(event, ctx);
    if (!run || !event.callId) return;
    const span = this.child(run, `chat ${event.model ?? "model"}`, "CHAT_MODEL", {
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": String(event.model ?? ""),
      "gen_ai.provider.name": String(event.provider ?? ""),
      "openclaw.call_id": String(event.callId),
      ...(event.api ? { "openclaw.api": String(event.api) } : {}),
    });
    if (run.pendingRequestInputs) {
      span.attributes["mlflow.spanInputs"] = run.pendingRequestInputs;
      run.pendingRequestInputs = undefined;
    }
    run.modelCalls.set(String(event.callId), span);
  }

  onModelCallEnded(event: Ev, ctx: Ctx) {
    const run = this.getRun(event, ctx);
    if (!run) return;
    const span = run.modelCalls.get(String(event.callId));
    if (!span) return;
    run.modelCalls.delete(String(event.callId));
    span.endNs = nowNs();
    for (const [k, a] of [["durationMs", "duration_ms"], ["timeToFirstByteMs", "ttfb_ms"], ["requestPayloadBytes", "request_bytes"], ["responseStreamBytes", "response_bytes"]]) {
      if (typeof event[k] === "number") span.attributes[`openclaw.${a}`] = event[k];
    }
    if (event.outcome === "error") {
      span.status = { code: 2, message: String(event.failureKind ?? event.errorCategory ?? "model call failed") };
    } else {
      span.status = { code: 1 };
    }
    run.lastModelCall = span;
  }

  onLlmOutput(event: Ev, ctx: Ctx) {
    const run = this.getRun(event, ctx);
    if (!run) return;
    const text = this.assistantText(event.assistantTexts);
    if (!text && run.root.attributes["mlflow.spanOutputs"]) return this.endIfEnded(run);
    this.setContent(run.root, "mlflow.spanOutputs", { choices: [{ message: { role: "assistant", content: text } }] });
    // The last request of the attempt produced the final text.
    if (run.lastModelCall && !run.lastModelCall.attributes["mlflow.spanOutputs"]) {
      this.setContent(run.lastModelCall, "mlflow.spanOutputs", { choices: [{ message: { role: "assistant", content: text } }] });
    }
    const model = event.resolvedRef ?? event.model;
    if (model) run.root.attributes["gen_ai.response.model"] = String(event.model ?? model);
    const u = event.usage;
    if (u && typeof u === "object") {
      run.usage.input_tokens += u.input ?? 0;
      run.usage.output_tokens += u.output ?? 0;
      run.usage.cache_read_input_tokens += u.cacheRead ?? 0;
      run.usage.cache_creation_input_tokens += u.cacheWrite ?? 0;
      run.hasUsage = true;
    }
    this.endIfEnded(run);
  }

  /** llm_output arrived after agent_end: the run is complete now. */
  private endIfEnded(run: RunState) {
    if (run.endTimer) {
      clearTimeout(run.endTimer);
      void this.finish(run);
    }
  }

  private toolKey(event: Ev, ctx: Ctx): string {
    return String(event.toolCallId ?? ctx?.toolCallId ?? `${event.toolName ?? ctx?.toolName}`);
  }

  onBeforeToolCall(event: Ev, ctx: Ctx) {
    const run = this.ensureRun(event, ctx);
    if (!run) return;
    const name = String(event.toolName ?? ctx?.toolName ?? "tool");
    const span = this.child(run, name, "TOOL", {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": name,
      ...(event.toolCallId ?? ctx?.toolCallId ? { "gen_ai.tool.call.id": String(event.toolCallId ?? ctx.toolCallId) } : {}),
      ...(event.toolKind ? { "openclaw.tool_kind": String(event.toolKind) } : {}),
    });
    this.setContent(span, "mlflow.spanInputs", event.params ?? {});
    run.tools.set(this.toolKey(event, ctx), span);
  }

  onAfterToolCall(event: Ev, ctx: Ctx) {
    const run = this.getRun(event, ctx);
    if (!run) return;
    const key = this.toolKey(event, ctx);
    let span = run.tools.get(key);
    run.tools.delete(key);
    if (!span) {  // e.g. a call blocked before it started: reconstruct from its duration
      const start = typeof event.durationMs === "number" ? nowNs() - BigInt(Math.round(event.durationMs * 1e6)) : nowNs();
      this.onBeforeToolCall(event, ctx);
      span = run.tools.get(key)!;
      run.tools.delete(key);
      span.startNs = start;
    }
    span.endNs = nowNs();
    if (typeof event.durationMs === "number") span.attributes["openclaw.duration_ms"] = event.durationMs;
    if (event.error) {
      span.status = { code: 2, message: String(event.error).slice(0, 500) };
      this.setContent(span, "mlflow.spanOutputs", { error: event.error });
    } else {
      span.status = { code: 1 };
      this.setContent(span, "mlflow.spanOutputs", event.result ?? null);
    }
  }

  onAgentEnd(event: Ev, ctx: Ctx): Promise<boolean> {
    const runId = this.runIdOf(event, ctx);
    const run = runId ? this.runs.get(runId) : undefined;
    if (!run) return Promise.resolve(true);
    if (!run.root.attributes["mlflow.spanOutputs"] && Array.isArray(event.messages)) {
      const last = this.toChatMessages(event.messages).filter((m) => m.role === "assistant").at(-1);
      if (last) this.setContent(run.root, "mlflow.spanOutputs", { choices: [{ message: last }] });
    }
    if (typeof event.durationMs === "number") run.root.attributes["openclaw.duration_ms"] = event.durationMs;
    run.root.status = event.success === false ? { code: 2, message: String(event.error ?? "agent run failed").slice(0, 500) } : { code: 1 };
    run.root.endNs = nowNs();
    const grace = this.opts.endGraceMs ?? 0;
    if (grace <= 0) return this.finish(run);
    return new Promise((resolve) => {
      run.onFinished = resolve;
      run.endTimer = setTimeout(() => void this.finish(run), grace);
      run.endTimer.unref?.();
    });
  }

  /** End the run's open spans and send the whole trace. */
  private finish(run: RunState, reason?: string): Promise<boolean> {
    const spans = this.collect(run, reason);
    const p = this.opts.export(spans).finally(() => this.inflight.delete(p));
    this.inflight.add(p);
    run.onFinished?.(p);
    return p;
  }

  /** Close the run and return its spans. */
  private collect(run: RunState, reason?: string): SpanRecord[] {
    if (run.endTimer) clearTimeout(run.endTimer);
    this.runs.delete(run.runId);
    if (run.sessionKey && this.latestRunBySession.get(run.sessionKey) === run.runId) this.latestRunBySession.delete(run.sessionKey);
    const end = nowNs();
    for (const span of [...run.modelCalls.values(), ...run.tools.values()]) {
      span.endNs = end;
      span.attributes["openclaw.unfinished"] = true;
    }
    if (run.hasUsage) {
      run.root.attributes["mlflow.chat.tokenUsage"] = J({
        ...run.usage, total_tokens: run.usage.input_tokens + run.usage.output_tokens,
      });
    }
    if (reason) run.root.status = { code: 2, message: reason };
    run.root.endNs ??= end;
    return [run.root, ...run.children];
  }

  /** At process exit: close every open run synchronously and hand back the spans. */
  drainSync(): SpanRecord[] {
    return [...this.runs.values()].flatMap((r) =>
      this.collect(r, r.endTimer ? undefined : "process exited before agent_end"));
  }

  /** Export runs that never reached agent_end (crash, abort, a hook that never fired). */
  sweep(force = false): Promise<unknown> {
    const cutoff = Date.now() - this.opts.staleRunMs;
    const stale = [...this.runs.values()].filter((r) => force || r.lastActivity < cutoff);
    return Promise.all(stale.map((r) => this.finish(r, force ? "process exiting before agent_end" : "run did not end (no agent_end within timeout)")));
  }

  /** Wait for in-flight exports. */
  flush(): Promise<unknown> {
    return Promise.all([...this.inflight]);
  }

  get openRuns(): number {
    return this.runs.size;
  }
}
