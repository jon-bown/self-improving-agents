/**
 * MLflow Tracing service for OpenClaw.
 *
 * Creates an OpenClawPluginService that subscribes to agent lifecycle events
 * via api.on() and creates MLflow traces in real-time. Maps OpenClaw events
 * to a span hierarchy:  root AGENT → child LLM / TOOL / sub-AGENT spans.
 */
import type { OpenClawPluginApi, OpenClawPluginService } from 'openclaw/plugin-sdk/plugin-entry';
import { startSpan as coreStartSpan, flushTraces as coreFlushTraces, InMemoryTraceManager } from '@mlflow/core';
export type SharedSdk = {
    startSpan: typeof coreStartSpan;
    flushTraces: typeof coreFlushTraces;
    traceManager: typeof InMemoryTraceManager;
    trackingUri: string;
    experimentId: string;
};
export declare function acquireSharedSdk(trackingUri: string, experimentId: string, initialize?: (config: {
    trackingUri: string;
    experimentId: string;
}) => void): SharedSdk;
type SpanLike = ReturnType<typeof coreStartSpan>;
/**
 * Write trace-level metadata (what MLflow's Sessions view and `metadata.*` searches
 * read). Span attributes alone don't reach it, and updateCurrentTrace() needs an
 * active span context, which OpenClaw's hook callbacks don't have.
 */
export declare function setTraceMetadata(rootSpan: SpanLike, metadata: Record<string, string>): void;
export interface PendingChild {
    span: SpanLike;
    name: string;
}
export interface ActiveTrace {
    rootSpan: SpanLike;
    pendingLlm: PendingChild | null;
    pendingTools: Map<string, PendingChild>;
    pendingSubagents: Map<string, PendingChild>;
    tokenUsage: {
        inputTokens: number;
        outputTokens: number;
        totalTokens: number;
        cost: number;
    };
    costMeta: {
        costUsd?: number;
        contextLimit?: number;
        contextUsed?: number;
        model?: string;
        provider?: string;
    };
    firstPrompt: string;
    lastResponse: string;
    agentEndData: {
        success?: boolean;
        error?: string;
        durationMs?: number;
        messages?: unknown[];
    } | null;
    channelId?: string;
    trigger?: string;
    model?: string;
    provider?: string;
    lastActivityMs: number;
}
export declare function evictOldest<K, V>(map: Map<K, V>, maxSize: number): void;
export declare function toolKey(toolName: string, toolCallId?: string): string;
export declare function normalizeProvider(value: unknown): string | undefined;
export declare function asNonEmptyString(value: unknown): string | undefined;
/**
 * Sanitize OpenClaw internal markers from text before storing in MLflow.
 */
export declare function sanitizeOpenClawText(value: string): string;
export declare function sanitizeValue(value: unknown): unknown;
/**
 * End any child spans that weren't explicitly closed by their matching
 * `*_end` event (crash, timeout, or malformed event sequence).
 */
export declare function closePendingSpans(trace: ActiveTrace): void;
export declare function attachTokenUsage(rootSpan: SpanLike, tokenUsage: ActiveTrace['tokenUsage']): void;
/**
 * Build the outputs payload for the root agent span. Falls back to
 * `agent_end.messages` when no `llm_output` was captured. Mutates
 * `trace.lastResponse` when the fallback kicks in so the final state is
 * observable.
 */
export declare function resolveTraceOutputs(trace: ActiveTrace): {
    outputs: Record<string, unknown>;
    errorStatus?: string;
};
export declare function attachTraceMetadata(rootSpan: SpanLike, trace: ActiveTrace): void;
export declare function createMLflowService(api: OpenClawPluginApi, pluginConfig?: Record<string, unknown>): OpenClawPluginService & {
    registerHooks: () => void;
};
export {};
//# sourceMappingURL=service.d.ts.map