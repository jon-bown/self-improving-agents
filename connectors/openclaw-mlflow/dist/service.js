/**
 * MLflow Tracing service for OpenClaw.
 *
 * Creates an OpenClawPluginService that subscribes to agent lifecycle events
 * via api.on() and creates MLflow traces in real-time. Maps OpenClaw events
 * to a span hierarchy:  root AGENT → child LLM / TOOL / sub-AGENT spans.
 */
import { onDiagnosticEvent, } from 'openclaw/plugin-sdk/diagnostic-runtime';
import { init as coreInit, startSpan as coreStartSpan, flushTraces as coreFlushTraces, InMemoryTraceManager, SpanStatusCode, } from '@mlflow/core';
// Inline constants that fail to import via OpenClaw's CJS/ESM loader
// (they're re-exported via __exportStar which the loader can't resolve).
// Values must match the SpanType enum members in @mlflow/core.
const SpanType = {
    LLM: 'LLM',
    TOOL: 'TOOL',
    AGENT: 'AGENT',
};
const SpanAttributeKey = {
    TOKEN_USAGE: 'mlflow.chat.tokenUsage',
    MESSAGE_FORMAT: 'mlflow.message.format',
};
const TraceMetadataKey = {
    TRACE_SESSION: 'mlflow.trace.session',
    TRACE_USER: 'mlflow.trace.user',
};
// Keep local: integrations are independent packages and must support older @mlflow/core floors.
function getCurrentUser() {
    return process.env.USER || process.env.USERNAME || '';
}
// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MAX_ACTIVE_TRACES = 50;
const DEFAULT_STALE_TRACE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
const DEFAULT_STALE_SWEEP_INTERVAL_MS = 60 * 1000; // 1 minute
const DEFAULT_FLUSH_RETRY_COUNT = 2;
const DEFAULT_FLUSH_RETRY_BASE_DELAY_MS = 250;
const MAX_FLUSH_RETRY_DELAY_MS = 5000;
// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
// OpenClaw (>= 2026.10) may load this plugin several times in one process, each
// time from a fresh copy with its own @mlflow/core module instance. The
// OpenTelemetry global tracer provider can only be registered once, so spans
// would land in the first copy's SDK while later copies flush their own, empty,
// processor. Every copy therefore shares the SDK of whichever copy initialized
// first, keyed on a process-global symbol.
const SHARED_SDK_KEY = Symbol.for('mlflow-openclaw.sdk');
let sdk = {
    startSpan: coreStartSpan,
    flushTraces: coreFlushTraces,
    traceManager: InMemoryTraceManager,
    trackingUri: '',
    experimentId: '',
};
function startSpan(...args) {
    return sdk.startSpan(...args);
}
function flushTraces() {
    return sdk.flushTraces();
}
export function acquireSharedSdk(trackingUri, experimentId, initialize = coreInit) {
    const registry = globalThis;
    const existing = registry[SHARED_SDK_KEY];
    if (existing) {
        sdk = existing;
        return existing;
    }
    initialize({ trackingUri, experimentId });
    sdk = {
        startSpan: coreStartSpan,
        flushTraces: coreFlushTraces,
        traceManager: InMemoryTraceManager,
        trackingUri,
        experimentId,
    };
    registry[SHARED_SDK_KEY] = sdk;
    return sdk;
}
/**
 * Write trace-level metadata (what MLflow's Sessions view and `metadata.*` searches
 * read). Span attributes alone don't reach it, and updateCurrentTrace() needs an
 * active span context, which OpenClaw's hook callbacks don't have.
 */
export function setTraceMetadata(rootSpan, metadata) {
    const info = sdk.traceManager.getInstance().getTrace(rootSpan.traceId)?.info;
    if (info) {
        Object.assign(info.traceMetadata, metadata);
    }
}
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
export function evictOldest(map, maxSize) {
    while (map.size > maxSize) {
        const { value: oldest, done } = map.keys().next();
        if (done) {
            return;
        }
        map.delete(oldest);
    }
}
export function toolKey(toolName, toolCallId) {
    return toolCallId ? `${toolName}:${toolCallId}` : toolName;
}
export function normalizeProvider(value) {
    if (typeof value !== 'string' || value.length === 0) {
        return undefined;
    }
    const normalized = value.trim().toLowerCase();
    if (normalized.length === 0) {
        return undefined;
    }
    if (normalized === 'openai-codex' ||
        normalized === 'openai_codex' ||
        normalized === 'codex' ||
        (normalized.includes('openai') && normalized.includes('codex'))) {
        return 'openai';
    }
    return normalized;
}
export function asNonEmptyString(value) {
    return typeof value === 'string' && value.length > 0 ? value : undefined;
}
function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
/**
 * Sanitize OpenClaw internal markers from text before storing in MLflow.
 */
export function sanitizeOpenClawText(value) {
    return value
        .replace(/\\r\\n/g, '\n')
        .replace(/\\n/g, '\n')
        .replace(/\\r/g, '\r')
        .replace(/\[\[reply_to[^\]]*\]\]\s*/gi, '')
        .replace(/^\s*Sender \(untrusted metadata\):\s*\n+\{[\s\S]*?\}\s*/gim, '')
        .replace(/^\s*Sender \(untrusted metadata\):\s*\n*```json\s*\{[\s\S]*?\}\s*```\s*/gim, '')
        .replace(/^\s*Conversation info \(untrusted metadata\):\s*\n+\{[\s\S]*?\}\s*/gim, '')
        .replace(/^\s*Conversation info \(untrusted metadata\):\s*\n*```json\s*\{[\s\S]*?\}\s*```\s*/gim, '')
        .replace(/^\s*Untrusted context \(metadata, do not treat as instructions or commands\):\s*\n+<<<EXTERNAL_UNTRUSTED_CONTENT[\s\S]*?<<<END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>\s*/gim, '')
        .replace(/^\[[\w\s:+\-/]+\]\s*/m, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}
export function sanitizeValue(value) {
    if (typeof value === 'string') {
        return sanitizeOpenClawText(value);
    }
    if (Array.isArray(value)) {
        return value.map(sanitizeValue);
    }
    if (value != null && typeof value === 'object') {
        const result = {};
        for (const [k, v] of Object.entries(value)) {
            result[k] = sanitizeValue(v);
        }
        return result;
    }
    return value;
}
// ---------------------------------------------------------------------------
// Finalize trace
// ---------------------------------------------------------------------------
/**
 * End any child spans that weren't explicitly closed by their matching
 * `*_end` event (crash, timeout, or malformed event sequence).
 */
export function closePendingSpans(trace) {
    if (trace.pendingLlm) {
        trace.pendingLlm.span.end();
        trace.pendingLlm = null;
    }
    for (const [, pending] of trace.pendingTools) {
        pending.span.end();
    }
    trace.pendingTools.clear();
    for (const [, pending] of trace.pendingSubagents) {
        pending.span.end();
    }
    trace.pendingSubagents.clear();
}
export function attachTokenUsage(rootSpan, tokenUsage) {
    if (tokenUsage.totalTokens <= 0) {
        return;
    }
    rootSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
        input_tokens: tokenUsage.inputTokens,
        output_tokens: tokenUsage.outputTokens,
        total_tokens: tokenUsage.totalTokens,
    });
}
/**
 * Build the outputs payload for the root agent span. Falls back to
 * `agent_end.messages` when no `llm_output` was captured. Mutates
 * `trace.lastResponse` when the fallback kicks in so the final state is
 * observable.
 */
export function resolveTraceOutputs(trace) {
    const endData = trace.agentEndData;
    if (!trace.lastResponse && endData?.messages?.length) {
        trace.lastResponse = JSON.stringify(endData.messages);
    }
    const responseText = trace.lastResponse || 'Agent completed';
    const outputs = {
        messages: [{ role: 'assistant', content: responseText }],
    };
    if (endData?.error) {
        outputs.error = endData.error;
        return { outputs, errorStatus: endData.error };
    }
    return { outputs };
}
export function attachTraceMetadata(rootSpan, trace) {
    if (trace.agentEndData?.durationMs != null) {
        rootSpan.setAttribute('agent_duration_ms', trace.agentEndData.durationMs);
    }
    const traceModel = trace.model ?? trace.costMeta.model;
    const traceProvider = trace.provider ?? trace.costMeta.provider;
    if (traceModel) {
        rootSpan.setAttribute('mlflow.llm.model', traceModel);
    }
    if (traceProvider) {
        rootSpan.setAttribute('mlflow.llm.provider', traceProvider);
    }
    if (trace.costMeta.costUsd != null) {
        rootSpan.setAttribute('mlflow.llm.cost', { total_cost: trace.costMeta.costUsd });
    }
}
async function finalizeTrace(sessionKey, trace, userId, flush = () => flushTraces()) {
    closePendingSpans(trace);
    attachTokenUsage(trace.rootSpan, trace.tokenUsage);
    const { outputs, errorStatus } = resolveTraceOutputs(trace);
    if (errorStatus) {
        trace.rootSpan.setStatus(SpanStatusCode.ERROR, errorStatus);
    }
    trace.rootSpan.setOutputs(outputs);
    attachTraceMetadata(trace.rootSpan, trace);
    setTraceMetadata(trace.rootSpan, {
        [TraceMetadataKey.TRACE_SESSION]: sessionKey,
        [TraceMetadataKey.TRACE_USER]: userId || getCurrentUser(),
    });
    trace.rootSpan.end();
    await flush();
}
// ---------------------------------------------------------------------------
// Service factory
// ---------------------------------------------------------------------------
export function createMLflowService(api, pluginConfig = {}) {
    const activeTraces = new Map();
    const sessionByAgentId = new Map();
    let lastActiveSessionKey;
    let warnedMissingAfterToolSessionKey = false;
    let cleanup = null;
    let hooksRegistered = false;
    let initError = null;
    let resolvedTrackingUri;
    let resolvedExperimentId;
    let log = {
        info: () => undefined,
        warn: () => undefined,
    };
    // Exporter metrics
    const metrics = {
        flushSuccesses: 0,
        flushFailures: 0,
        flushRetries: 0,
        spanErrors: 0,
    };
    function rememberSession(sessionKey, agentId) {
        lastActiveSessionKey = sessionKey;
        if (typeof agentId === 'string' && agentId.length > 0) {
            sessionByAgentId.set(agentId, sessionKey);
        }
    }
    function forgetSession(sessionKey) {
        if (lastActiveSessionKey === sessionKey) {
            lastActiveSessionKey = undefined;
        }
        for (const [agentId, mapped] of sessionByAgentId) {
            if (mapped === sessionKey) {
                sessionByAgentId.delete(agentId);
            }
        }
    }
    function resolveAfterToolSessionKey(ctx) {
        // Primary: from context
        const direct = asNonEmptyString(ctx.sessionKey);
        if (direct && activeTraces.has(direct)) {
            return direct;
        }
        // Fallback 1: agentId → sessionKey map
        const agentId = asNonEmptyString(ctx.agentId);
        if (agentId) {
            const byAgent = sessionByAgentId.get(agentId);
            if (byAgent && activeTraces.has(byAgent)) {
                if (!warnedMissingAfterToolSessionKey) {
                    warnedMissingAfterToolSessionKey = true;
                    log.warn('mlflow: after_tool_call missing sessionKey; using agentId fallback');
                }
                return byAgent;
            }
        }
        // Fallback 2: single active trace
        if (activeTraces.size === 1) {
            if (!warnedMissingAfterToolSessionKey) {
                warnedMissingAfterToolSessionKey = true;
                log.warn('mlflow: after_tool_call missing sessionKey; using single-active-trace fallback');
            }
            return activeTraces.keys().next().value;
        }
        // Fallback 3: last active session
        if (lastActiveSessionKey && activeTraces.has(lastActiveSessionKey)) {
            if (!warnedMissingAfterToolSessionKey) {
                warnedMissingAfterToolSessionKey = true;
                log.warn('mlflow: after_tool_call missing sessionKey; using last-active-session fallback');
            }
            return lastActiveSessionKey;
        }
        return undefined;
    }
    function getOrCreateTrace(sessionKey, prompt) {
        const existing = activeTraces.get(sessionKey);
        if (existing) {
            activeTraces.delete(sessionKey);
            activeTraces.set(sessionKey, existing);
            existing.lastActivityMs = Date.now();
            return existing;
        }
        const cleanPrompt = sanitizeOpenClawText(prompt);
        const rootSpan = startSpan({
            name: 'openclaw_agent',
            inputs: {
                messages: [{ role: 'user', content: cleanPrompt }],
            },
            spanType: SpanType.AGENT,
        });
        rootSpan.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'openai');
        rootSpan.setAttribute(TraceMetadataKey.TRACE_SESSION, sessionKey);
        rootSpan.setAttribute(TraceMetadataKey.TRACE_USER, getCurrentUser());
        const trace = {
            rootSpan,
            pendingLlm: null,
            pendingTools: new Map(),
            pendingSubagents: new Map(),
            tokenUsage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, cost: 0 },
            costMeta: {},
            firstPrompt: cleanPrompt,
            lastResponse: '',
            agentEndData: null,
            lastActivityMs: Date.now(),
        };
        activeTraces.set(sessionKey, trace);
        evictOldest(activeTraces, MAX_ACTIVE_TRACES);
        return trace;
    }
    // Flush with exponential backoff retry
    async function flushWithRetry(reason) {
        const attempts = DEFAULT_FLUSH_RETRY_COUNT + 1;
        for (let attempt = 1; attempt <= attempts; attempt++) {
            try {
                await flushTraces();
                metrics.flushSuccesses += 1;
                return;
            }
            catch (err) {
                metrics.flushFailures += 1;
                log.warn(`mlflow: flush failed (${reason}) attempt ${attempt}/${attempts}: ${String(err)}`);
                if (attempt >= attempts) {
                    return;
                }
                metrics.flushRetries += 1;
                const delayMs = Math.min(DEFAULT_FLUSH_RETRY_BASE_DELAY_MS * 2 ** (attempt - 1), MAX_FLUSH_RETRY_DELAY_MS);
                await sleep(delayMs);
            }
        }
    }
    // =====================================================================
    // registerHooks — MUST be called during register(), not start().
    // OpenClaw only accepts api.on() subscriptions during the register phase.
    // Hooks guard on `initialized` so events before SDK init are silently skipped.
    // =====================================================================
    function registerHooks() {
        if (hooksRegistered) {
            return;
        }
        if (pluginConfig.enabled === false) {
            return;
        }
        const trackingUri = (typeof pluginConfig.trackingUri === 'string' ? pluginConfig.trackingUri : '') ||
            process.env.MLFLOW_TRACKING_URI;
        const experimentId = (typeof pluginConfig.experimentId === 'string' ? pluginConfig.experimentId : '') ||
            process.env.MLFLOW_EXPERIMENT_ID;
        if (!trackingUri || !experimentId) {
            return;
        }
        initError = null;
        try {
            const shared = acquireSharedSdk(trackingUri, experimentId);
            if (shared.trackingUri !== trackingUri || shared.experimentId !== experimentId) {
                log.warn(`mlflow: already exporting to ${shared.trackingUri} (experiment=${shared.experimentId}) in this process; ignoring ${trackingUri} (experiment=${experimentId})`);
            }
        }
        catch (err) {
            initError = err;
            log.warn(`mlflow: init failed: ${String(err)}`);
        }
        hooksRegistered = true;
        resolvedTrackingUri = trackingUri;
        resolvedExperimentId = experimentId;
        api.on('llm_input', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            rememberSession(sessionKey, ctx.agentId);
            const prompt = evt.prompt ?? '';
            const historyMessages = evt.historyMessages;
            const trace = getOrCreateTrace(sessionKey, prompt);
            const channelId = asNonEmptyString(ctx.channelId) ?? asNonEmptyString(ctx.messageProvider);
            if (channelId) {
                trace.channelId = channelId;
            }
            const trigger = asNonEmptyString(ctx.trigger);
            if (trigger) {
                trace.trigger = trigger;
            }
            if (trace.pendingLlm) {
                trace.pendingLlm.span.end();
            }
            const rawProvider = evt.provider;
            const provider = normalizeProvider(rawProvider) ?? rawProvider;
            const model = evt.model;
            const modelLabel = provider && model ? `${provider}/${model}` : model || 'unknown';
            if (model) {
                trace.model = model;
            }
            if (provider) {
                trace.provider = provider;
            }
            const messages = [];
            if (evt.systemPrompt) {
                messages.push({ role: 'system', content: evt.systemPrompt });
            }
            if (historyMessages?.length) {
                for (const msg of historyMessages) {
                    const m = msg;
                    if (!m.role) {
                        continue;
                    }
                    const role = m.role === 'toolResult' ? 'tool' : m.role;
                    const content = typeof m.content === 'string'
                        ? m.content
                        : Array.isArray(m.content)
                            ? m.content.map((p) => p.text ?? '').join('\n')
                            : m.content != null
                                ? JSON.stringify(m.content)
                                : '';
                    messages.push({ role, content });
                }
            }
            messages.push({ role: 'user', content: prompt });
            const llmInputs = sanitizeValue({
                messages,
                model: modelLabel,
            });
            const llmSpan = startSpan({
                name: 'llm_call',
                parent: trace.rootSpan,
                spanType: SpanType.LLM,
                inputs: llmInputs,
                attributes: {
                    ...(model ? { 'mlflow.llm.model': model } : {}),
                    ...(provider ? { 'mlflow.llm.provider': provider } : {}),
                },
            });
            llmSpan.setAttribute(SpanAttributeKey.MESSAGE_FORMAT, 'openai');
            trace.pendingLlm = { span: llmSpan, name: 'llm_call' };
        });
        api.on('llm_output', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            rememberSession(sessionKey, ctx.agentId);
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            trace.lastActivityMs = Date.now();
            const assistantTexts = evt.assistantTexts ?? [];
            const lastAssistant = evt.lastAssistant;
            const rawResponse = assistantTexts.length > 0 ? assistantTexts.join('\n') : evt.response || '';
            const response = sanitizeOpenClawText(rawResponse);
            trace.lastResponse = response;
            const rawProvider = evt.provider;
            if (rawProvider) {
                trace.provider = normalizeProvider(rawProvider) ?? rawProvider;
            }
            if (evt.model) {
                trace.model = evt.model;
            }
            if (trace.pendingLlm) {
                const evtUsage = evt.usage;
                const assistantUsage = lastAssistant?.usage;
                const usage = evtUsage ?? assistantUsage;
                if (usage && (usage.input || usage.output || usage.total || usage.totalTokens)) {
                    const inputTokens = usage.input ?? 0;
                    const outputTokens = usage.output ?? 0;
                    const cacheRead = usage.cacheRead ?? 0;
                    const cacheWrite = usage.cacheWrite ?? 0;
                    trace.pendingLlm.span.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
                        input_tokens: inputTokens,
                        output_tokens: outputTokens,
                        total_tokens: inputTokens + outputTokens + cacheRead + cacheWrite,
                        ...(cacheRead ? { cache_read_input_tokens: cacheRead } : {}),
                        ...(cacheWrite ? { cache_creation_input_tokens: cacheWrite } : {}),
                    });
                }
                trace.pendingLlm.span.setOutputs({
                    choices: [{ message: { role: 'assistant', content: response } }],
                });
                trace.pendingLlm.span.end();
                trace.pendingLlm = null;
            }
        });
        api.on('before_tool_call', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            rememberSession(sessionKey, ctx.agentId);
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            trace.lastActivityMs = Date.now();
            const toolName = evt.toolName;
            const toolCallId = evt.toolCallId;
            const key = toolKey(toolName, toolCallId);
            const toolSpan = startSpan({
                name: toolName,
                parent: trace.rootSpan,
                spanType: SpanType.TOOL,
                inputs: sanitizeValue(evt.params || {}),
                attributes: {
                    tool_name: toolName,
                    ...(toolCallId ? { tool_id: toolCallId } : {}),
                },
            });
            trace.pendingTools.set(key, { span: toolSpan, name: toolName });
        });
        api.on('after_tool_call', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = resolveAfterToolSessionKey(ctx);
            if (!sessionKey) {
                return;
            }
            rememberSession(sessionKey, ctx.agentId);
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            trace.lastActivityMs = Date.now();
            const toolName = evt.toolName;
            const toolCallId = evt.toolCallId;
            const key = toolKey(toolName, toolCallId);
            const pending = trace.pendingTools.get(key);
            if (pending) {
                if (evt.error) {
                    pending.span.setOutputs({ error: evt.error });
                }
                else {
                    pending.span.setOutputs({
                        result: evt.result || '',
                    });
                }
                pending.span.end();
                trace.pendingTools.delete(key);
            }
        });
        api.on('subagent_spawning', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            trace.lastActivityMs = Date.now();
            const agentId = evt.agentId;
            const label = evt.label;
            const subSpan = startSpan({
                name: `subagent_${label || agentId}`,
                parent: trace.rootSpan,
                spanType: SpanType.AGENT,
                inputs: {
                    agent_id: agentId,
                    ...(label ? { label } : {}),
                },
            });
            trace.pendingSubagents.set(agentId, { span: subSpan, name: agentId });
        });
        api.on('subagent_ended', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            trace.lastActivityMs = Date.now();
            const agentId = evt.agentId;
            const pending = trace.pendingSubagents.get(agentId);
            if (pending) {
                if (evt.error) {
                    pending.span.setOutputs({ error: evt.error });
                }
                else {
                    pending.span.setOutputs({
                        result: evt.result || '',
                    });
                }
                pending.span.end();
                trace.pendingSubagents.delete(agentId);
            }
        });
        api.on('agent_end', (event, agentCtx) => {
            const ctx = agentCtx;
            const evt = event;
            const sessionKey = ctx.sessionKey;
            if (!sessionKey) {
                return;
            }
            rememberSession(sessionKey, ctx.agentId);
            const trace = activeTraces.get(sessionKey);
            if (!trace) {
                return;
            }
            const channelId = asNonEmptyString(ctx.channelId) ?? asNonEmptyString(ctx.messageProvider);
            if (channelId && !trace.channelId) {
                trace.channelId = channelId;
            }
            const trigger = asNonEmptyString(ctx.trigger);
            if (trigger && !trace.trigger) {
                trace.trigger = trigger;
            }
            trace.agentEndData = {
                success: evt.success,
                error: evt.error,
                durationMs: evt.durationMs,
                messages: evt.messages,
            };
            const userId = ctx.userId;
            queueMicrotask(() => {
                void (async () => {
                    try {
                        const t = activeTraces.get(sessionKey);
                        if (t) {
                            activeTraces.delete(sessionKey);
                            forgetSession(sessionKey);
                            await finalizeTrace(sessionKey, t, userId, () => flushWithRetry('agent-end'));
                        }
                    }
                    catch {
                        // Silently ignore finalization errors
                    }
                })();
            });
        });
    }
    return {
        id: 'mlflow-tracing',
        registerHooks,
        start(ctx) {
            log = { info: ctx.logger.info.bind(ctx.logger), warn: ctx.logger.warn.bind(ctx.logger) };
            registerHooks();
            if (!hooksRegistered) {
                if (initError) {
                    ctx.logger.warn(`mlflow: tracing is disabled (init failed: ${String(initError)})`);
                }
                else {
                    ctx.logger.warn('mlflow: tracing is disabled (missing trackingUri/experimentId or explicitly disabled)');
                }
                return;
            }
            ctx.logger.info(`mlflow: exporting traces to ${resolvedTrackingUri} (experiment=${resolvedExperimentId})`);
            const unsubDiagnostics = onDiagnosticEvent((evt) => {
                if (evt.type !== 'model.usage') {
                    return;
                }
                const sessionKey = evt.sessionKey;
                if (!sessionKey) {
                    return;
                }
                const trace = activeTraces.get(sessionKey);
                if (!trace) {
                    return;
                }
                trace.lastActivityMs = Date.now();
                if (evt.usage) {
                    trace.tokenUsage.inputTokens += evt.usage.input || 0;
                    trace.tokenUsage.outputTokens += evt.usage.output || 0;
                    trace.tokenUsage.totalTokens += evt.usage.total || 0;
                }
                if (evt.costUsd != null) {
                    trace.costMeta.costUsd = (trace.costMeta.costUsd ?? 0) + evt.costUsd;
                }
                if (evt.context?.limit != null) {
                    trace.costMeta.contextLimit = evt.context.limit;
                }
                if (evt.context?.used != null) {
                    trace.costMeta.contextUsed = evt.context.used;
                }
                if (evt.model) {
                    trace.costMeta.model = evt.model;
                }
                if (evt.provider) {
                    trace.costMeta.provider = normalizeProvider(evt.provider) ?? evt.provider;
                }
            });
            const sweepInterval = setInterval(() => {
                const now = Date.now();
                for (const [key, trace] of activeTraces) {
                    if (now - trace.lastActivityMs > DEFAULT_STALE_TRACE_TIMEOUT_MS) {
                        log.warn(`mlflow: force-closing stale trace sessionKey=${key}`);
                        activeTraces.delete(key);
                        forgetSession(key);
                        trace.rootSpan.setStatus(SpanStatusCode.ERROR, 'Trace exceeded inactivity timeout');
                        finalizeTrace(key, trace, undefined, () => flushWithRetry('stale-sweep')).catch(() => undefined);
                    }
                }
            }, DEFAULT_STALE_SWEEP_INTERVAL_MS);
            cleanup = () => {
                unsubDiagnostics();
                clearInterval(sweepInterval);
            };
        },
        async stop() {
            cleanup?.();
            cleanup = null;
            for (const [sessionKey, trace] of activeTraces) {
                try {
                    await finalizeTrace(sessionKey, trace, undefined, () => flushWithRetry('shutdown'));
                }
                catch (err) {
                    log.warn(`mlflow: error finalizing trace ${sessionKey} during shutdown: ${String(err)}`);
                }
            }
            activeTraces.clear();
            sessionByAgentId.clear();
            lastActiveSessionKey = undefined;
            log.info(`mlflow: exporter metrics flushSuccesses=${metrics.flushSuccesses} flushFailures=${metrics.flushFailures} flushRetries=${metrics.flushRetries} spanErrors=${metrics.spanErrors}`);
        },
    };
}
