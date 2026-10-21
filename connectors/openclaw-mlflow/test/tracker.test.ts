import assert from "node:assert/strict";
import { test } from "node:test";
import { toOtlpJson, type SpanRecord } from "../src/otlp.ts";
import { RunTracker } from "../src/tracker.ts";

function tracker(overrides = {}) {
  const exported: SpanRecord[][] = [];
  const t = new RunTracker({
    export: async (spans) => { exported.push(spans); return true; },
    captureContent: true, captureSystemPrompt: false, maxContentChars: 20_000, staleRunMs: 60_000,
    ...overrides,
  });
  return { t, exported };
}

const ctx = { runId: "run-1", sessionKey: "agent:main:main", sessionId: "s-1", agentId: "main", trigger: "user" };

function runOnce(t: RunTracker) {
  t.onLlmInput({
    runId: "run-1", sessionId: "s-1", provider: "ollama", model: "gemma4:e4b", systemPrompt: "SYSTEM",
    prompt: "How long do refunds take?",
    historyMessages: [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
    ],
    imagesCount: 0, tools: [{ name: "read" }],
  }, ctx);
  t.onModelCallStarted({ runId: "run-1", callId: "c1", provider: "ollama", model: "gemma4:e4b" }, ctx);
  t.onModelCallEnded({ runId: "run-1", callId: "c1", provider: "ollama", model: "gemma4:e4b", durationMs: 120, outcome: "completed" }, ctx);
  t.onBeforeToolCall({ toolName: "read", params: { path: "knowledge/billing.md" }, toolCallId: "t1" }, { ...ctx, toolName: "read" });
  t.onAfterToolCall({ toolName: "read", params: { path: "knowledge/billing.md" }, toolCallId: "t1", result: "Refunds: 14 days", durationMs: 3 }, { ...ctx, toolName: "read" });
  t.onModelCallStarted({ runId: "run-1", callId: "c2", provider: "ollama", model: "gemma4:e4b" }, ctx);
  t.onModelCallEnded({ runId: "run-1", callId: "c2", provider: "ollama", model: "gemma4:e4b", durationMs: 80, outcome: "completed" }, ctx);
  t.onLlmOutput({ runId: "run-1", sessionId: "s-1", provider: "ollama", model: "gemma4:e4b", assistantTexts: ["Refunds take 14 days."], usage: { input: 100, output: 20, cacheRead: 5 } }, ctx);
  return t.onAgentEnd({ runId: "run-1", messages: [], success: true, durationMs: 300 }, ctx);
}

test("one run becomes one trace: AGENT root, CHAT_MODEL and TOOL children", async () => {
  const { t, exported } = tracker();
  await runOnce(t);
  assert.equal(exported.length, 1);
  const [root, ...children] = exported[0];
  assert.equal(root.attributes["mlflow.spanType"], "AGENT");
  assert.equal(root.attributes["session.id"], "agent:main:main");
  assert.equal(root.attributes["mlflow.traceTag.mlflow.traceName"], "openclaw_agent");
  assert.deepEqual(children.map((s) => [s.name, s.attributes["mlflow.spanType"]]), [
    ["chat gemma4:e4b", "CHAT_MODEL"], ["read", "TOOL"], ["chat gemma4:e4b", "CHAT_MODEL"],
  ]);
  assert.ok(children.every((s) => s.traceId === root.traceId && s.parentSpanId === root.spanId && s.endNs));

  const inputs = JSON.parse(String(root.attributes["mlflow.spanInputs"]));
  assert.deepEqual(inputs.messages.map((m: any) => m.role), ["user", "assistant", "user"]);  // no system prompt by default
  assert.equal(JSON.parse(String(root.attributes["mlflow.spanOutputs"])).choices[0].message.content, "Refunds take 14 days.");
  assert.deepEqual(JSON.parse(String(root.attributes["mlflow.chat.tokenUsage"])), {
    input_tokens: 100, output_tokens: 20, cache_read_input_tokens: 5, cache_creation_input_tokens: 0, total_tokens: 120,
  });
  // first request gets the prompt, last request gets the reply
  assert.ok(children[0].attributes["mlflow.spanInputs"]);
  assert.ok(children[2].attributes["mlflow.spanOutputs"]);
  assert.equal(JSON.parse(String(children[1].attributes["mlflow.spanOutputs"])), "Refunds: 14 days");
  assert.equal(t.openRuns, 0);
});

test("captureContent=false records structure and usage but no text", async () => {
  const { t, exported } = tracker({ captureContent: false });
  await runOnce(t);
  for (const s of exported[0]) {
    assert.equal(s.attributes["mlflow.spanInputs"], undefined);
    assert.equal(s.attributes["mlflow.spanOutputs"], undefined);
  }
  assert.ok(exported[0][0].attributes["mlflow.chat.tokenUsage"]);
});

test("failed runs and failed tools are marked as errors; unfinished spans are closed", async () => {
  const { t, exported } = tracker();
  t.onBeforeToolCall({ toolName: "exec", params: { cmd: "ls" }, toolCallId: "t9" }, ctx);
  t.onModelCallStarted({ runId: "run-1", callId: "c1", model: "m" }, ctx);
  t.onAfterToolCall({ toolName: "exec", toolCallId: "t9", error: "denied" }, ctx);
  await t.onAgentEnd({ runId: "run-1", messages: [], success: false, error: "provider 401" }, ctx);
  const [root, model, tool] = [exported[0][0], exported[0][2], exported[0][1]];
  assert.equal(root.status?.code, 2);
  assert.equal(root.status?.message, "provider 401");
  assert.equal(tool.status?.code, 2);
  assert.equal(model.attributes["openclaw.unfinished"], true);
});

test("agent_end without runId falls back to the session's latest run", async () => {
  const { t, exported } = tracker();
  t.onLlmInput({ runId: "run-1", prompt: "q", historyMessages: [] }, ctx);
  await t.onAgentEnd({ messages: [{ role: "assistant", content: "a" }], success: true }, { sessionKey: ctx.sessionKey });
  assert.equal(exported.length, 1);
});

test("stale runs are swept", async () => {
  const { t, exported } = tracker({ staleRunMs: -1 });
  t.onLlmInput({ runId: "run-1", prompt: "q", historyMessages: [] }, ctx);
  await t.sweep();
  assert.equal(exported[0][0].status?.code, 2);
});

test("long content is truncated", async () => {
  const { t, exported } = tracker({ maxContentChars: 300 });
  t.onLlmInput({ runId: "run-1", prompt: "x".repeat(5000), historyMessages: [] }, ctx);
  await t.onAgentEnd({ runId: "run-1", messages: [], success: true }, ctx);
  assert.ok(String(exported[0][0].attributes["mlflow.spanInputs"]).length < 400);
});

test("OTLP JSON encoding", () => {
  const body = toOtlpJson([{ traceId: "a".repeat(32), spanId: "b".repeat(16), name: "s", startNs: 1n, endNs: 2n,
    attributes: { s: "x", i: 3, d: 0.5, b: true } }], { "service.name": "openclaw" }, { name: "t", version: "0" });
  const span = body.resourceSpans[0].scopeSpans[0].spans[0];
  assert.deepEqual(span.attributes.map((a) => a.value), [{ stringValue: "x" }, { intValue: "3" }, { doubleValue: 0.5 }, { boolValue: true }]);
  assert.equal(span.startTimeUnixNano, "1");
  assert.equal("parentSpanId" in span, false);
});

test("real OpenClaw order: llm_output arrives just after agent_end and still lands in the trace", async () => {
  const { t, exported } = tracker({ endGraceMs: 1000 });
  t.onLlmInput({ runId: "run-1", prompt: "q", historyMessages: [] }, ctx);
  const ended = t.onAgentEnd({ runId: "run-1", messages: [], success: true }, ctx);
  assert.equal(exported.length, 0);   // waiting for llm_output
  t.onLlmOutput({ runId: "run-1", assistantTexts: ["the answer"], usage: { input: 7, output: 3 } }, ctx);
  await ended;
  assert.equal(exported.length, 1);
  const root = exported[0][0];
  assert.equal(JSON.parse(String(root.attributes["mlflow.spanOutputs"])).choices[0].message.content, "the answer");
  assert.equal(JSON.parse(String(root.attributes["mlflow.chat.tokenUsage"])).total_tokens, 10);
  // a straggler for a finished run doesn't start a new trace
  t.onLlmOutput({ runId: "run-1", assistantTexts: ["late"] }, ctx);
  t.onModelCallEnded({ runId: "run-1", callId: "x" }, ctx);
  assert.equal(t.openRuns, 0);
});

test("without llm_output the run is exported after the grace period", async () => {
  const { t, exported } = tracker({ endGraceMs: 20 });
  t.onLlmInput({ runId: "run-1", prompt: "q", historyMessages: [] }, ctx);
  await t.onAgentEnd({ runId: "run-1", messages: [{ role: "assistant", content: [{ type: "text", text: "from messages" }] }], success: true }, ctx);
  assert.equal(exported.length, 1);
  assert.equal(JSON.parse(String(exported[0][0].attributes["mlflow.spanOutputs"])).choices[0].message.content, "from messages");
});

test("drainSync closes ended and unfinished runs at exit", () => {
  const { t } = tracker({ endGraceMs: 10_000 });
  t.onLlmInput({ runId: "run-1", prompt: "q", historyMessages: [] }, ctx);
  void t.onAgentEnd({ runId: "run-1", messages: [], success: true }, ctx);
  t.onLlmInput({ runId: "run-2", prompt: "q2", historyMessages: [] }, { ...ctx, runId: "run-2", sessionKey: "other" });
  const spans = t.drainSync();
  assert.equal(spans.length, 2);
  assert.deepEqual(spans.map((s) => s.status?.code), [1, 2]);
  assert.equal(t.openRuns, 0);
});
