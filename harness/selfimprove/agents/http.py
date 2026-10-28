"""SI_AGENT=http: your agent is a service.

    SI_AGENT_URL=http://127.0.0.1:8000/answer

The harness POSTs JSON and reads the answer back:

    request   {"input": "<question>", "instructions": "<instructions text>"}
    response  {"output": "<answer>"}   (or "text" / "answer" / "response", or a plain-text body)

The service must apply the instructions it receives, so that candidates can be evaluated
without deploying them. Send a W3C `traceparent` header onward if your service traces with MLflow
too; the harness includes one for the current span.
"""

from __future__ import annotations

import json
import os
import urllib.request

import mlflow

from selfimprove import config


def call_http(question, instructions, workspace):
    s = config.settings
    body = json.dumps({"input": question, "instructions": instructions}).encode()
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("SI_AGENT_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    span = mlflow.get_current_active_span()
    if span is not None:
        headers["traceparent"] = f"00-{span.trace_id.removeprefix('tr-')[:32]:0>32}-{span.span_id[:16]:0>16}-01"
    req = urllib.request.Request(s.agent_url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=s.agent_timeout) as resp:
        raw = resp.read().decode()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, str):
        return payload
    for key in ("output", "text", "answer", "response"):
        if isinstance(payload.get(key), str):
            return payload[key]
    raise ValueError(f"{s.agent_url} returned no output/text/answer field: {raw[:300]}")
