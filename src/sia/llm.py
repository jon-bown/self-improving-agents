"""Provider-agnostic model access.

Every model in this repo is named with MLflow's model URI format, "<provider>:/<model>":

    anthropic:/claude-haiku-4-5      openai:/gpt-5.4-mini      gemini:/gemini-3.5-flash
    mistral:/mistral-small-latest    groq:/llama-3.3-70b-versatile    ollama:/gpt-oss:20b

MLflow judges, GEPA, and judge alignment take these URIs directly. Our own calls (the direct
agent runtime and the trace analyst) go through LiteLLM, which speaks OpenAI-style chat + tool
calling to all of these providers. OpenClaw uses "<provider>/<model>" with a few different
provider names; openclaw_model() translates.
"""

from __future__ import annotations

import json
import os
import re

# provider -> (env var holding the API key, or None if no key is needed)
PROVIDER_KEYS: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "ollama": None,
}

# LiteLLM prefixes that differ from MLflow's provider names
_LITELLM_PREFIX = {"ollama": "ollama_chat"}      # ollama_chat supports tool calling
# OpenClaw provider ids that differ from MLflow's provider names
_OPENCLAW_PREFIX = {"gemini": "google", "together_ai": "together"}

# One-line provider switch: SIA_PROVIDER=openai picks sensible agent + judge models.
# The default is anthropic (Claude). ollama runs everything locally with no API key.
# Override either with SIA_AGENT_MODEL / MLFLOW_GENAI_JUDGE_DEFAULT_MODEL.
PRESETS: dict[str, tuple[str, str]] = {
    #            agent (small, cheap: failures easy to see)   judge / analyst / optimizer (strong)
    # Default. Haiku 5.5 as the support agent (small, so the seeded failures show); Opus 5.5
    # for judging, analysis, and optimization.
    "anthropic": ("anthropic:/claude-haiku-5-5",            "anthropic:/claude-opus-5-5"),
    "openai":    ("openai:/gpt-5.4-mini",                    "openai:/gpt-5.5"),
    "gemini":    ("gemini:/gemini-3.5-flash",                "gemini:/gemini-3.1-pro-preview"),
    "mistral":   ("mistral:/mistral-small-latest",           "mistral:/mistral-large-latest"),
    "groq":      ("groq:/llama-3.1-8b-instant",              "groq:/llama-3.3-70b-versatile"),
    # One local model for both roles so a 16 GB laptop can run the tutorial.
    "ollama":    ("ollama:/gemma4:e4b",                      "ollama:/gemma4:e4b"),
}

# Embedding models for MemAlign judge alignment (Anthropic and Groq have none; SIMBA is used then).
EMBEDDINGS = {
    "openai": "openai:/text-embedding-3-small",
    "gemini": "gemini:/gemini-embedding-001",
    "mistral": "mistral:/mistral-embed",
}


def split(uri: str) -> tuple[str, str]:
    """'openai:/gpt-5.5' -> ('openai', 'gpt-5.5'). Also accepts 'openai/gpt-5.5'."""
    if ":/" in uri:
        provider, model = uri.split(":/", 1)
    elif "/" in uri:
        provider, model = uri.split("/", 1)
    else:
        raise ValueError(f"Model '{uri}' needs a provider, e.g. 'openai:/{uri}' or 'anthropic:/{uri}'")
    return provider.strip().lower(), model.strip()


def normalize(uri: str) -> str:
    provider, model = split(uri)
    return f"{provider}:/{model}"


def litellm_model(uri: str) -> str:
    provider, model = split(uri)
    return f"{_LITELLM_PREFIX.get(provider, provider)}/{model}"


def openclaw_model(uri: str) -> str:
    provider, model = split(uri)
    return f"{_OPENCLAW_PREFIX.get(provider, provider)}/{model}"


def has_credentials(uri: str) -> bool:
    provider, _ = split(uri)
    if provider == "ollama":
        return ollama_ready(uri)
    if provider not in PROVIDER_KEYS:
        return True  # unknown provider: assume the user configured it; LiteLLM will say otherwise
    key = PROVIDER_KEYS[provider]
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return key is None or bool(os.getenv(key))


def missing_key_hint(*uris: str) -> str:
    hints = [ollama_hint(u) for u in uris if split(u)[0] == "ollama" and not ollama_ready(u)]
    missing = sorted({PROVIDER_KEYS.get(split(u)[0]) or "" for u in uris if not has_credentials(u)} - {""})
    if missing:
        hints.append(f"set {', '.join(missing)}")
    return "; ".join(dict.fromkeys(hints))


# --------------------------------------------------------------------------------------------
# Ollama (local): no key, but the server must be running and the model pulled
# --------------------------------------------------------------------------------------------
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
# Ollama's default context is small; OpenClaw's system prompt + AGENTS.md + a knowledge file need more.
OLLAMA_NUM_CTX = int(os.getenv("SIA_OLLAMA_NUM_CTX", "32768"))


def ollama_models() -> set[str] | None:
    """Names of locally pulled models, or None if the server is unreachable."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except Exception:
        return None
    return names | {n.removesuffix(":latest") for n in names}


def ollama_ready(uri: str) -> bool:
    models = ollama_models()
    return models is not None and split(uri)[1] in models


def ollama_hint(uri: str) -> str:
    model = split(uri)[1]
    if ollama_models() is None:
        return f"start Ollama (ollama serve) at {OLLAMA_BASE}, then: ollama pull {model}"
    return f"ollama pull {model}"


def ollama_ctx_variant(model: str, num_ctx: int) -> str:
    """Name of a local alias of `model` with a larger context window baked in, e.g.
    gemma4:e4b -> gemma4-e4b-64k. Needed by runtimes that talk to Ollama's OpenAI-compatible
    /v1 endpoint, which ignores per-request num_ctx (Ollama's default window is only 4K)."""
    return f"{model.replace(':', '-').replace('/', '-')}-{num_ctx // 1024}k"


def ensure_ollama_ctx_variant(model: str, num_ctx: int) -> str:
    """Create the variant if missing (metadata only: it shares the base model's weights).
    Remove with: ollama rm <variant>."""
    import urllib.request

    name = ollama_ctx_variant(model, num_ctx)
    models = ollama_models() or set()
    if name in models:
        return name
    if model not in models:
        raise RuntimeError(f"{ollama_hint('ollama:/' + model)} (needed to build {name})")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/create", method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": name, "from": model, "parameters": {"num_ctx": num_ctx}, "stream": False}).encode(),
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        status = json.load(r).get("status")
    if status != "success":
        raise RuntimeError(f"ollama create {name} failed: {status}")
    return name


# --------------------------------------------------------------------------------------------
# Hermes Agent: every provider through its OpenAI-compatible endpoint ("custom" provider)
# --------------------------------------------------------------------------------------------
OPENAI_COMPAT_BASE = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together_ai": "https://api.together.xyz/v1",
}
HERMES_NUM_CTX = 65536  # Hermes refuses tool use below 64K tokens of context


def hermes_endpoint(uri: str) -> dict:
    """AIAgent(model=..., provider=..., base_url=..., api_key=...) for a provider:/model URI.

    Claude uses Hermes' native Anthropic provider (the Anthropic SDK). Ollama and the rest go
    through Hermes' "custom" provider and each provider's OpenAI-compatible endpoint."""
    provider, model = split(uri)
    if provider == "anthropic":
        return {"model": model, "provider": "anthropic", "base_url": None,
                "api_key": os.getenv("ANTHROPIC_API_KEY", "")}
    if provider == "ollama":
        return {"model": ollama_ctx_variant(model, HERMES_NUM_CTX), "provider": "custom",
                "base_url": f"{OLLAMA_BASE}/v1", "api_key": "ollama"}
    if provider not in OPENAI_COMPAT_BASE:
        raise ValueError(f"No OpenAI-compatible endpoint known for provider '{provider}' (Hermes backend)")
    key_env = PROVIDER_KEYS.get(provider)
    key = os.getenv(key_env or "", "") or (os.getenv("GOOGLE_API_KEY", "") if provider == "gemini" else "")
    return {"model": model, "provider": "custom", "base_url": OPENAI_COMPAT_BASE[provider], "api_key": key}


# --------------------------------------------------------------------------------------------
# Chat + tools (OpenAI message format, any provider)
# --------------------------------------------------------------------------------------------
def tool(name: str, description: str, parameters: dict) -> dict:
    """Define a tool once, in OpenAI's schema; LiteLLM adapts it for each provider."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def chat(uri: str, messages: list[dict], tools: list[dict] | None = None, tool_choice=None,
         max_tokens: int = 1500):
    """One completion. Returns the LiteLLM message (OpenAI shape: .content, .tool_calls)."""
    import litellm

    kwargs = {"model": litellm_model(uri), "messages": messages, "max_tokens": max_tokens}
    if split(uri)[0] == "ollama":
        kwargs |= {"api_base": OLLAMA_BASE, "num_ctx": OLLAMA_NUM_CTX}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    return litellm.completion(**kwargs).choices[0].message


def structured(uri: str, system: str, user: str, schema_tool: dict, max_tokens: int = 8000) -> dict:
    """Get a JSON object matching schema_tool's parameters, from any provider.

    Claude: the Anthropic SDK with structured outputs (see claude_structured). Others: force a
    call to the tool where the provider supports tool_choice; otherwise ask for the JSON in the
    reply and parse it.
    """
    if split(uri)[0] == "anthropic":
        return claude_structured(split(uri)[1], system, user, schema_tool["function"]["parameters"])
    name = schema_tool["function"]["name"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        msg = chat(uri, messages, tools=[schema_tool],
                   tool_choice={"type": "function", "function": {"name": name}}, max_tokens=max_tokens)
        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments)
        text = msg.content or ""
    except Exception:
        schema = json.dumps(schema_tool["function"]["parameters"])
        messages[-1]["content"] += f"\n\nReply with only a JSON object matching this schema:\n{schema}"
        text = chat(uri, messages, max_tokens=max_tokens).content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"{uri} returned no JSON: {text[:300]}")
    return json.loads(match.group(0))


# --------------------------------------------------------------------------------------------
# Claude, through the Anthropic SDK
# --------------------------------------------------------------------------------------------
# Current Claude models reject forced tool_choice, disabled thinking, and non-default sampling,
# so Claude calls go through the official SDK with structured outputs instead of LiteLLM.
CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")


def claude_client():
    """The Anthropic client. Credentials resolve from the environment (ANTHROPIC_API_KEY, ...)."""
    import anthropic

    return anthropic.Anthropic()


def strict_schema(schema: dict) -> dict:
    """JSON schema for structured outputs: every object closed and every property required."""
    if not isinstance(schema, dict):
        return schema
    out = {k: strict_schema(v) if isinstance(v, dict) else
           ([strict_schema(i) for i in v] if isinstance(v, list) else v) for k, v in schema.items()}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    return out


def claude_structured(model: str, system: str, user: str, schema: dict, effort: str = "high") -> dict:
    """One structured-output request. Streams (long input and output) and, on models that
    support it, opts into server-side refusal fallbacks."""
    client = claude_client()
    params = dict(
        model=model, max_tokens=64000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": strict_schema(schema)}},
    )
    if model in CLAUDE_FALLBACK_MODELS:
        # Server-side refusal fallback; SDK 1.9 has no named parameter for it yet, so extra_body.
        stream = client.beta.messages.stream(**params, betas=["server-side-fallback-2026-07-01"],
                                             extra_body={"fallbacks": "default"})
    else:
        stream = client.messages.stream(**params)
    with stream as s:
        response = s.get_final_message()
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(f"{model} declined the request ({getattr(details, 'category', None)})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} ran out of output tokens before finishing the JSON")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
