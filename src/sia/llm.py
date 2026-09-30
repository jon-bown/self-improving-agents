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
# Override either with SIA_AGENT_MODEL / MLFLOW_GENAI_JUDGE_DEFAULT_MODEL.
PRESETS: dict[str, tuple[str, str]] = {
    #            agent (small, cheap: failures easy to see)   judge / analyst / optimizer (strong)
    "anthropic": ("anthropic:/claude-haiku-4-5",            "anthropic:/claude-sonnet-5-5"),
    "openai":    ("openai:/gpt-5.4-mini",                    "openai:/gpt-5.5"),
    "gemini":    ("gemini:/gemini-3.5-flash",                "gemini:/gemini-3.1-pro-preview"),
    "mistral":   ("mistral:/mistral-small-latest",           "mistral:/mistral-large-latest"),
    "groq":      ("groq:/llama-3.1-8b-instant",              "groq:/llama-3.3-70b-versatile"),
    "ollama":    ("ollama:/gpt-oss:20b",                     "ollama:/gpt-oss:20b"),
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
    if provider not in PROVIDER_KEYS:
        return True  # unknown provider: assume the user configured it; LiteLLM will say otherwise
    key = PROVIDER_KEYS[provider]
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return key is None or bool(os.getenv(key))


def missing_key_hint(*uris: str) -> str:
    missing = sorted({PROVIDER_KEYS.get(split(u)[0]) or "" for u in uris if not has_credentials(u)} - {""})
    return f"set {', '.join(missing)}" if missing else ""


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
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    return litellm.completion(**kwargs).choices[0].message


def structured(uri: str, system: str, user: str, schema_tool: dict, max_tokens: int = 8000) -> dict:
    """Get a JSON object matching schema_tool's parameters, from any provider.

    Forces a call to the tool where the provider supports tool_choice; otherwise falls back to
    asking for the JSON in the reply and parsing it.
    """
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
