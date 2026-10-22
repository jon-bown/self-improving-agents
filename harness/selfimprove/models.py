"""Model access for the harness's own LLM calls (analyst, reviewer agent, litellm agent).

Models are named in MLflow's URI format, "<provider>:/<model>", the same strings MLflow's judges,
optimizers and judge alignment take. Our own calls go through LiteLLM so any provider works.
"""

from __future__ import annotations

import json
import os
import re

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
    "databricks": "DATABRICKS_TOKEN",
    "typesafe": "TYPESAFE_API_KEY",     # Jev decision models (judges only)
    "ollama": None,
}
_LITELLM_PREFIX = {"ollama": "ollama_chat"}          # ollama_chat supports tool calling
_OPENCLAW_PREFIX = {"gemini": "google", "together_ai": "together"}


def split(uri: str) -> tuple[str, str]:
    """'anthropic:/claude-opus-5-5' -> ('anthropic', 'claude-opus-5-5'). Also accepts 'a/b'."""
    if ":/" in uri:
        provider, model = uri.split(":/", 1)
    elif "/" in uri:
        provider, model = uri.split("/", 1)
    else:
        raise ValueError(f"Model '{uri}' needs a provider, e.g. 'anthropic:/{uri}'")
    return provider.strip().lower(), model.strip()


def litellm_model(uri: str) -> str:
    provider, model = split(uri)
    return f"{_LITELLM_PREFIX.get(provider, provider)}/{model}"


def openclaw_model(uri: str) -> str:
    provider, model = split(uri)
    return f"{_OPENCLAW_PREFIX.get(provider, provider)}/{model}"


def has_credentials(uri: str) -> bool:
    try:
        provider, _ = split(uri)
    except ValueError:
        return False
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if provider not in PROVIDER_KEYS:
        return True  # unknown provider: assume it is configured; LiteLLM will say otherwise
    key = PROVIDER_KEYS[provider]
    return key is None or bool(os.getenv(key))


def tool(name: str, description: str, parameters: dict) -> dict:
    """A tool in OpenAI's schema; LiteLLM adapts it for each provider."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def chat(uri: str, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4000):
    """One completion. Returns the LiteLLM message (OpenAI shape: .content, .tool_calls)."""
    import litellm

    kwargs = {"model": litellm_model(uri), "messages": messages, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return litellm.completion(**kwargs).choices[0].message


def structured(uri: str, system: str, user: str, schema_tool: dict, max_tokens: int = 16000) -> dict:
    """A JSON object matching schema_tool's parameters, from any provider.

    The tool is offered with tool_choice="auto" and named in the prompt. Several current models
    (Claude Opus 5.5, Sonnet 5.5) reject forced tool choice, so we never force it. If the model
    answers in text instead, the JSON is parsed out of the reply.
    """
    name = schema_tool["function"]["name"]
    schema = json.dumps(schema_tool["function"]["parameters"])
    messages = [
        {"role": "system", "content": f"{system}\n\nSubmit your answer by calling the `{name}` tool exactly once."},
        {"role": "user", "content": user},
    ]
    msg = chat(uri, messages, tools=[schema_tool], max_tokens=max_tokens)
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments)
    text = msg.content or ""
    if not re.search(r"\{.*\}", text, re.DOTALL):
        messages.append({"role": "assistant", "content": text or "(no answer)"})
        messages.append({"role": "user", "content": f"Reply with only a JSON object matching this schema:\n{schema}"})
        text = chat(uri, messages, max_tokens=max_tokens).content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"{uri} returned no JSON: {text[:300]}")
    return json.loads(match.group(0))
