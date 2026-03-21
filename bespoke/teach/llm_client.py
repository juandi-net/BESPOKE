"""Unified LLM client using OpenAI SDK with configurable provider."""

from openai import OpenAI
from typing import Optional
import json

from bespoke.config import config


def get_client() -> OpenAI:
    """Get an OpenAI-compatible client pointed at the configured provider."""
    return OpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )


def chat(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    response_format: Optional[dict] = None,
) -> str:
    """Send a chat completion request. Returns the assistant's response text."""
    client = get_client()

    kwargs = {
        "model": model or config.llm.model_stage_2a,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _parse_json_response(text: str | None) -> dict | list:
    """Parse an LLM response as JSON. Handles markdown fences, None, empty."""
    if text is None:
        raise ValueError("LLM returned None (likely content filter)")

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text:
        raise ValueError("LLM returned empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned non-JSON: {text[:200]}")


def chat_json(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict | list:
    """Send a chat request expecting JSON output. Returns parsed dict or list."""
    text = chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_json_response(text)
