"""OpenAI-compatible client helpers.

Configure the client with OPENAI_API_KEY and, for compatible gateways,
OPENAI_BASE_URL. The module keeps the historical helper names so existing
experiment scripts continue to work.
"""
from __future__ import annotations

import base64
import functools
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import openai


cloudgpt_available_models = str


def _client_kwargs() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before calling an API model.")
    kwargs = {"api_key": api_key}
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    return kwargs


@functools.lru_cache(maxsize=1)
def get_openai_client(**_: Any) -> openai.OpenAI:
    return openai.OpenAI(**_client_kwargs())


@functools.lru_cache(maxsize=1)
def async_get_openai_client(**_: Any) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(**_client_kwargs())


def get_chat_completion(
    messages: Iterable[dict[str, Any]],
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs: Any,
):
    model_name = model or kwargs.pop("engine", None)
    if not model_name:
        raise ValueError("model name must be specified")
    return get_openai_client().chat.completions.create(
        messages=messages, model=model_name, stream=stream, **kwargs
    )


async def async_get_chat_completion(
    messages: Iterable[dict[str, Any]],
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs: Any,
):
    model_name = model or kwargs.pop("engine", None)
    if not model_name:
        raise ValueError("model name must be specified")
    client = async_get_openai_client()
    return await client.chat.completions.create(
        messages=messages, model=model_name, stream=stream, **kwargs
    )


def encode_image(image_path: str | os.PathLike[str]) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


__all__ = [
    "async_get_chat_completion",
    "async_get_openai_client",
    "cloudgpt_available_models",
    "encode_image",
    "get_chat_completion",
    "get_openai_client",
]
