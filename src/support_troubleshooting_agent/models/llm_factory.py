"""Centralized factory for selecting and validating the active chat model.

OpenAI is selected when ``OPENAI_API_KEY`` is configured. Otherwise the factory
validates the local Ollama service and returns a ``ChatOllama`` instance.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

try:
    from ollama import Client
except ImportError:  # pragma: no cover - dependency guard
    Client = None


def _get_openai_model() -> str:
    """Return the configured OpenAI model name."""

    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _get_ollama_model() -> str:
    """Return the configured Ollama model name."""

    return os.getenv("OLLAMA_MODEL", "llama3.2")


def get_model_configuration() -> tuple[str, str]:
    """Return the provider and model selected by the current environment."""

    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI", _get_openai_model()
    return "Ollama", _get_ollama_model()


def get_chat_model() -> BaseChatModel:
    """Return the active chat model based on the available environment configuration.

    Prefer OpenAI when an API key is present. Otherwise, select the local Ollama
    model and fail with a clear message if the Ollama service is unavailable.
    """

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return ChatOpenAI(
            model=_get_openai_model(),
            api_key=openai_api_key,
            temperature=0,
            max_retries=2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Ollama model selected, but langchain-ollama is not installed.") from exc

    if Client is None:
        raise RuntimeError(
            "Ollama is selected but the Ollama Python client is not available. "
            "Install the Ollama dependency and ensure the service is running."
        )

    try:
        Client().list()
    except Exception as exc:  # pragma: no cover - runtime validation path
        raise RuntimeError(
            "Ollama is selected but the Ollama service is unavailable. "
            "Start the Ollama service and ensure the model is pulled."
        ) from exc

    return ChatOllama(model=_get_ollama_model(), temperature=0)


__all__ = ["get_chat_model", "get_model_configuration"]
