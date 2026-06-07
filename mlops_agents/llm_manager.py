"""
mlops_agents/llm_manager.py

Central LLM factory.  Set LLM_PROVIDER in your .env to switch backends:

    LLM_PROVIDER=ollama   (default)
        OLLAMA_MODEL      — model tag,   default "llama3.2:1b"
        OLLAMA_BASE_URL   — server URL,  default "http://localhost:11434"

    LLM_PROVIDER=google
        GOOGLE_MODEL      — model name,  default "gemini-2.0-flash"
        GOOGLE_API_KEY    — required
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(temperature: float = 0, **kwargs: Any) -> BaseChatModel:
    """Return a configured chat model for the active provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "google":
        return _google_llm(temperature=temperature, **kwargs)
    if provider == "ollama":
        return _ollama_llm(temperature=temperature, **kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported values: 'ollama', 'google'."
    )


def _ollama_llm(temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        **kwargs,
    )


def _google_llm(temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is required when LLM_PROVIDER=google."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        google_api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
