"""
cag_service/backends.py
=======================
Pluggable LLM backend adapters for the CAG engine.

Each backend implements BaseLLMBackend with two methods:
  - complete(messages, **kwargs) -> str
  - stream(messages, **kwargs)  -> Iterator[str]

Supported backends
------------------
  OllamaBackend     – local models via Ollama
  OpenAIBackend     – OpenAI API / Azure OpenAI
  AnthropicBackend  – Claude models via Anthropic API
  HuggingFaceBackend– HuggingFace Inference API
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLLMBackend(ABC):
    """Abstract base class all backends must implement."""

    model_name: str = "unknown"

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Run a blocking completion and return the full response string."""

    @abstractmethod
    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Yield response text chunks as they arrive."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaBackend(BaseLLMBackend):
    """
    Local LLM via Ollama.

    Install: pip install ollama
    Run:     ollama serve && ollama pull llama3.3

    Example
    -------
    backend = OllamaBackend(model="llama3.3")
    """

    def __init__(
        self,
        model: str = "llama3.3",
        host: str | None = None,
        options: dict | None = None,
    ):
        try:
            import ollama  # noqa: F401
        except ImportError as e:
            raise ImportError("Install ollama: pip install ollama") from e

        import ollama as _ollama

        self.model_name = model
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._options = options or {}
        self._client = _ollama.Client(host=self._host)

    def complete(self, messages: list[dict], **kwargs) -> str:
        opts = {**self._options, **kwargs}
        resp = self._client.chat(model=self.model_name, messages=messages, options=opts)
        return resp.message.content or ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        opts = {**self._options, **kwargs}
        for chunk in self._client.chat(
            model=self.model_name, messages=messages, stream=True, options=opts
        ):
            if chunk.message and chunk.message.content:
                yield chunk.message.content


# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI
# ---------------------------------------------------------------------------

class OpenAIBackend(BaseLLMBackend):
    """
    OpenAI or Azure OpenAI backend.

    Install: pip install openai
    Env vars: OPENAI_API_KEY, OPENAI_BASE_URL (optional for Azure)

    Example
    -------
    backend = OpenAIBackend(model="gpt-4o")
    # Azure:
    backend = OpenAIBackend(
        model="gpt-4o",
        api_key="...",
        base_url="https://<resource>.openai.azure.com/",
        api_version="2024-02-01",
    )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
    ):
        try:
            from openai import OpenAI, AzureOpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError("Install openai: pip install openai") from e

        from openai import OpenAI, AzureOpenAI

        self.model_name = model
        key = api_key or os.getenv("OPENAI_API_KEY", "")

        if api_version:  # Azure
            self._client = AzureOpenAI(
                api_key=key,
                azure_endpoint=base_url or os.getenv("OPENAI_BASE_URL", ""),
                api_version=api_version,
            )
        else:
            self._client = OpenAI(
                api_key=key,
                base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                organization=organization,
            )

    def complete(self, messages: list[dict], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        with self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicBackend(BaseLLMBackend):
    """
    Anthropic Claude backend.

    Install: pip install anthropic
    Env vars: ANTHROPIC_API_KEY

    Example
    -------
    backend = AnthropicBackend(model="claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError("Install anthropic: pip install anthropic") from e

        import anthropic as _anthropic

        self.model_name = model
        self._max_tokens = max_tokens
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "")
        )

    def _split_system(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Anthropic API requires system prompt separate from messages."""
        system = ""
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(m)
        return system, rest

    def complete(self, messages: list[dict], **kwargs) -> str:
        system, msgs = self._split_system(messages)
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            system=system,
            messages=msgs,
            **kwargs,
        )
        return resp.content[0].text if resp.content else ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        system, msgs = self._split_system(messages)
        with self._client.messages.stream(
            model=self.model_name,
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            system=system,
            messages=msgs,
            **kwargs,
        ) as stream:
            yield from stream.text_stream


# ---------------------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------------------

class HuggingFaceBackend(BaseLLMBackend):
    """
    HuggingFace Inference API backend.

    Install: pip install huggingface_hub
    Env vars: HF_TOKEN

    Example
    -------
    backend = HuggingFaceBackend(model="mistralai/Mistral-7B-Instruct-v0.3")
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_key: str | None = None,
    ):
        try:
            from huggingface_hub import InferenceClient  # noqa: F401
        except ImportError as e:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub") from e

        from huggingface_hub import InferenceClient

        self.model_name = model
        self._client = InferenceClient(
            model=model,
            token=api_key or os.getenv("HF_TOKEN", ""),
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        resp = self._client.chat_completion(messages=messages, **kwargs)
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        for chunk in self._client.chat_completion(
            messages=messages, stream=True, **kwargs
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

BACKEND_MAP = {
    "ollama":       OllamaBackend,
    "openai":       OpenAIBackend,
    "anthropic":    AnthropicBackend,
    "huggingface":  HuggingFaceBackend,
}


def create_backend(provider: str, model: str, **kwargs) -> BaseLLMBackend:
    """
    Factory function to instantiate a backend by name.

    Example
    -------
    backend = create_backend("ollama", "llama3.3")
    backend = create_backend("anthropic", "claude-sonnet-4-20250514")
    """
    cls = BACKEND_MAP.get(provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {list(BACKEND_MAP.keys())}"
        )
    return cls(model=model, **kwargs)
