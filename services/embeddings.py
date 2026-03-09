from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from services.config import EmbeddingSettings


class BaseEmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    settings: EmbeddingSettings
    default_base_url: str | None = None
    name: str = "openai_compatible"
    dimensions: int = 0
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for provider-backed embeddings.") from exc

        if not self.settings.api_key:
            raise ValueError("EMBEDDING_API_KEY is required for the configured embedding provider.")

        base_url = self.settings.base_url or self.default_base_url
        client_kwargs: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "timeout": self.settings.timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if self.settings.extra_headers:
            client_kwargs["default_headers"] = self.settings.extra_headers

        self._client = OpenAI(**client_kwargs)
        self.dimensions = self.settings.dimensions or 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = self.settings.batch_size
        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_batch([text])
        if not vectors:
            raise RuntimeError("Embedding provider returned no vectors for query input.")
        return vectors[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "input": batch,
            **self.settings.extra_kwargs,
        }
        if self.settings.dimensions:
            kwargs["dimensions"] = self.settings.dimensions

        try:
            response = self._client.embeddings.create(**kwargs)
        except Exception:
            kwargs.pop("dimensions", None)
            response = self._client.embeddings.create(**kwargs)

        vectors = [list(item.embedding) for item in response.data]
        if vectors and not self.settings.dimensions:
            self.dimensions = len(vectors[0])
        return vectors


@dataclass(slots=True)
class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    name: str = "openai"


@dataclass(slots=True)
class DeepSeekEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    name: str = "deepseek"

    def __post_init__(self) -> None:
        self.default_base_url = self.default_base_url or "https://api.deepseek.com/v1"
        super().__post_init__()


def build_embedding_provider(settings: EmbeddingSettings) -> BaseEmbeddingProvider:
    provider = settings.provider
    if provider == "openai":
        return OpenAIEmbeddingProvider(settings=settings)
    if provider == "deepseek":
        return DeepSeekEmbeddingProvider(settings=settings)
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(settings=settings)
    raise ValueError(f"Unsupported embedding provider: {provider}")
