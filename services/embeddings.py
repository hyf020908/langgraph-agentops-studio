from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from services.config import EmbeddingSettings
from services.resilience import ProviderGuard


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
    _guard: ProviderGuard = field(init=False, default_factory=ProviderGuard)

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
            "max_retries": 2,
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
        with self._guard.call():
            return self._request_batch(batch)

    def _request_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "input": batch,
            **self.settings.extra_kwargs,
        }
        if self.settings.dimensions:
            kwargs["dimensions"] = self.settings.dimensions

        # Do not retry by removing dimensions: this masks auth/outage errors and
        # may silently change the collection's vector dimension.
        response = self._client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(batch))):
            raise RuntimeError("Embedding response indexes do not match the input batch.")
        vectors = [list(item.embedding) for item in ordered]
        if any(not vector or not all(math.isfinite(value) for value in vector) or (self.settings.dimensions and len(vector) != self.settings.dimensions) for vector in vectors):
            raise RuntimeError("Embedding response dimensions do not match configuration.")
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
