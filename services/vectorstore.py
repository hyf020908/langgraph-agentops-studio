from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from services.config import VectorDBSettings


class BaseVectorStore(Protocol):
    provider_name: str

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        ...

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        ...

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        ...


@dataclass(slots=True)
class InMemoryVectorStore:
    settings: VectorDBSettings
    provider_name: str = "memory"
    _vector_size: int = field(init=False, default=0)
    _records: dict[str, tuple[list[float], dict[str, Any]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.provider_name = "memory"

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        if recreate:
            self._records.clear()
        if self._vector_size and self._vector_size != vector_size:
            raise ValueError(
                f"Embedding size mismatch for memory vector store: existing={self._vector_size}, incoming={vector_size}"
            )
        self._vector_size = vector_size

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        for record_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            self._records[record_id] = (vector, payload)

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        if not self._records:
            return []
        ranked: list[tuple[str, float, dict[str, Any]]] = []
        for record_id, (vector, payload) in self._records.items():
            score = _cosine_similarity(query_vector, vector)
            if score_threshold is not None and score < score_threshold:
                continue
            ranked.append((record_id, score, payload))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                "id": record_id,
                "score": score,
                "payload": payload,
            }
            for record_id, score, payload in ranked[:top_k]
        ]


@dataclass(slots=True)
class QdrantVectorStore:
    settings: VectorDBSettings
    provider_name: str = "qdrant"
    _collection: str = field(init=False, default="")
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "qdrant"
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("qdrant-client is required for VECTOR_DB_PROVIDER=qdrant.") from exc

        client_kwargs: dict[str, Any] = {"timeout": self.settings.timeout}
        if self.settings.qdrant_url:
            client_kwargs["url"] = self.settings.qdrant_url
            if self.settings.qdrant_api_key:
                client_kwargs["api_key"] = self.settings.qdrant_api_key
        else:
            client_kwargs["path"] = self.settings.qdrant_local_path

        self._collection = self.settings.qdrant_collection
        self._client = QdrantClient(**client_kwargs)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        from qdrant_client.http import models

        if recreate:
            self._safe_delete_collection()

        existing_size = self._collection_vector_size()
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"Embedding size mismatch for collection '{self._collection}': existing={existing_size}, incoming={vector_size}. "
                "Re-run ingestion with recreate=True or use a matching embedding dimensions setting."
            )
        if existing_size is not None:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        from qdrant_client.http import models

        points = [
            models.PointStruct(id=record_id, vector=vector, payload=payload)
            for record_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        result = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        payloads: list[dict[str, Any]] = []
        for point in result:
            payloads.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": dict(point.payload or {}),
                }
            )
        return payloads

    def _collection_vector_size(self) -> int | None:
        try:
            info = self._client.get_collection(collection_name=self._collection)
        except Exception:
            return None

        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None

        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first and hasattr(first, "size"):
                return int(first.size)
        return None

    def _safe_delete_collection(self) -> None:
        try:
            self._client.delete_collection(collection_name=self._collection)
        except Exception:
            return


def build_vector_store(settings: VectorDBSettings) -> BaseVectorStore:
    if settings.provider == "memory":
        return InMemoryVectorStore(settings=settings)
    if settings.provider == "qdrant":
        return QdrantVectorStore(settings=settings)
    raise ValueError(f"Unsupported vector DB provider: {settings.provider}")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(x * x for x in left)) or 1.0
    right_norm = math.sqrt(sum(y * y for y in right)) or 1.0
    return dot / (left_norm * right_norm)
