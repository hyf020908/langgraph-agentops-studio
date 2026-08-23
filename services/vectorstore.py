from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from services.config import VectorDBSettings


_DEFAULT_UPSERT_BATCH_SIZE = 100


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
            self._vector_size = 0
        if self._vector_size and self._vector_size != vector_size:
            raise ValueError(
                f"Embedding size mismatch for memory vector store: existing={self._vector_size}, incoming={vector_size}"
            )
        self._vector_size = vector_size

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
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
            raise RuntimeError(
                "qdrant-client is required for VECTOR_DB_PROVIDER=qdrant. Install the project base dependencies."
            ) from exc

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

        exists = bool(self._client.collection_exists(collection_name=self._collection))
        if recreate and exists:
            self._client.delete_collection(collection_name=self._collection)
            exists = False

        if exists:
            existing_size = self._collection_vector_size()
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        from qdrant_client.http import models

        _validate_batch(ids, vectors, payloads)
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
        info = self._client.get_collection(collection_name=self._collection)

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

@dataclass(slots=True)
class MilvusVectorStore:
    settings: VectorDBSettings
    provider_name: str = "milvus"
    _collection: str = field(init=False, default="")
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "milvus"
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "pymilvus is required for VECTOR_DB_PROVIDER=milvus. Install with pip install -e '.[milvus]'."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "uri": self.settings.milvus_uri,
            "db_name": self.settings.milvus_db_name,
            "timeout": self.settings.timeout,
        }
        if self.settings.milvus_token:
            client_kwargs["token"] = self.settings.milvus_token
        self._collection = self.settings.milvus_collection
        self._client = MilvusClient(**client_kwargs)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = bool(self._client.has_collection(collection_name=self._collection))
        if recreate and exists:
            self._client.drop_collection(collection_name=self._collection, timeout=self.settings.timeout)
            exists = False

        if exists:
            description = self._client.describe_collection(
                collection_name=self._collection,
                timeout=self.settings.timeout,
            )
            existing_size = _milvus_vector_size(description)
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            return

        self._client.create_collection(
            collection_name=self._collection,
            dimension=vector_size,
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            max_length=512,
            consistency_level="Strong",
            timeout=self.settings.timeout,
        )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
        data = [
            {"id": record_id, "vector": vector, "payload": payload}
            for record_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        for batch in _batches(data, _DEFAULT_UPSERT_BATCH_SIZE):
            self._client.upsert(collection_name=self._collection, data=batch, timeout=self.settings.timeout)

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        response = self._client.search(
            collection_name=self._collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["payload"],
            search_params={"metric_type": "COSINE", "params": {}},
            timeout=self.settings.timeout,
        )
        hits = response[0] if response else []
        results: list[dict[str, Any]] = []
        for hit in hits:
            score = float(_item_value(hit, "distance", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            entity = _item_value(hit, "entity", {})
            payload = _item_value(entity, "payload", {}) if entity else {}
            results.append(
                {
                    "id": str(_item_value(hit, "id", "")),
                    "score": score,
                    "payload": dict(payload) if isinstance(payload, Mapping) else {},
                }
            )
        return results


@dataclass(slots=True)
class WeaviateVectorStore:
    settings: VectorDBSettings
    provider_name: str = "weaviate"
    _collection: str = field(init=False, default="")
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "weaviate"
        try:
            import weaviate
            from weaviate.classes.init import AdditionalConfig, Auth, Timeout
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "weaviate-client is required for VECTOR_DB_PROVIDER=weaviate. "
                "Install with pip install -e '.[weaviate]'."
            ) from exc

        endpoint = urlparse(self.settings.weaviate_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("WEAVIATE_URL must be an http:// or https:// URL with a hostname.")

        secure = endpoint.scheme == "https"
        auth = Auth.api_key(self.settings.weaviate_api_key) if self.settings.weaviate_api_key else None
        grpc_secure = self.settings.weaviate_grpc_secure
        self._collection = self.settings.weaviate_collection
        self._client = weaviate.connect_to_custom(
            http_host=endpoint.hostname,
            http_port=endpoint.port or (443 if secure else 80),
            http_secure=secure,
            grpc_host=self.settings.weaviate_grpc_host or endpoint.hostname,
            grpc_port=self.settings.weaviate_grpc_port or (443 if secure else 50051),
            grpc_secure=secure if grpc_secure is None else grpc_secure,
            auth_credentials=auth,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    init=self.settings.timeout,
                    query=self.settings.timeout,
                    insert=self.settings.timeout,
                )
            ),
        )

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        exists = bool(self._client.collections.exists(self._collection))
        if recreate and exists:
            self._client.collections.delete(self._collection)
            exists = False

        if exists:
            existing_size = self._collection_vector_size()
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            return

        self._client.collections.create(
            name=self._collection,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
            ),
            properties=[
                Property(name="record_id", data_type=DataType.TEXT),
                Property(name="payload_json", data_type=DataType.TEXT),
            ],
        )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
        collection = self._client.collections.use(self._collection)
        for record_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            object_id = _weaviate_uuid(record_id)
            properties = {
                "record_id": record_id,
                "payload_json": _encode_payload(payload),
            }
            if collection.data.exists(object_id):
                collection.data.update(uuid=object_id, properties=properties, vector=vector)
            else:
                collection.data.insert(uuid=object_id, properties=properties, vector=vector)

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        from weaviate.classes.query import MetadataQuery

        collection = self._client.collections.use(self._collection)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["record_id", "payload_json"],
            return_metadata=MetadataQuery(distance=True),
        )
        results: list[dict[str, Any]] = []
        for item in response.objects:
            distance = float(getattr(item.metadata, "distance", 1.0))
            score = 1.0 - distance
            if score_threshold is not None and score < score_threshold:
                continue
            properties = dict(item.properties or {})
            results.append(
                {
                    "id": str(properties.get("record_id") or item.uuid),
                    "score": score,
                    "payload": _decode_payload(properties.get("payload_json")),
                }
            )
        return results

    def _collection_vector_size(self) -> int | None:
        collection = self._client.collections.use(self._collection)
        response = collection.query.fetch_objects(limit=1, include_vector=True)
        if not response.objects:
            return None
        vector = response.objects[0].vector
        if isinstance(vector, Mapping):
            vector = next(iter(vector.values()), None)
        if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
            return len(vector)
        return None


@dataclass(slots=True)
class ChromaVectorStore:
    settings: VectorDBSettings
    provider_name: str = "chroma"
    _collection: str = field(init=False, default="")
    _client: Any = field(init=False, default=None)
    _not_found_error: type[Exception] = field(init=False, default=Exception)
    _collection_handle: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "chroma"
        try:
            import chromadb
            from chromadb.errors import NotFoundError
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "chromadb is required for VECTOR_DB_PROVIDER=chroma. Install with pip install -e '.[chroma]'."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "tenant": self.settings.chroma_tenant,
            "database": self.settings.chroma_database,
        }
        if self.settings.chroma_host:
            client_kwargs.update(
                {
                    "host": self.settings.chroma_host,
                    "port": self.settings.chroma_port,
                    "ssl": self.settings.chroma_ssl,
                    "headers": self.settings.chroma_headers,
                }
            )
            self._client = chromadb.HttpClient(**client_kwargs)
        else:
            client_kwargs["path"] = self.settings.chroma_path
            self._client = chromadb.PersistentClient(**client_kwargs)

        self._not_found_error = NotFoundError
        self._collection = self.settings.chroma_collection

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        collection = self._get_collection()
        if recreate and collection is not None:
            self._client.delete_collection(name=self._collection)
            collection = None
        if collection is not None:
            existing_size = _chroma_vector_size(collection)
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            distance = (collection.metadata or {}).get("hnsw:space")
            if distance is not None and str(distance).lower() != "cosine":
                raise ValueError(
                    f"Distance mismatch for Chroma collection '{self._collection}': "
                    f"existing={distance}, required=cosine. Re-run ingestion with recreate=True."
                )
            self._collection_handle = collection
            return

        self._collection_handle = self._client.create_collection(
            name=self._collection,
            metadata={"hnsw:space": "cosine", "embedding_dimensions": vector_size},
            embedding_function=None,
        )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
        collection = self._require_collection()
        metadata = [{"payload_json": _encode_payload(payload)} for payload in payloads]
        max_batch_size = getattr(self._client, "get_max_batch_size", None)
        configured_batch_size = int(max_batch_size()) if callable(max_batch_size) else _DEFAULT_UPSERT_BATCH_SIZE
        batch_size = max(1, configured_batch_size)
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            collection.upsert(
                ids=ids[start:stop],
                embeddings=vectors[start:stop],
                metadatas=metadata[start:stop],
            )

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        collection = self._require_collection()
        result_count = min(top_k, int(collection.count()))
        if result_count <= 0:
            return []
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=result_count,
            include=["metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        metadata_groups = response.get("metadatas")
        metadatas = metadata_groups[0] if metadata_groups else [None] * len(ids)
        results: list[dict[str, Any]] = []
        for record_id, distance, metadata in zip(ids, distances, metadatas, strict=True):
            score = 1.0 - float(distance)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                {
                    "id": str(record_id),
                    "score": score,
                    "payload": _decode_payload((metadata or {}).get("payload_json")),
                }
            )
        return results

    def _get_collection(self) -> Any | None:
        try:
            return self._client.get_collection(name=self._collection, embedding_function=None)
        except self._not_found_error:
            return None

    def _require_collection(self) -> Any:
        if self._collection_handle is None:
            self._collection_handle = self._get_collection()
        if self._collection_handle is None:
            raise RuntimeError(f"Chroma collection '{self._collection}' does not exist; run ingestion first.")
        return self._collection_handle


@dataclass(slots=True)
class PgVectorStore:
    settings: VectorDBSettings
    provider_name: str = "pgvector"
    _collection: str = field(init=False, default="")
    _qualified_table: str = field(init=False, default="")
    _connection: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "pgvector"
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "psycopg[binary] is required for VECTOR_DB_PROVIDER=pgvector. "
                "Install with pip install -e '.[pgvector]'."
            ) from exc

        if not self.settings.pgvector_dsn:
            raise ValueError("PGVECTOR_DSN is required for VECTOR_DB_PROVIDER=pgvector.")
        self._collection = self.settings.pgvector_table
        self._qualified_table = f'"{self.settings.pgvector_schema}"."{self.settings.pgvector_table}"'
        self._connection = psycopg.connect(
            self.settings.pgvector_dsn,
            connect_timeout=max(1, math.ceil(self.settings.timeout)),
            autocommit=True,
        )

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.settings.pgvector_schema}"')
            if recreate:
                cursor.execute(f"DROP TABLE IF EXISTS {self._qualified_table}")

            cursor.execute(
                """
                SELECT format_type(attribute.atttypid, attribute.atttypmod)
                FROM pg_attribute AS attribute
                JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = %s
                  AND relation.relname = %s
                  AND attribute.attname = 'embedding'
                  AND NOT attribute.attisdropped
                """,
                (self.settings.pgvector_schema, self.settings.pgvector_table),
            )
            row = cursor.fetchone()
            existing_size = _pgvector_size(row[0]) if row else None
            if row and existing_size is None:
                raise ValueError(
                    f"Existing pgvector table '{self._collection}' must define embedding as vector(N); "
                    f"found {row[0]!s}. Re-run ingestion with recreate=True."
                )
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            if row:
                return

            cursor.execute(
                f"""
                CREATE TABLE {self._qualified_table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({int(vector_size)}) NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
        statement = f"""
            INSERT INTO {self._qualified_table} (id, embedding, payload)
            VALUES (%s, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                payload = EXCLUDED.payload
        """
        rows = [
            (record_id, _encode_vector(vector), _encode_payload(payload))
            for record_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        with self._connection.cursor() as cursor:
            cursor.executemany(statement, rows)

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        statement = f"""
            SELECT id, score, payload
            FROM (
                SELECT id, 1 - (embedding <=> %s::vector) AS score, payload
                FROM {self._qualified_table}
            ) AS ranked
            WHERE (%s IS NULL OR score >= %s)
            ORDER BY score DESC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                statement,
                (_encode_vector(query_vector), score_threshold, score_threshold, top_k),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": str(record_id),
                "score": float(score),
                "payload": _decode_payload(payload),
            }
            for record_id, score, payload in rows
        ]


@dataclass(slots=True)
class PineconeVectorStore:
    settings: VectorDBSettings
    provider_name: str = "pinecone"
    _collection: str = field(init=False, default="")
    _client: Any = field(init=False, default=None)
    _serverless_spec: Any = field(init=False, default=None)
    _index: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider_name = "pinecone"
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "pinecone is required for VECTOR_DB_PROVIDER=pinecone. Install with pip install -e '.[pinecone]'."
            ) from exc

        if not self.settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for VECTOR_DB_PROVIDER=pinecone.")
        self._collection = self.settings.pinecone_index
        self._client = Pinecone(api_key=self.settings.pinecone_api_key, timeout=self.settings.timeout)
        self._serverless_spec = ServerlessSpec

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        indexes = self._client.indexes
        exists = bool(indexes.exists(self._collection))
        timeout = max(1, math.ceil(self.settings.timeout))
        if recreate and exists:
            indexes.configure(self._collection, deletion_protection="disabled")
            indexes.delete(self._collection, timeout=timeout)
            exists = False

        if exists:
            description = indexes.describe(self._collection)
            configured_size = _item_value(description, "dimension")
            existing_size = int(configured_size) if configured_size is not None else None
            _raise_for_dimension_mismatch(self.provider_name, self._collection, existing_size, vector_size)
            metric = str(_item_value(description, "metric", "cosine")).lower()
            if metric != "cosine":
                raise ValueError(
                    f"Metric mismatch for Pinecone index '{self._collection}': existing={metric}, required=cosine. "
                    "Re-run ingestion with recreate=True."
                )
        else:
            indexes.create(
                name=self._collection,
                vector_type="dense",
                dimension=vector_size,
                metric="cosine",
                spec=self._serverless_spec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
                deletion_protection="disabled",
                timeout=timeout,
            )
        self._index = self._build_index()

    def upsert(self, *, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        _validate_batch(ids, vectors, payloads)
        index = self._require_index()
        records = [
            {
                "id": record_id,
                "values": vector,
                "metadata": {"payload_json": _encode_payload(payload)},
            }
            for record_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        for batch in _batches(records, _DEFAULT_UPSERT_BATCH_SIZE):
            index.upsert(
                vectors=batch,
                namespace=self.settings.pinecone_namespace,
                timeout=self.settings.timeout,
            )

    def search(self, *, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict[str, Any]]:
        response = self._require_index().query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.settings.pinecone_namespace,
            include_metadata=True,
            timeout=self.settings.timeout,
        )
        results: list[dict[str, Any]] = []
        for match in _item_value(response, "matches", []) or []:
            score = float(_item_value(match, "score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            metadata = _item_value(match, "metadata", {}) or {}
            results.append(
                {
                    "id": str(_item_value(match, "id", "")),
                    "score": score,
                    "payload": _decode_payload(_item_value(metadata, "payload_json")),
                }
            )
        return results

    def _build_index(self) -> Any:
        if self.settings.pinecone_host:
            return self._client.index(host=self.settings.pinecone_host)
        return self._client.index(self._collection)

    def _require_index(self) -> Any:
        if self._index is None:
            self._index = self._build_index()
        return self._index


def build_vector_store(settings: VectorDBSettings) -> BaseVectorStore:
    providers: dict[str, type[Any]] = {
        "memory": InMemoryVectorStore,
        "qdrant": QdrantVectorStore,
        "milvus": MilvusVectorStore,
        "weaviate": WeaviateVectorStore,
        "chroma": ChromaVectorStore,
        "pgvector": PgVectorStore,
        "pinecone": PineconeVectorStore,
    }
    store_type = providers.get(settings.provider)
    if store_type is None:
        raise ValueError(f"Unsupported vector DB provider: {settings.provider}")
    return store_type(settings=settings)


def _validate_batch(ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
    if len(ids) != len(vectors) or len(ids) != len(payloads):
        raise ValueError("ids, vectors, and payloads must contain the same number of items.")


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _raise_for_dimension_mismatch(
    provider: str,
    collection: str,
    existing_size: int | None,
    incoming_size: int,
) -> None:
    if existing_size is None or existing_size == incoming_size:
        return
    raise ValueError(
        f"Embedding size mismatch for {provider} collection '{collection}': "
        f"existing={existing_size}, incoming={incoming_size}. "
        "Re-run ingestion with recreate=True or use a matching embedding dimensions setting."
    )


def _milvus_vector_size(description: Any) -> int | None:
    for item in _item_value(description, "fields", []) or []:
        if _item_value(item, "name") != "vector":
            continue
        params = _item_value(item, "params", {}) or {}
        dimension = _item_value(params, "dim")
        return int(dimension) if dimension is not None else None
    return None


def _chroma_vector_size(collection: Any) -> int | None:
    metadata = collection.metadata or {}
    configured_size = metadata.get("embedding_dimensions")
    if configured_size is not None:
        return int(configured_size)
    records = collection.get(limit=1, include=["embeddings"])
    embeddings = records.get("embeddings")
    if embeddings is not None and len(embeddings) > 0:
        return len(embeddings[0])
    return None


def _pgvector_size(type_name: Any) -> int | None:
    match = re.fullmatch(r"vector\((\d+)\)", str(type_name))
    return int(match.group(1)) if match else None


def _weaviate_uuid(record_id: str) -> str:
    try:
        return str(UUID(record_id))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, record_id))


def _encode_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _decode_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, (str, bytes, bytearray)):
        return {}
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _encode_vector(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"), allow_nan=False)


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(x * x for x in left)) or 1.0
    right_norm = math.sqrt(sum(y * y for y in right)) or 1.0
    return dot / (left_norm * right_norm)
