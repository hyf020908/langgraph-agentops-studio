from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from services.config import VectorDBSettings, load_settings
from services.vectorstore import (
    ChromaVectorStore,
    InMemoryVectorStore,
    MilvusVectorStore,
    PgVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
    _chroma_vector_size,
    _decode_payload,
    _weaviate_uuid,
    build_vector_store,
)


def _bare_store(store_type: type[Any], settings: VectorDBSettings, **attributes: Any) -> Any:
    store = object.__new__(store_type)
    store.settings = settings
    store.provider_name = settings.provider
    for name, value in attributes.items():
        setattr(store, name, value)
    return store


def test_memory_vector_store_preserves_default_contract() -> None:
    store = InMemoryVectorStore(settings=VectorDBSettings(provider="memory"))
    store.ensure_collection(vector_size=2)
    store.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        payloads=[{"text": "alpha"}, {"text": "beta"}],
    )

    assert store.search(query_vector=[1.0, 0.0], top_k=2, score_threshold=0.5) == [
        {"id": "a", "score": 1.0, "payload": {"text": "alpha"}}
    ]
    with pytest.raises(ValueError, match="same number"):
        store.upsert(ids=["a"], vectors=[], payloads=[])
    with pytest.raises(ValueError, match="Embedding size mismatch"):
        store.ensure_collection(vector_size=3)
    store.ensure_collection(vector_size=3, recreate=True)
    assert store.search(query_vector=[1.0, 0.0, 0.0], top_k=1) == []


def test_optional_providers_are_not_imported_for_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    for package in ("pymilvus", "weaviate", "chromadb", "psycopg", "pinecone"):
        monkeypatch.setitem(sys.modules, package, None)

    store = build_vector_store(VectorDBSettings(provider="memory"))

    assert isinstance(store, InMemoryVectorStore)


def test_vector_store_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "weaviate")
    monkeypatch.setenv("WEAVIATE_URL", "https://vectors.example.com")
    monkeypatch.setenv("WEAVIATE_API_KEY", "test-key")
    monkeypatch.setenv("WEAVIATE_COLLECTION", "ProjectKnowledge")
    monkeypatch.setenv("WEAVIATE_GRPC_PORT", "443")
    monkeypatch.setenv("WEAVIATE_GRPC_SECURE", "true")
    monkeypatch.setenv("CHROMA_HEADERS_JSON", '{"Authorization":"Bearer test"}')
    monkeypatch.setenv("PGVECTOR_SCHEMA", "agentops")
    monkeypatch.setenv("PINECONE_CLOUD", "gcp")
    load_settings.cache_clear()
    try:
        settings = load_settings().vector_db

        assert settings.provider == "weaviate"
        assert settings.weaviate_url == "https://vectors.example.com"
        assert settings.weaviate_api_key == "test-key"
        assert settings.weaviate_collection == "ProjectKnowledge"
        assert settings.weaviate_grpc_port == 443
        assert settings.weaviate_grpc_secure is True
        assert settings.chroma_headers == {"Authorization": "Bearer test"}
        assert settings.pgvector_schema == "agentops"
        assert settings.pinecone_cloud == "gcp"
    finally:
        load_settings.cache_clear()


@pytest.mark.parametrize(
    ("provider", "package", "extra"),
    [
        ("milvus", "pymilvus", "milvus"),
        ("weaviate", "weaviate", "weaviate"),
        ("chroma", "chromadb", "chroma"),
        ("pgvector", "psycopg", "pgvector"),
        ("pinecone", "pinecone", "pinecone"),
    ],
)
def test_optional_provider_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    package: str,
    extra: str,
) -> None:
    monkeypatch.setitem(sys.modules, package, None)
    settings_kwargs: dict[str, Any] = {"provider": provider}
    if provider == "pgvector":
        settings_kwargs["pgvector_dsn"] = "postgresql://example"
    if provider == "pinecone":
        settings_kwargs["pinecone_api_key"] = "test-key"

    with pytest.raises(RuntimeError, match=rf"\.\[{extra}\]"):
        build_vector_store(VectorDBSettings(**settings_kwargs))


def test_selected_providers_construct_only_their_configured_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    pymilvus = ModuleType("pymilvus")

    class FakeMilvusClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["milvus"] = kwargs

    pymilvus.MilvusClient = FakeMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)

    chromadb = ModuleType("chromadb")
    chromadb.__path__ = []
    chroma_errors = ModuleType("chromadb.errors")

    class NotFoundError(Exception):
        pass

    def persistent_client(**kwargs: Any) -> SimpleNamespace:
        captured["chroma"] = kwargs
        return SimpleNamespace()

    chromadb.PersistentClient = persistent_client
    chromadb.HttpClient = lambda **kwargs: SimpleNamespace()
    chroma_errors.NotFoundError = NotFoundError
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.errors", chroma_errors)

    psycopg = ModuleType("psycopg")

    def connect(dsn: str, **kwargs: Any) -> SimpleNamespace:
        captured["pgvector"] = {"dsn": dsn, **kwargs}
        return SimpleNamespace()

    psycopg.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)

    pinecone = ModuleType("pinecone")

    class FakePinecone:
        def __init__(self, **kwargs: Any) -> None:
            captured["pinecone"] = kwargs

    pinecone.Pinecone = FakePinecone
    pinecone.ServerlessSpec = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "pinecone", pinecone)

    milvus = build_vector_store(
        VectorDBSettings(
            provider="milvus",
            milvus_uri="https://milvus.example.com",
            milvus_token="token",
            milvus_db_name="agentops",
        )
    )
    chroma = build_vector_store(VectorDBSettings(provider="chroma", chroma_path="/tmp/test-chroma"))
    pgvector = build_vector_store(
        VectorDBSettings(provider="pgvector", pgvector_dsn="postgresql://db/agentops")
    )
    pinecone_store = build_vector_store(
        VectorDBSettings(provider="pinecone", pinecone_api_key="pinecone-key")
    )

    assert isinstance(milvus, MilvusVectorStore)
    assert isinstance(chroma, ChromaVectorStore)
    assert isinstance(pgvector, PgVectorStore)
    assert isinstance(pinecone_store, PineconeVectorStore)
    assert captured["milvus"]["uri"] == "https://milvus.example.com"
    assert captured["milvus"]["token"] == "token"
    assert captured["chroma"]["path"] == "/tmp/test-chroma"
    assert captured["pgvector"]["dsn"] == "postgresql://db/agentops"
    assert captured["pgvector"]["autocommit"] is True
    assert captured["pinecone"]["api_key"] == "pinecone-key"


def test_weaviate_constructor_maps_http_and_grpc_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    weaviate = ModuleType("weaviate")
    weaviate.__path__ = []
    classes = ModuleType("weaviate.classes")
    classes.__path__ = []
    init_module = ModuleType("weaviate.classes.init")

    class ConfigValue:
        def __init__(self, **kwargs: Any) -> None:
            self.values = kwargs

    class Auth:
        @staticmethod
        def api_key(value: str) -> str:
            return f"auth:{value}"

    def connect_to_custom(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    init_module.AdditionalConfig = ConfigValue
    init_module.Timeout = ConfigValue
    init_module.Auth = Auth
    weaviate.connect_to_custom = connect_to_custom
    monkeypatch.setitem(sys.modules, "weaviate", weaviate)
    monkeypatch.setitem(sys.modules, "weaviate.classes", classes)
    monkeypatch.setitem(sys.modules, "weaviate.classes.init", init_module)

    store = build_vector_store(
        VectorDBSettings(
            provider="weaviate",
            weaviate_url="https://vectors.example.com:8443",
            weaviate_api_key="test-key",
            weaviate_grpc_host="grpc.example.com",
            weaviate_grpc_port=9443,
            weaviate_grpc_secure=True,
        )
    )

    assert isinstance(store, WeaviateVectorStore)
    assert captured["http_host"] == "vectors.example.com"
    assert captured["http_port"] == 8443
    assert captured["http_secure"] is True
    assert captured["grpc_host"] == "grpc.example.com"
    assert captured["grpc_port"] == 9443
    assert captured["grpc_secure"] is True
    assert captured["auth_credentials"] == "auth:test-key"


def test_milvus_adapter_creates_string_id_collection_and_normalizes_hits() -> None:
    class FakeMilvusClient:
        def __init__(self) -> None:
            self.created: dict[str, Any] | None = None
            self.upserted: list[dict[str, Any]] | None = None

        def has_collection(self, **kwargs: Any) -> bool:
            return False

        def create_collection(self, **kwargs: Any) -> None:
            self.created = kwargs

        def upsert(self, **kwargs: Any) -> None:
            self.upserted = kwargs["data"]

        def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
            return [
                [
                    {
                        "id": "doc-1",
                        "distance": 0.91,
                        "entity": {"payload": {"text": "alpha", "metadata": {"rank": 1}}},
                    },
                    {"id": "doc-2", "distance": 0.2, "entity": {"payload": {"text": "beta"}}},
                ]
            ]

    client = FakeMilvusClient()
    store = _bare_store(
        MilvusVectorStore,
        VectorDBSettings(provider="milvus"),
        _collection="knowledge",
        _client=client,
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[{"metadata": {"rank": 1}}])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=2, score_threshold=0.5)

    assert client.created is not None
    assert client.created["id_type"] == "string"
    assert client.created["metric_type"] == "COSINE"
    assert client.upserted == [
        {"id": "doc-1", "vector": [0.1, 0.2, 0.3], "payload": {"metadata": {"rank": 1}}}
    ]
    assert results == [
        {
            "id": "doc-1",
            "score": 0.91,
            "payload": {"text": "alpha", "metadata": {"rank": 1}},
        }
    ]


def test_milvus_adapter_rejects_existing_dimension_mismatch() -> None:
    client = SimpleNamespace(
        has_collection=lambda **kwargs: True,
        describe_collection=lambda **kwargs: {
            "fields": [{"name": "vector", "params": {"dim": "2"}}]
        },
    )
    store = _bare_store(
        MilvusVectorStore,
        VectorDBSettings(provider="milvus"),
        _collection="knowledge",
        _client=client,
    )

    with pytest.raises(ValueError, match="existing=2, incoming=3"):
        store.ensure_collection(vector_size=3)


def _install_weaviate_method_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    config_module = ModuleType("weaviate.classes.config")

    class VectorIndex:
        @staticmethod
        def hnsw(**kwargs: Any) -> dict[str, Any]:
            return kwargs

    class Vectors:
        @staticmethod
        def self_provided(**kwargs: Any) -> dict[str, Any]:
            return kwargs

    class Property:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    config_module.Configure = SimpleNamespace(VectorIndex=VectorIndex, Vectors=Vectors)
    config_module.DataType = SimpleNamespace(TEXT="text")
    config_module.Property = Property
    config_module.VectorDistances = SimpleNamespace(COSINE="cosine")

    query_module = ModuleType("weaviate.classes.query")

    class MetadataQuery:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    query_module.MetadataQuery = MetadataQuery
    monkeypatch.setitem(sys.modules, "weaviate.classes.config", config_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes.query", query_module)


def test_weaviate_adapter_upserts_and_decodes_nested_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_weaviate_method_modules(monkeypatch)

    class FakeData:
        def __init__(self) -> None:
            self.objects: dict[str, dict[str, Any]] = {}

        def exists(self, object_id: str) -> bool:
            return object_id in self.objects

        def insert(self, *, uuid: str, properties: dict[str, Any], vector: list[float]) -> None:
            self.objects[uuid] = {"properties": properties, "vector": vector}

        def update(self, *, uuid: str, properties: dict[str, Any], vector: list[float]) -> None:
            self.objects[uuid] = {"properties": properties, "vector": vector}

    nested_payload = {"text": "alpha", "metadata": {"section": 2}}

    class FakeQuery:
        def fetch_objects(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(objects=[])

        def near_vector(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                objects=[
                    SimpleNamespace(
                        uuid="generated-uuid",
                        properties={"record_id": "doc-1", "payload_json": json.dumps(nested_payload)},
                        metadata=SimpleNamespace(distance=0.08),
                    ),
                    SimpleNamespace(
                        uuid="low-score",
                        properties={"record_id": "doc-2", "payload_json": "{}"},
                        metadata=SimpleNamespace(distance=0.8),
                    ),
                ]
            )

    collection = SimpleNamespace(data=FakeData(), query=FakeQuery())

    class FakeCollections:
        def __init__(self) -> None:
            self.created: dict[str, Any] | None = None

        def exists(self, name: str) -> bool:
            return False

        def create(self, **kwargs: Any) -> None:
            self.created = kwargs

        def use(self, name: str) -> Any:
            return collection

    collections = FakeCollections()
    store = _bare_store(
        WeaviateVectorStore,
        VectorDBSettings(provider="weaviate"),
        _collection="AgentOpsKnowledge",
        _client=SimpleNamespace(collections=collections),
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[nested_payload])
    store.upsert(ids=["doc-1"], vectors=[[0.3, 0.2, 0.1]], payloads=[nested_payload])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=2, score_threshold=0.5)

    assert collections.created is not None
    assert collections.created["name"] == "AgentOpsKnowledge"
    assert len(collection.data.objects) == 1
    assert results == [{"id": "doc-1", "score": 0.92, "payload": nested_payload}]


def test_weaviate_non_uuid_ids_are_stable() -> None:
    assert _weaviate_uuid("doc-1") == _weaviate_uuid("doc-1")
    assert _weaviate_uuid("doc-1") != _weaviate_uuid("doc-2")


def test_chroma_adapter_round_trips_nested_payload() -> None:
    class NotFoundError(Exception):
        pass

    nested_payload = {"text": "alpha", "metadata": {"extension": ".md"}}

    class FakeCollection:
        metadata = {"hnsw:space": "cosine", "embedding_dimensions": 3}

        def __init__(self) -> None:
            self.upsert_call: dict[str, Any] | None = None

        def upsert(self, **kwargs: Any) -> None:
            self.upsert_call = kwargs

        def count(self) -> int:
            return 2

        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ids": [["doc-1", "doc-2"]],
                "distances": [[0.05, 0.7]],
                "metadatas": [
                    [
                        {"payload_json": json.dumps(nested_payload)},
                        {"payload_json": json.dumps({"text": "beta"})},
                    ]
                ],
            }

    collection = FakeCollection()

    class FakeClient:
        def get_collection(self, **kwargs: Any) -> Any:
            raise NotFoundError

        def create_collection(self, **kwargs: Any) -> Any:
            self.create_call = kwargs
            return collection

    client = FakeClient()
    store = _bare_store(
        ChromaVectorStore,
        VectorDBSettings(provider="chroma"),
        _collection="knowledge",
        _client=client,
        _not_found_error=NotFoundError,
        _collection_handle=None,
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[nested_payload])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=5, score_threshold=0.5)

    assert client.create_call["metadata"]["embedding_dimensions"] == 3
    assert json.loads(collection.upsert_call["metadatas"][0]["payload_json"]) == nested_payload
    assert results == [{"id": "doc-1", "score": 0.95, "payload": nested_payload}]


def test_chroma_dimension_fallback_uses_collection_get() -> None:
    class FakeCollection:
        metadata: dict[str, Any] = {}

        def __init__(self) -> None:
            self.get_call: dict[str, Any] | None = None

        def get(self, **kwargs: Any) -> dict[str, Any]:
            self.get_call = kwargs
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    collection = FakeCollection()

    assert _chroma_vector_size(collection) == 3
    assert collection.get_call == {"limit": 1, "include": ["embeddings"]}


def test_pgvector_adapter_uses_cosine_sql_and_json_payload() -> None:
    nested_payload = {"text": "alpha", "metadata": {"rank": 1}}

    class FakeCursor:
        def __init__(self) -> None:
            self.executed: list[tuple[str, Any]] = []
            self.executed_many: tuple[str, list[tuple[Any, ...]]] | None = None

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, statement: str, params: Any = None) -> None:
            self.executed.append((statement, params))

        def executemany(self, statement: str, rows: list[tuple[Any, ...]]) -> None:
            self.executed_many = (statement, rows)

        def fetchone(self) -> None:
            return None

        def fetchall(self) -> list[tuple[str, float, dict[str, Any]]]:
            return [("doc-1", 0.93, nested_payload)]

    cursor = FakeCursor()
    connection = SimpleNamespace(cursor=lambda: cursor)
    settings = VectorDBSettings(
        provider="pgvector",
        pgvector_dsn="postgresql://example",
        pgvector_schema="agentops",
        pgvector_table="knowledge",
    )
    store = _bare_store(
        PgVectorStore,
        settings,
        _collection="knowledge",
        _qualified_table='"agentops"."knowledge"',
        _connection=connection,
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[nested_payload])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=2, score_threshold=0.5)

    assert any("embedding vector(3)" in statement for statement, _ in cursor.executed)
    assert cursor.executed_many is not None
    assert json.loads(cursor.executed_many[1][0][2]) == nested_payload
    search_statement, search_params = cursor.executed[-1]
    assert "embedding <=> %s::vector" in search_statement
    assert search_params[1:] == (0.5, 0.5, 2)
    assert results == [{"id": "doc-1", "score": 0.93, "payload": nested_payload}]


def test_pgvector_identifiers_are_strictly_validated() -> None:
    with pytest.raises(ValidationError, match="pgvector_table"):
        VectorDBSettings(
            provider="pgvector",
            pgvector_dsn="postgresql://example",
            pgvector_table='knowledge"; DROP TABLE users; --',
        )


def test_pinecone_adapter_creates_index_and_round_trips_payload() -> None:
    nested_payload = {"text": "alpha", "metadata": {"section": 1}}

    class FakeIndexes:
        def __init__(self) -> None:
            self.create_call: dict[str, Any] | None = None

        def exists(self, name: str) -> bool:
            return False

        def configure(self, name: str, **kwargs: Any) -> None:
            return None

        def create(self, **kwargs: Any) -> None:
            self.create_call = kwargs

    class FakeIndex:
        def __init__(self) -> None:
            self.upsert_call: dict[str, Any] | None = None

        def upsert(self, **kwargs: Any) -> None:
            self.upsert_call = kwargs

        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "matches": [
                    {
                        "id": "doc-1",
                        "score": 0.9,
                        "metadata": {"payload_json": json.dumps(nested_payload)},
                    },
                    {"id": "doc-2", "score": 0.2, "metadata": {"payload_json": "{}"}},
                ]
            }

    indexes = FakeIndexes()
    index = FakeIndex()
    client = SimpleNamespace(indexes=indexes, index=lambda *args, **kwargs: index)
    settings = VectorDBSettings(provider="pinecone", pinecone_api_key="test-key")
    store = _bare_store(
        PineconeVectorStore,
        settings,
        _collection="agentops-knowledge",
        _client=client,
        _serverless_spec=lambda **kwargs: kwargs,
        _index=None,
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[nested_payload])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=2, score_threshold=0.5)

    assert indexes.create_call is not None
    assert indexes.create_call["dimension"] == 3
    assert indexes.create_call["metric"] == "cosine"
    assert index.upsert_call is not None
    stored_metadata = index.upsert_call["vectors"][0]["metadata"]
    assert json.loads(stored_metadata["payload_json"]) == nested_payload
    assert results == [{"id": "doc-1", "score": 0.9, "payload": nested_payload}]


def _install_qdrant_models(monkeypatch: pytest.MonkeyPatch) -> None:
    models_module = ModuleType("qdrant_client.http.models")

    class VectorParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class PointStruct:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    models_module.VectorParams = VectorParams
    models_module.PointStruct = PointStruct
    models_module.Distance = SimpleNamespace(COSINE="cosine")
    http_module = ModuleType("qdrant_client.http")
    http_module.models = models_module
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models_module)


def test_qdrant_adapter_contract_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_qdrant_models(monkeypatch)

    class FakeClient:
        def __init__(self) -> None:
            self.created: dict[str, Any] | None = None
            self.upserted: dict[str, Any] | None = None

        def get_collection(self, **kwargs: Any) -> Any:
            raise LookupError

        def collection_exists(self, **kwargs: Any) -> bool:
            return False

        def create_collection(self, **kwargs: Any) -> None:
            self.created = kwargs

        def upsert(self, **kwargs: Any) -> None:
            self.upserted = kwargs

        def search(self, **kwargs: Any) -> list[Any]:
            return [SimpleNamespace(id="doc-1", score=0.94, payload={"metadata": {"rank": 1}})]

    client = FakeClient()
    store = _bare_store(
        QdrantVectorStore,
        VectorDBSettings(provider="qdrant"),
        _collection="knowledge",
        _client=client,
    )

    store.ensure_collection(vector_size=3)
    store.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[{"metadata": {"rank": 1}}])
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=1)

    assert client.created is not None
    assert client.created["vectors_config"].distance == "cosine"
    assert client.upserted is not None
    assert client.upserted["points"][0].id == "doc-1"
    assert results == [{"id": "doc-1", "score": 0.94, "payload": {"metadata": {"rank": 1}}}]


def test_payload_decoder_is_defensive() -> None:
    assert _decode_payload('{"nested":{"ok":true}}') == {"nested": {"ok": True}}
    assert _decode_payload("not-json") == {}
    assert _decode_payload("[1, 2]") == {}
