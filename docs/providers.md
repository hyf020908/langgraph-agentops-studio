# Provider Configuration

## LLM Providers

`LLM_PROVIDER` supports:

- `openai`
- `deepseek`
- `openai_compatible`

Key variables:

- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_TIMEOUT`
- `LLM_EXTRA_HEADERS_JSON`
- `LLM_EXTRA_KWARGS_JSON`

## Embedding Providers

`EMBEDDING_PROVIDER` supports:

- `openai`
- `deepseek`
- `openai_compatible`

Key variables:

- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_DIMENSIONS`
- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_TIMEOUT`
- `EMBEDDING_EXTRA_HEADERS_JSON`
- `EMBEDDING_EXTRA_KWARGS_JSON`

## Vector Store Providers

`VECTOR_DB_PROVIDER` supports:

- `qdrant`
- `milvus`
- `weaviate`
- `chroma`
- `pgvector`
- `pinecone`
- `memory`

Qdrant remains the default and is included in the base installation. The other
production clients are optional so selecting one backend does not import or
require the SDKs for the others:

```bash
pip install -e ".[milvus]"
pip install -e ".[weaviate]"
pip install -e ".[chroma]"
pip install -e ".[pgvector]"
pip install -e ".[pinecone]"
# Or install every optional vector-store client:
pip install -e ".[vectorstores]"
```

Backend variables:

- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `QDRANT_LOCAL_PATH`
- Milvus: `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_DB_NAME`, `MILVUS_COLLECTION`
- Weaviate: `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`,
  `WEAVIATE_GRPC_HOST`, `WEAVIATE_GRPC_PORT`, `WEAVIATE_GRPC_SECURE`
- Chroma: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`, `CHROMA_HEADERS_JSON`,
  `CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_PATH`, `CHROMA_COLLECTION`
- pgvector: `PGVECTOR_DSN`, `PGVECTOR_SCHEMA`, `PGVECTOR_TABLE`
- Pinecone: `PINECONE_API_KEY`, `PINECONE_HOST`, `PINECONE_INDEX`,
  `PINECONE_NAMESPACE`, `PINECONE_CLOUD`, `PINECONE_REGION`

`VECTOR_DB_TIMEOUT` applies to every network-backed adapter. When `CHROMA_HOST`
is empty, Chroma uses `CHROMA_PATH` for local persistent storage. pgvector
creates the `vector` extension, schema, and table if the database account has
permission. Pinecone creates a serverless cosine index when the configured
index does not exist.

All adapters expose the same collection/create, upsert, and cosine-search
contract. Nested retrieval payloads are JSON encoded for stores whose metadata
model accepts only scalar values, then decoded before results enter the
grounding pipeline.

Official SDK references used by the adapters:

- [Qdrant Python client](https://python-client.qdrant.tech/)
- [Milvus collection creation](https://milvus.io/docs/create-collection.md)
- [Milvus upsert](https://milvus.io/docs/upsert-entities.md)
- [Weaviate self-provided vectors](https://docs.weaviate.io/weaviate/manage-collections/vector-config)
- [Weaviate vector similarity search](https://docs.weaviate.io/weaviate/search/similarity)
- [Chroma Python client](https://docs.trychroma.com/reference/python/client)
- [Chroma collection API](https://docs.trychroma.com/reference/python/collection)
- [pgvector cosine distance](https://github.com/pgvector/pgvector#querying)
- [Pinecone index creation](https://docs.pinecone.io/guides/index-data/create-an-index)
- [Pinecone Python SDK](https://sdk.pinecone.io/python/reference/pinecone.html)

## Web Grounding Providers

Search:

- Tavily (`tavily`)
- Exa Search (`exa`)

Reader/content:

- Jina Reader (`jina`)
- Exa Contents (`exa`)

Mode selector:

- `WEB_SEARCH_MODE=auto`
- `WEB_SEARCH_MODE=tavily_jina`
- `WEB_SEARCH_MODE=exa`

## Governance Configuration

Policy controls are loaded from `GOVERNANCE_*` variables and `risk_threshold_for_human_review`:

- overall risk threshold
- recommendation confidence threshold
- evidence completeness threshold
- contradiction severity threshold
- unresolved question count threshold
- high-stakes category list
- manual approval policy by task type
