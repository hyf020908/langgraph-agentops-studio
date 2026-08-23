# Vector RAG and Grounding

## Ingestion Flow

`services/retrieval.py` ingests local documents by:

1. reading supported files from `RAG_SOURCE_DIR`
2. chunking text with configured chunk size and overlap
3. generating embeddings through the configured provider
4. upserting vectors into the configured vector store

Run ingestion:

```bash
python app/ingest.py --source-dir examples/knowledge_base --recreate-collection
```

## Retrieval Flow

During research grounding, vector retrieval returns normalized records with:

- `source_type=vector`
- provider metadata
- title, snippet, content
- score and retrieval timestamp

## Hybrid Grounding

`services/web_grounding.py` merges:

- vector retrieval output
- web search output
- reader/content extraction output

Then it applies dedupe and ranking to produce a unified source list for evidence scoring.

## Configuration Notes

- `ENABLE_VECTOR_RAG=true` enables vector retrieval.
- `VECTOR_DB_PROVIDER` selects `qdrant`, `milvus`, `weaviate`, `chroma`,
  `pgvector`, `pinecone`, or the non-persistent `memory` test backend.
- `RAG_TOP_K` controls vector hit count.
- `RAG_SCORE_THRESHOLD` can be set for score filtering.
- `EVIDENCE_MERGE_STRATEGY` controls ranking behavior across source types.

Every provider returns cosine similarity with higher scores ranked first. On
ingestion, stores with fixed vector dimensions reject a collection created for
a different embedding size. Use `--recreate-collection` only when replacing the
existing corpus is intended. Provider SDKs other than the default Qdrant client
are installed through the optional extras documented in `docs/providers.md`.
