from __future__ import annotations

# Local retrieval and ingestion service.
# This module owns knowledge-base ingestion into the vector store and similarity
# search over embedded chunks. It is the vector-backed half of hybrid grounding.

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from services.config import RAGSettings
from services.embeddings import BaseEmbeddingProvider
from services.vectorstore import BaseVectorStore
from services.hybrid import BM25Index, Reranker, reciprocal_rank_fusion


SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv"}


class RetrievalService:
    def __init__(
        self,
        *,
        rag_settings: RAGSettings,
        embeddings: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> None:
        self.settings = rag_settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.lexical = BM25Index(rag_settings.lexical_path) if rag_settings.hybrid_enabled else None
        self.reranker = Reranker(rag_settings)

    def ingest_directory(self, source_dir: str | Path | None = None, recreate_collection: bool = False) -> dict[str, Any]:
        directory = Path(source_dir or self.settings.source_dir)
        if not directory.exists() or not directory.is_dir():
            return {
                "status": "skipped",
                "collection": getattr(self.vector_store, "_collection", None),
                "source_dir": str(directory),
                "document_count": 0,
                "chunk_count": 0,
                "message": "Source directory does not exist.",
            }

        documents = self._load_documents(directory)
        if not documents:
            return {
                "status": "skipped",
                "collection": getattr(self.vector_store, "_collection", None),
                "source_dir": str(directory),
                "document_count": 0,
                "chunk_count": 0,
                "message": "No supported files found for ingestion.",
            }

        chunks = self._split_documents(documents)
        if not chunks:
            return {
                "status": "skipped",
                "collection": getattr(self.vector_store, "_collection", None),
                "source_dir": str(directory),
                "document_count": len(documents),
                "chunk_count": 0,
                "message": "No text chunks generated from source documents.",
            }

        # Lexical evidence remains searchable during embedding/provider outages.
        if self.lexical is not None:
            self.lexical.upsert(chunks, recreate=recreate_collection)
        dimensions = None
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            vectors = self.embeddings.embed_documents([item["text"] for item in batch])
            if len(vectors) != len(batch) or any(not vector for vector in vectors):
                raise RuntimeError("Embedding response does not match ingestion batch.")
            if dimensions is None:
                dimensions = len(vectors[0])
                self.vector_store.ensure_collection(vector_size=dimensions, recreate=recreate_collection)
            if any(len(vector) != dimensions for vector in vectors):
                raise RuntimeError("Embedding dimensions changed during ingestion.")
            self.vector_store.upsert(ids=[item["point_id"] for item in batch], vectors=vectors,
                                     payloads=[item["payload"] for item in batch])
        return {
            "status": "ok",
            "collection": getattr(self.vector_store, "_collection", None),
            "source_dir": str(directory),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_dimensions": dimensions,
            "vector_store_provider": self.vector_store.provider_name,
        }

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.settings.enabled:
            return []

        limit = min(20, max(1, top_k or self.settings.top_k))
        candidates = max(limit, self.settings.candidate_k)
        lexical_hits = []
        if self.lexical is not None:
            try:
                lexical_hits = self.lexical.search(query, candidates)
            except Exception:
                logging.getLogger("agentops.retrieval").warning("BM25 unavailable; using vector recall.", exc_info=True)

        try:
            # Retrieval failures are intentionally swallowed here so grounding can
            # still continue with web sources when vector infrastructure is down.
            query_vector = self.embeddings.embed_query(query)
            search_hits = self.vector_store.search(
                query_vector=query_vector,
                top_k=candidates,
                score_threshold=self.settings.score_threshold,
            )
        except Exception:
            logging.getLogger("agentops.retrieval").warning("Vector recall unavailable; continuing with BM25 candidates.", exc_info=True)
            search_hits = []

        search_hits = reciprocal_rank_fusion([search_hits, lexical_hits], self.settings.rrf_k)
        search_hits = self.reranker.rerank(query, search_hits[:candidates], limit)

        now = datetime.now(UTC).isoformat()
        results: list[dict[str, Any]] = []
        for hit in search_hits:
            payload = hit.get("payload", {})
            source = str(payload.get("source", "local_document"))
            title = str(payload.get("title", "Knowledge Base Document"))
            chunk_id = str(payload.get("chunk_id", "chunk-0"))
            text = str(payload.get("text", ""))
            source_id = str(payload.get("source_id", f"RAG-{uuid4().hex[:8]}"))
            metadata = dict(payload.get("metadata", {}))
            results.append(
                {
                    "source_id": source_id,
                    "provider": self.vector_store.provider_name,
                    "source_type": "vector",
                    "title": title,
                    "url": payload.get("url", source),
                    "snippet": text[:500],
                    "content": text,
                    "domain": payload.get("domain", "local"),
                    "published_at": payload.get("published_at", "unknown"),
                    "credibility": float(payload.get("credibility", 0.82)),
                    "source": source,
                    "chunk_id": chunk_id,
                    "score": float(hit.get("score", 0.0)),
                    "retrieved_at": now,
                    "metadata": {
                        **metadata,
                        "text": text,
                        "retrieval_ranks": hit.get("retrieval_ranks", {}),
                        "rrf_score": hit.get("score", 0),
                        "rerank_score": hit.get("rerank_score"),
                    },
                }
            )
        return results

    def _split_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # Prefer LangChain's splitter when available so chunking behavior
            # matches common retrieval defaults used in production stacks.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            split_chunks: list[dict[str, Any]] = []
            for document in documents:
                text_chunks = splitter.split_text(document["text"])
                for index, chunk_text in enumerate(text_chunks):
                    chunk_id = f"{document['doc_id']}-chunk-{index}"
                    split_chunks.append(
                        {
                            "point_id": self._point_id(document["source"], index),
                            "text": chunk_text,
                            "payload": {
                                "source_id": document["doc_id"],
                                "title": document["title"],
                                "source": document["source"],
                                "url": document["source"],
                                "domain": "local",
                                "published_at": "unknown",
                                "credibility": 0.82,
                                "chunk_id": chunk_id,
                                "text": chunk_text,
                                "metadata": {
                                    "extension": document["extension"],
                                    "source_type": "vector",
                                },
                            },
                        }
                    )
            return split_chunks
        except Exception:
            # The fallback keeps ingestion usable in minimal environments.
            return self._fallback_split_documents(documents)

    def _fallback_split_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        step = max(1, size - overlap)

        for document in documents:
            text = document["text"]
            for index, start in enumerate(range(0, len(text), step)):
                chunk_text = text[start : start + size]
                if not chunk_text.strip():
                    continue
                chunk_id = f"{document['doc_id']}-chunk-{index}"
                chunks.append(
                    {
                        "point_id": self._point_id(document["source"], index),
                        "text": chunk_text,
                        "payload": {
                            "source_id": document["doc_id"],
                            "title": document["title"],
                            "source": document["source"],
                            "url": document["source"],
                            "domain": "local",
                            "published_at": "unknown",
                            "credibility": 0.82,
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "metadata": {
                                "extension": document["extension"],
                                "source_type": "vector",
                            },
                        },
                    }
                )
        return chunks

    def _load_documents(self, directory: Path) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if directory.resolve() not in path.resolve().parents:
                continue
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Knowledge documents must be at most 2 MiB each.")
            if len(documents) >= 1000:
                raise ValueError("Ingest at most 1000 documents per operation.")
            text = self._read_text(path)
            if not text.strip():
                continue
            relative = path.relative_to(directory)
            documents.append(
                {
                    "doc_id": f"DOC-{uuid5(NAMESPACE_URL, str(relative)).hex[:16]}",
                    "title": path.stem.replace("_", " "),
                    "source": str(relative),
                    "extension": path.suffix.lower(),
                    "text": text,
                }
            )
        return documents

    @staticmethod
    def _read_text(path: Path) -> str:
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:
                return path.read_text(encoding="utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _point_id(source: str, chunk_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{source}:{chunk_index}"))
