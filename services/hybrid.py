"""Persistent SQLite FTS5 BM25, rank fusion and bounded second-stage reranking."""
import json
import logging
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx

from services.config import RAGSettings

logger = logging.getLogger("agentops.retrieval")


def tokenize(text: str) -> list[str]:
    # FTS5's default tokenizer does not segment unspaced Chinese. Index/query
    # identically with CJK characters and bigrams; retain Latin technical terms.
    tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", text):
        tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return tokens


class BM25Index:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(point_id UNINDEXED, terms, payload UNINDEXED)")
            conn.execute("CREATE TABLE IF NOT EXISTS chunk_keys(id INTEGER PRIMARY KEY, point_id TEXT UNIQUE NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def upsert(self, chunks: list[dict], recreate=False):
        with closing(self.connect()) as conn, conn:
            if recreate:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM chunk_keys")
            for chunk in chunks:
                conn.execute("INSERT OR IGNORE INTO chunk_keys(point_id) VALUES (?)", (chunk["point_id"],))
                rowid = conn.execute("SELECT id FROM chunk_keys WHERE point_id=?", (chunk["point_id"],)).fetchone()[0]
                conn.execute("DELETE FROM chunks WHERE rowid = ?", (rowid,))
                conn.execute("INSERT INTO chunks(rowid, point_id, terms, payload) VALUES (?, ?, ?, ?)",
                             (rowid, chunk["point_id"], " ".join(tokenize(chunk["text"])), json.dumps(chunk["payload"], ensure_ascii=False)))

    def search(self, query: str, top_k: int) -> list[dict]:
        terms = list(dict.fromkeys(tokenize(query)))[:64]
        if not terms:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT point_id, payload, bm25(chunks) AS score FROM chunks WHERE chunks MATCH ? ORDER BY score, point_id LIMIT ?", (match, top_k)).fetchall()
        return [{"id": row[0], "payload": json.loads(row[1]), "score": -row[2]} for row in rows]


def reciprocal_rank_fusion(rankings: list[list[dict]], k=60) -> list[dict]:
    fused: dict[str, dict] = {}
    for lane, ranking in enumerate(rankings):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            identifier = str(hit.get("id") or hit["payload"].get("chunk_id"))
            if identifier in seen:
                continue
            seen.add(identifier)
            item = fused.setdefault(identifier, {**hit, "score": 0.0, "retrieval_ranks": {}})
            item["score"] += 1 / (k + rank)
            item["retrieval_ranks"][str(lane)] = rank
    return sorted(fused.values(), key=lambda hit: (-hit["score"], str(hit.get("id", ""))))


class Reranker:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.client = httpx.Client(timeout=settings.rerank_timeout, limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
        if settings.rerank_provider == "http" and not settings.rerank_url:
            raise ValueError("RAG_RERANK_URL is required for HTTP reranking.")

    def close(self):
        self.client.close()

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits or self.settings.rerank_provider == "none":
            return hits[:top_k]
        if self.settings.rerank_provider == "lexical":
            # Explicitly a local heuristic, not a learned semantic cross-encoder.
            terms = set(tokenize(query))
            for hit in hits:
                text = str(hit["payload"].get("text", ""))
                coverage = len(terms & set(tokenize(text))) / max(1, len(terms))
                hit["rerank_score"] = coverage + hit["score"]
            return sorted(hits, key=lambda hit: (-hit["rerank_score"], -hit["score"]))[:top_k]
        try:
            headers = {"Authorization": f"Bearer {self.settings.rerank_api_key}"} if self.settings.rerank_api_key else {}
            response = self.client.post(self.settings.rerank_url, headers=headers, json={
                "model": self.settings.rerank_model, "query": query,
                "documents": [str(hit["payload"].get("text", ""))[:8000] for hit in hits],
                "top_n": top_k,
            })
            response.raise_for_status()
            results = response.json()["results"]
            indexes = [item["index"] for item in results]
            if len(results) != min(top_k, len(hits)) or len(set(indexes)) != len(indexes):
                raise ValueError("Incomplete or duplicate rerank results.")
            output = []
            for item in results:
                index, score = item["index"], float(item["relevance_score"])
                if type(index) is not int or not 0 <= index < len(hits) or not math.isfinite(score):
                    raise ValueError("Invalid rerank result.")
                output.append({**hits[index], "rerank_score": score})
            return sorted(output, key=lambda hit: -hit["rerank_score"])
        except Exception:
            logger.warning("Rerank unavailable; using RRF order.", exc_info=True)
            return hits[:top_k]
