from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from schemas.models import SearchResult
from services.config import Settings
from services.retrieval import RetrievalService
from services.web_reader import BaseWebReaderProvider
from services.web_search import BaseWebSearchProvider


class ResearchGroundingService:
    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        web_search_provider: BaseWebSearchProvider | None,
        web_reader_provider: BaseWebReaderProvider | None,
        logger: Any,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.web_search_provider = web_search_provider
        self.web_reader_provider = web_reader_provider
        self.logger = logger

    def ground_query(self, query: str) -> dict[str, Any]:
        web_enabled = self.settings.web_grounding.enable_web_search
        vector_enabled = self.settings.web_grounding.enable_vector_rag

        if not web_enabled and not vector_enabled:
            raise RuntimeError("Both ENABLE_WEB_SEARCH and ENABLE_VECTOR_RAG are disabled; no grounding source available.")

        merged: list[SearchResult] = []
        vector_count = 0
        web_count = 0
        page_count = 0

        if vector_enabled:
            vector_results = self.retrieval.search(query=query, top_k=self.settings.rag.top_k)
            for item in vector_results:
                enriched = {
                    **item,
                    "source_type": "vector",
                    "provider": item.get("provider", self.settings.vector_db.provider),
                    "retrieved_at": item.get("retrieved_at") or datetime.now(UTC).isoformat(),
                }
                merged.append(SearchResult.model_validate(enriched))
            vector_count = len(vector_results)

        if web_enabled:
            if self.web_search_provider is None:
                raise RuntimeError("ENABLE_WEB_SEARCH=true but no real web search provider is configured.")

            max_results = min(self.settings.web_grounding.web_search_top_k, self._provider_max_results())
            web_results = self.web_search_provider.search(query=query, max_results=max_results)
            web_count = len(web_results)

            result_by_url = {item.url: item for item in web_results if item.url}
            if self.web_reader_provider is not None and result_by_url:
                target_urls = list(result_by_url.keys())[: self.settings.web_grounding.web_reader_top_k]
                page_contents = self.web_reader_provider.read_urls(target_urls)
                page_count = len(page_contents)
                for url, page in page_contents.items():
                    if url not in result_by_url:
                        continue
                    source = result_by_url[url]
                    source.content = page.content
                    source.source_type = "webpage"
                    source.metadata = {
                        **source.metadata,
                        "reader_provider": page.provider,
                        "reader_retrieved_at": page.retrieved_at,
                    }
                    if page.title:
                        source.title = page.title
                    if source.snippet.strip() == "" and page.content.strip():
                        source.snippet = page.content[:500]

            merged.extend(web_results)

        deduped = self._dedupe(merged)
        ranked = self._rank(deduped)

        return {
            "query": query,
            "results": [item.model_dump() for item in ranked],
            "stats": {
                "vector_count": vector_count,
                "web_count": web_count,
                "webpage_count": page_count,
                "merged_count": len(ranked),
            },
            "providers": {
                "vector": self.settings.vector_db.provider if vector_enabled else None,
                "web_search": self.web_search_provider.name if self.web_search_provider else None,
                "web_reader": self.web_reader_provider.name if self.web_reader_provider else None,
            },
        }

    def _provider_max_results(self) -> int:
        if self.web_search_provider is None:
            return self.settings.web_grounding.web_search_top_k
        if self.web_search_provider.name == "tavily":
            return self.settings.tavily.max_results
        if self.web_search_provider.name == "exa":
            return self.settings.exa.max_results
        return self.settings.web_grounding.web_search_top_k

    @staticmethod
    def _dedupe(items: list[SearchResult]) -> list[SearchResult]:
        deduped: dict[str, SearchResult] = {}
        for item in items:
            key = f"{item.source_type}|{item.url}|{item.chunk_id or ''}|{item.provider}"
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                continue
            existing_score = existing.score if existing.score is not None else existing.credibility
            current_score = item.score if item.score is not None else item.credibility
            if current_score > existing_score:
                deduped[key] = item
        return list(deduped.values())

    def _rank(self, items: list[SearchResult]) -> list[SearchResult]:
        strategy = self.settings.web_grounding.evidence_merge_strategy

        def score_for(item: SearchResult) -> float:
            base = item.score if item.score is not None else item.credibility
            if strategy == "source_priority":
                if item.source_type == "vector":
                    return base + 0.08
                if item.source_type == "webpage":
                    return base + 0.05
                return base
            if item.source_type == "webpage":
                return base + 0.03
            return base

        ranked = sorted(items, key=score_for, reverse=True)
        for index, item in enumerate(ranked, start=1):
            if not item.source_id:
                item.source_id = f"SRC-{index:03d}-{uuid4().hex[:6]}"
        return ranked
