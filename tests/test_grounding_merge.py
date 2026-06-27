from __future__ import annotations

from schemas.models import SearchResult
from services.config import Settings, WebGroundingSettings
from services.web_grounding import ResearchGroundingService


def _build_service(strategy: str) -> ResearchGroundingService:
    service = object.__new__(ResearchGroundingService)
    service.settings = Settings(web_grounding=WebGroundingSettings(evidence_merge_strategy=strategy))
    return service


def test_dedupe_keeps_higher_scored_record() -> None:
    records = [
        SearchResult(
            source_id="A-1",
            provider="tavily",
            source_type="web_search",
            title="Record A",
            url="https://example.com/a",
            source="https://example.com/a",
            domain="example.com",
            snippet="one",
            score=0.4,
            credibility=0.4,
        ),
        SearchResult(
            source_id="A-2",
            provider="tavily",
            source_type="web_search",
            title="Record A Updated",
            url="https://example.com/a",
            source="https://example.com/a",
            domain="example.com",
            snippet="two",
            score=0.8,
            credibility=0.8,
        ),
    ]

    deduped = ResearchGroundingService._dedupe(records)
    assert len(deduped) == 1
    assert deduped[0].source_id == "A-2"


def test_rank_respects_source_priority_strategy() -> None:
    service = _build_service("source_priority")
    records = [
        SearchResult(
            source_id="SRC-WEB",
            provider="tavily",
            source_type="web_search",
            title="Web",
            url="https://example.com/web",
            source="https://example.com/web",
            domain="example.com",
            snippet="web",
            score=0.62,
            credibility=0.62,
        ),
        SearchResult(
            source_id="SRC-VEC",
            provider="qdrant",
            source_type="vector",
            title="Vector",
            url="local://vector",
            source="local://vector",
            domain="local",
            snippet="vector",
            score=0.6,
            credibility=0.6,
        ),
    ]

    ranked = service._rank(records)
    assert ranked[0].source_type == "vector"


def test_grounding_continues_when_reader_provider_fails() -> None:
    class Retrieval:
        def search(self, query: str, top_k: int) -> list[dict]:
            return []

    class SearchProvider:
        name = "tavily"

        def search(self, query: str, max_results: int) -> list[SearchResult]:
            return [
                SearchResult(
                    source_id="SRC-WEB",
                    provider="tavily",
                    source_type="web_search",
                    title="Web",
                    url="https://example.com/web",
                    source="https://example.com/web",
                    domain="example.com",
                    snippet="web snippet",
                    score=0.62,
                    credibility=0.62,
                )
            ]

    class ReaderProvider:
        name = "jina"

        def read_urls(self, urls: list[str]):
            raise RuntimeError("reader blocked")

    class Logger:
        def warning(self, *args, **kwargs) -> None:
            pass

    settings = Settings(
        web_grounding=WebGroundingSettings(enable_web_search=True, enable_vector_rag=False, web_search_top_k=1)
    )
    service = ResearchGroundingService(
        settings=settings,
        retrieval=Retrieval(),
        web_search_provider=SearchProvider(),
        web_reader_provider=ReaderProvider(),
        logger=Logger(),
    )

    report = service.ground_query("query")

    assert report["stats"]["web_count"] == 1
    assert report["stats"]["webpage_count"] == 0
    assert report["results"][0]["source_type"] == "web_search"
