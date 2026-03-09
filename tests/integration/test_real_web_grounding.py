from __future__ import annotations

import os

import pytest

from services.config import (
    ExaSettings,
    RAGSettings,
    Settings,
    TavilySettings,
    VectorDBSettings,
    WebGroundingSettings,
)
from services.logging import configure_logging
from services.retrieval import RetrievalService
from services.vectorstore import build_vector_store
from services.web_grounding import ResearchGroundingService
from services.web_reader import build_web_reader_provider
from services.web_search import build_web_search_provider


def _resolve_mode() -> str:
    explicit = os.getenv("WEB_SEARCH_MODE", "").strip().lower()
    if explicit in {"exa", "tavily_jina"}:
        return explicit
    if os.getenv("TAVILY_API_KEY"):
        return "tavily_jina"
    if os.getenv("EXA_API_KEY"):
        return "exa"
    pytest.skip("No web search key available for integration test.")


def _ensure_enabled() -> None:
    if os.getenv("RUN_REAL_WEB_GROUNDING", "").strip() != "1":
        pytest.skip("Set RUN_REAL_WEB_GROUNDING=1 to run real web grounding integration test.")


def test_real_web_grounding_path() -> None:
    _ensure_enabled()
    mode = _resolve_mode()
    if mode == "tavily_jina" and not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY is required for tavily_jina mode.")
    if mode == "exa" and not os.getenv("EXA_API_KEY"):
        pytest.skip("EXA_API_KEY is required for exa mode.")

    settings = Settings(
        output_root="runs",
        vector_db=VectorDBSettings(provider="memory"),
        rag=RAGSettings(enabled=False),
        web_grounding=WebGroundingSettings(
            mode=mode,
            enable_web_search=True,
            enable_vector_rag=False,
            web_search_top_k=3,
            web_reader_top_k=2,
        ),
        tavily=TavilySettings(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=3,
            topic="general",
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        ),
        exa=ExaSettings(
            api_key=os.getenv("EXA_API_KEY"),
            max_results=3,
            search_type=os.getenv("EXA_SEARCH_TYPE", "neural"),
            base_url=os.getenv("EXA_BASE_URL", "https://api.exa.ai"),
            use_contents=True,
        ),
    )

    retrieval = RetrievalService(
        rag_settings=settings.rag,
        embeddings=None,  # not used when rag.enabled is false
        vector_store=build_vector_store(settings.vector_db),
    )
    grounding = ResearchGroundingService(
        settings=settings,
        retrieval=retrieval,
        web_search_provider=build_web_search_provider(settings),
        web_reader_provider=build_web_reader_provider(settings),
        logger=configure_logging("INFO"),
    )

    report = grounding.ground_query("LangGraph policy driven human approval workflow patterns")
    results = report.get("results", [])
    providers = report.get("providers", {})

    assert isinstance(results, list)
    assert len(results) >= 1
    assert providers.get("web_search") in {"tavily", "exa"}
    assert providers.get("web_reader") in {"jina", "exa"}
