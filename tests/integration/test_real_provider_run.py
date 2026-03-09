from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.runner import WorkflowRunner
from services.config import (
    EmbeddingSettings,
    ExaSettings,
    LLMSettings,
    RAGSettings,
    Settings,
    TavilySettings,
    VectorDBSettings,
    WebGroundingSettings,
)
from services.runtime import build_runtime


def _ensure_enabled() -> None:
    if os.getenv("RUN_REAL_PROVIDER_RUN", "").strip() != "1":
        pytest.skip("Set RUN_REAL_PROVIDER_RUN=1 to run provider-backed integration test.")


def _resolve_mode() -> str:
    explicit = os.getenv("WEB_SEARCH_MODE", "").strip().lower()
    if explicit in {"exa", "tavily_jina"}:
        return explicit
    if os.getenv("TAVILY_API_KEY"):
        return "tavily_jina"
    if os.getenv("EXA_API_KEY"):
        return "exa"
    pytest.skip("No web search key available for provider-backed integration test.")


def _require_provider_env(mode: str) -> None:
    if not os.getenv("LLM_API_KEY"):
        pytest.skip("LLM_API_KEY is required.")
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("EMBEDDING_API_KEY is required.")
    if mode == "tavily_jina" and not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY is required for tavily_jina mode.")
    if mode == "exa" and not os.getenv("EXA_API_KEY"):
        pytest.skip("EXA_API_KEY is required for exa mode.")


def test_real_provider_backed_run_with_vector_and_web_grounding() -> None:
    _ensure_enabled()
    mode = _resolve_mode()
    _require_provider_env(mode)

    with TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir)
        (source_dir / "policy_notes.md").write_text(
            (
                "# Governance Checklist\n"
                "Use explicit policy thresholds for recommendation confidence, contradiction severity, and unresolved questions.\n"
                "Require manual approval for high-stakes architecture and security decisions.\n"
            ),
            encoding="utf-8",
        )

        settings = Settings(
            app_name="LangGraph AgentOps Studio",
            output_root="runs",
            max_search_results=3,
            llm=LLMSettings(
                provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL") or None,
                temperature=0.1,
                max_tokens=900,
            ),
            embedding=EmbeddingSettings(
                provider=os.getenv("EMBEDDING_PROVIDER", "openai_compatible"),
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
                base_url=os.getenv("EMBEDDING_BASE_URL") or None,
                dimensions=1024,
            ),
            vector_db=VectorDBSettings(provider="memory"),
            rag=RAGSettings(
                enabled=True,
                source_dir=str(source_dir),
                top_k=2,
                chunk_size=400,
                chunk_overlap=80,
            ),
            web_grounding=WebGroundingSettings(
                mode=mode,
                enable_web_search=True,
                enable_vector_rag=True,
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

        runtime = build_runtime(settings)
        ingest_report = runtime.retrieval.ingest_directory(source_dir=source_dir, recreate_collection=True)
        assert ingest_report["status"] == "ok"
        assert ingest_report["chunk_count"] >= 1

        runner = WorkflowRunner(runtime=runtime)
        state, interrupt_payload = runner.start(
            task="Assess governance controls for an architecture rollout under regulatory constraints.",
            task_type="architecture",
            auto_approve=True,
        )

    assert interrupt_payload is None
    assert state.get("status") == "completed"
    assert state.get("recommendation") is not None
    assert state.get("governance_evaluation") is not None
    assert len(state.get("ranked_evidence", [])) >= 1
