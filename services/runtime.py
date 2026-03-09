from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from schemas.models import TraceEvent
from services.config import Settings, load_settings
from services.embeddings import BaseEmbeddingProvider, build_embedding_provider
from services.evidence import EvidencePipeline
from services.governance import GovernanceService
from services.llm import BaseLLMProvider, build_llm_provider, build_reasoning_engine
from services.logging import configure_logging
from services.recommendation import RecommendationService
from services.retrieval import RetrievalService
from services.storage import LocalArtifactStore
from services.vectorstore import BaseVectorStore, build_vector_store
from services.web_grounding import ResearchGroundingService
from services.web_reader import BaseWebReaderProvider, build_web_reader_provider
from services.web_search import BaseWebSearchProvider, build_web_search_provider


@dataclass(slots=True)
class AgentRuntime:
    settings: Settings
    storage: LocalArtifactStore
    llm_provider: BaseLLMProvider
    embedding_provider: BaseEmbeddingProvider
    vector_store: BaseVectorStore
    retrieval: RetrievalService
    web_search_provider: BaseWebSearchProvider | None
    web_reader_provider: BaseWebReaderProvider | None
    grounding: ResearchGroundingService
    evidence_pipeline: EvidencePipeline
    recommendation_service: RecommendationService
    governance_service: GovernanceService
    reasoning: Any
    logger: Any

    def trace(self, node: str, status: str, message: str, metadata: dict[str, Any] | None = None) -> TraceEvent:
        return TraceEvent(
            timestamp=datetime.now(UTC).isoformat(),
            node=node,
            status=status,
            message=message,
            metadata=metadata or {},
        )


def build_runtime(settings: Settings | None = None) -> AgentRuntime:
    resolved_settings = settings or load_settings()
    logger = configure_logging(resolved_settings.log_level)
    storage = LocalArtifactStore(resolved_settings.output_root)

    llm_provider = build_llm_provider(resolved_settings.llm)
    embedding_provider = build_embedding_provider(resolved_settings.embedding)
    vector_store = build_vector_store(resolved_settings.vector_db)

    retrieval = RetrievalService(
        rag_settings=resolved_settings.rag,
        embeddings=embedding_provider,
        vector_store=vector_store,
    )

    web_search_provider = build_web_search_provider(resolved_settings)
    web_reader_provider = build_web_reader_provider(resolved_settings)

    grounding = ResearchGroundingService(
        settings=resolved_settings,
        retrieval=retrieval,
        web_search_provider=web_search_provider,
        web_reader_provider=web_reader_provider,
        logger=logger,
    )

    return AgentRuntime(
        settings=resolved_settings,
        storage=storage,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        retrieval=retrieval,
        web_search_provider=web_search_provider,
        web_reader_provider=web_reader_provider,
        grounding=grounding,
        evidence_pipeline=EvidencePipeline(),
        recommendation_service=RecommendationService(),
        governance_service=GovernanceService(resolved_settings.governance),
        reasoning=build_reasoning_engine(llm_provider),
        logger=logger,
    )
