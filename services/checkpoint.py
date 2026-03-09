from __future__ import annotations

import logging
from pathlib import Path

from schemas.models import (
    ApprovalDecision,
    ArtifactRecord,
    CitationRecord,
    ConflictRecord,
    CoverageRecord,
    ErrorInfo,
    EvidenceAssessment,
    EvidenceRecord,
    FindingRecord,
    GovernanceEvaluation,
    PlanStep,
    RecommendationRecord,
    ReviewFeedback,
    SourceRecord,
    SupportRecord,
    ToolCallRecord,
    TraceEvent,
)
from services.config import Settings


def build_checkpointer(settings: Settings):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    allowed_modules = [
        ApprovalDecision,
        ArtifactRecord,
        CitationRecord,
        ConflictRecord,
        CoverageRecord,
        ErrorInfo,
        EvidenceAssessment,
        EvidenceRecord,
        FindingRecord,
        GovernanceEvaluation,
        PlanStep,
        RecommendationRecord,
        ReviewFeedback,
        SourceRecord,
        SupportRecord,
        ToolCallRecord,
        TraceEvent,
    ]
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=allowed_modules)
    except TypeError:  # pragma: no cover - compatibility across langgraph versions
        serde = JsonPlusSerializer()

    if settings.checkpoint_mode.lower() == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpoint_path = Path(settings.output_root) / "checkpoints.sqlite"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver.from_conn_string(str(checkpoint_path), serde=serde)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logging.getLogger("agentops").warning(
                "Falling back to MemorySaver because SQLite checkpointing failed: %s",
                exc,
            )

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=serde)
