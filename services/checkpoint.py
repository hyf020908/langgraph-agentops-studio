from __future__ import annotations

import sqlite3
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
    ModelCallRecord,
    PlanStep,
    RecommendationRecord,
    ReviewFeedback,
    SourceRecord,
    SupportRecord,
    ToolCallRecord,
    TraceEvent,
    ContextArtifactPointer,
    ContextCompactionEvent,
    EvidenceCompactionMap,
    EvidenceDigest,
    RevisionLedger,
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
        ModelCallRecord,
        PlanStep,
        RecommendationRecord,
        ReviewFeedback,
        SourceRecord,
        SupportRecord,
        ToolCallRecord,
        TraceEvent,
        ContextArtifactPointer,
        ContextCompactionEvent,
        EvidenceCompactionMap,
        EvidenceDigest,
        RevisionLedger,
    ]
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=allowed_modules)
    except TypeError:  # pragma: no cover - compatibility across langgraph versions
        serde = JsonPlusSerializer()

    if settings.checkpoint_mode.lower() == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_path = Path(settings.output_root) / "checkpoints.sqlite"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            saver = SqliteSaver(conn, serde=serde)
            saver.setup()
            return saver
        except Exception:
            conn.close()
            raise

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=serde)
