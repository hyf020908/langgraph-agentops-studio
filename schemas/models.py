from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["vector", "web_search", "webpage", "unknown"]
ProviderType = Literal[
    "qdrant",
    "tavily",
    "jina",
    "exa",
    "openai",
    "deepseek",
    "openai_compatible",
    "memory",
    "unknown",
]


class CitationRecord(BaseModel):
    source_id: str
    provider: ProviderType | str = "unknown"
    title: str
    source: str
    url: str | None = None
    chunk_id: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieved_at: str | None = None


class SearchResult(BaseModel):
    source_id: str
    provider: ProviderType | str = "unknown"
    source_type: SourceType = "unknown"
    title: str
    url: str
    source: str | None = None
    domain: str = "unknown"
    snippet: str
    content: str | None = None
    published_at: str = "unknown"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    chunk_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    step_id: str
    objective: str
    owner: str
    done_definition: str
    dependencies: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    source_id: str
    provider: ProviderType | str = "unknown"
    source_type: SourceType = "unknown"
    title: str
    url: str
    source: str = ""
    domain: str
    snippet: str
    content: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    parsed_sections: list[str] = Field(default_factory=list)
    published_at: str = "unknown"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceScoreBreakdown(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    source_credibility: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    corroboration: float = Field(ge=0.0, le=1.0)
    contradiction_penalty: float = Field(ge=0.0, le=1.0)
    extraction_quality: float = Field(ge=0.0, le=1.0)
    actionability: float = Field(ge=0.0, le=1.0)


class ConflictRecord(BaseModel):
    left_source_id: str
    right_source_id: str
    severity: float = Field(ge=0.0, le=1.0)
    reason: str


class SupportRecord(BaseModel):
    source_id: str
    supports_source_id: str
    strength: float = Field(ge=0.0, le=1.0)
    reason: str


class CoverageRecord(BaseModel):
    query_coverage: float = Field(ge=0.0, le=1.0)
    criteria_coverage: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    coverage_notes: list[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    source_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    score_breakdown: EvidenceScoreBreakdown
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str
    claim: str
    supporting_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    summary: str
    citations: list[CitationRecord] = Field(default_factory=list)
    assessment: EvidenceAssessment | None = None


class FindingRecord(BaseModel):
    finding_id: str
    theme: str
    insight: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"


class RecommendationRecord(BaseModel):
    recommendation_type: Literal["insufficient_evidence", "conditional", "directional"]
    summary: str
    rationale: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)


class GovernanceEvaluation(BaseModel):
    requires_human_review: bool
    triggered_policies: list[str] = Field(default_factory=list)
    risk_summary: str
    evidence_gaps: list[str] = Field(default_factory=list)
    contradiction_summary: str
    recommendation_confidence: float = Field(ge=0.0, le=1.0)
    required_human_action: str
    overall_risk_score: float = Field(ge=0.0, le=1.0)


class ReviewFeedback(BaseModel):
    verdict: Literal["approve", "revise", "escalate"]
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str
    questions: list[str] = Field(default_factory=list)
    revision_requests: list[str] = Field(default_factory=list)
    major_risks: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool_name: str
    status: Literal["success", "error"]
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""
    error: str | None = None


class TraceEvent(BaseModel):
    timestamp: str
    node: str
    status: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    name: str
    path: str
    media_type: str


class ErrorInfo(BaseModel):
    stage: str
    message: str
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str
    rationale: str


class DecisionRecord(BaseModel):
    task_id: str
    recommendation: str
    confidence: float
    risks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    approval: ApprovalDecision | None = None


class RunRequest(BaseModel):
    task: str
    task_id: str | None = None
    auto_approve: bool = False
    task_type: str = "general"


class ContinueRequest(BaseModel):
    approved: bool
    reviewer: str = "api-reviewer"
    rationale: str = "Continuation requested from API."


class IngestRequest(BaseModel):
    source_dir: str | None = None
    recreate_collection: bool = False


class IngestResponse(BaseModel):
    status: str
    source_dir: str
    collection: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    embedding_dimensions: int | None = None
    vector_store_provider: str | None = None
    message: str | None = None


class RunResponse(BaseModel):
    task_id: str
    status: str
    approval_required: bool
    approval_payload: dict[str, Any] | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    review_summary: str | None = None
