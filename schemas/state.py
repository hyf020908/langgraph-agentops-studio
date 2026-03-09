from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from schemas.models import (
    ApprovalDecision,
    ArtifactRecord,
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


def append_records(left: list | None, right: list | None) -> list:
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    user_request: str
    task_type: str
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    plan: list[PlanStep]
    acceptance_criteria: list[str]
    search_queries: list[str]
    retrieved_sources: list[SourceRecord]
    retrieved_chunks: list[SourceRecord]
    ranked_evidence: list[EvidenceRecord]
    evidence_assessments: list[EvidenceAssessment]
    evidence_conflicts: list[ConflictRecord]
    evidence_supports: list[SupportRecord]
    coverage_record: CoverageRecord | None
    findings: list[FindingRecord]
    recommendation: RecommendationRecord | None
    governance_evaluation: GovernanceEvaluation | None
    draft_report: str
    review_feedback: ReviewFeedback | None
    reviewer_history: Annotated[list[ReviewFeedback], append_records]
    revision_count: int
    tool_call_history: Annotated[list[ToolCallRecord], append_records]
    execution_trace: Annotated[list[TraceEvent], append_records]
    artifacts: list[ArtifactRecord]
    human_approval_required: bool
    approval_decision: ApprovalDecision | None
    status: str
    error_info: ErrorInfo | None
    retry_count: int
    next_step: str
    sender: str


def initial_state(user_request: str, task_id: str | None = None, task_type: str = "general") -> AgentState:
    run_id = task_id or f"task-{uuid4().hex[:10]}"
    now = datetime.now(UTC).isoformat()
    return {
        "user_request": user_request,
        "task_type": task_type,
        "messages": [HumanMessage(content=user_request)],
        "task_id": run_id,
        "plan": [],
        "acceptance_criteria": [],
        "search_queries": [],
        "retrieved_sources": [],
        "retrieved_chunks": [],
        "ranked_evidence": [],
        "evidence_assessments": [],
        "evidence_conflicts": [],
        "evidence_supports": [],
        "coverage_record": None,
        "findings": [],
        "recommendation": None,
        "governance_evaluation": None,
        "draft_report": "",
        "review_feedback": None,
        "reviewer_history": [],
        "revision_count": 0,
        "tool_call_history": [],
        "execution_trace": [
            TraceEvent(
                timestamp=now,
                node="bootstrap",
                status="created",
                message="Initialized task state.",
                metadata={"task_id": run_id},
            )
        ],
        "artifacts": [],
        "human_approval_required": False,
        "approval_decision": None,
        "status": "initialized",
        "error_info": None,
        "retry_count": 0,
        "next_step": "planner_agent",
        "sender": "user",
    }
