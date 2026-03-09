from __future__ import annotations

from statistics import mean
from typing import Any

from schemas.models import (
    ApprovalDecision,
    ArtifactRecord,
    ConflictRecord,
    CoverageRecord,
    DecisionRecord,
    EvidenceAssessment,
    EvidenceRecord,
    FindingRecord,
    GovernanceEvaluation,
    RecommendationRecord,
    ReviewFeedback,
    SourceRecord,
    SupportRecord,
    ToolCallRecord,
    TraceEvent,
)
from schemas.state import AgentState
from services.serialization import to_jsonable


def _coerce_review(review: ReviewFeedback | dict[str, Any] | None) -> ReviewFeedback | None:
    if review is None or isinstance(review, ReviewFeedback):
        return review
    return ReviewFeedback.model_validate(review)


def _coerce_recommendation(value: RecommendationRecord | dict[str, Any] | None) -> RecommendationRecord | None:
    if value is None or isinstance(value, RecommendationRecord):
        return value
    return RecommendationRecord.model_validate(value)


def _coerce_governance(value: GovernanceEvaluation | dict[str, Any] | None) -> GovernanceEvaluation | None:
    if value is None or isinstance(value, GovernanceEvaluation):
        return value
    return GovernanceEvaluation.model_validate(value)


def _coerce_sources(state: AgentState) -> list[SourceRecord]:
    return [
        item if isinstance(item, SourceRecord) else SourceRecord.model_validate(item)
        for item in state.get("retrieved_sources", [])
    ]


def _coerce_evidence(state: AgentState) -> list[EvidenceRecord]:
    return [
        item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
        for item in state.get("ranked_evidence", [])
    ]


def _coerce_assessments(state: AgentState) -> list[EvidenceAssessment]:
    return [
        item if isinstance(item, EvidenceAssessment) else EvidenceAssessment.model_validate(item)
        for item in state.get("evidence_assessments", [])
    ]


def _coerce_conflicts(state: AgentState) -> list[ConflictRecord]:
    return [
        item if isinstance(item, ConflictRecord) else ConflictRecord.model_validate(item)
        for item in state.get("evidence_conflicts", [])
    ]


def _coerce_supports(state: AgentState) -> list[SupportRecord]:
    return [
        item if isinstance(item, SupportRecord) else SupportRecord.model_validate(item)
        for item in state.get("evidence_supports", [])
    ]


def _coerce_findings(state: AgentState) -> list[FindingRecord]:
    return [
        item if isinstance(item, FindingRecord) else FindingRecord.model_validate(item)
        for item in state.get("findings", [])
    ]


def _coerce_trace(state: AgentState) -> list[TraceEvent]:
    return [
        item if isinstance(item, TraceEvent) else TraceEvent.model_validate(item)
        for item in state.get("execution_trace", [])
    ]


def _coerce_tool_calls(state: AgentState) -> list[ToolCallRecord]:
    return [
        item if isinstance(item, ToolCallRecord) else ToolCallRecord.model_validate(item)
        for item in state.get("tool_call_history", [])
    ]


def _coerce_artifacts(state: AgentState) -> list[ArtifactRecord]:
    return [
        item if isinstance(item, ArtifactRecord) else ArtifactRecord.model_validate(item)
        for item in state.get("artifacts", [])
    ]


def render_final_report(state: AgentState) -> str:
    findings = _coerce_findings(state)
    evidence = _coerce_evidence(state)
    assessments = _coerce_assessments(state)
    conflicts = _coerce_conflicts(state)
    supports = _coerce_supports(state)
    sources = _coerce_sources(state)
    review = _coerce_review(state.get("review_feedback"))
    recommendation = _coerce_recommendation(state.get("recommendation"))
    governance = _coerce_governance(state.get("governance_evaluation"))
    coverage_payload = state.get("coverage_record")
    coverage = CoverageRecord.model_validate(coverage_payload) if coverage_payload else None

    acceptance_lines = "\n".join(f"- {item}" for item in state.get("acceptance_criteria", [])) or "- none"
    finding_lines = "\n".join(
        f"### {finding.theme}\n"
        f"- Insight: {finding.insight}\n"
        f"- Rationale: {finding.rationale}\n"
        f"- Evidence IDs: {', '.join(finding.evidence_ids)}\n"
        f"- Risk level: {finding.risk_level}"
        for finding in findings
    ) or "No findings available."

    assessment_by_source = {item.source_id: item for item in assessments}
    evidence_lines_list: list[str] = []
    for record in evidence:
        line = (
            f"- `{record.evidence_id}`: {record.claim} "
            f"(confidence={record.confidence:.2f}, sources={', '.join(record.supporting_sources)})"
        )
        assessment = record.assessment
        if assessment is None and record.supporting_sources:
            assessment = assessment_by_source.get(record.supporting_sources[0])
        if assessment is not None:
            breakdown = assessment.score_breakdown
            line += (
                f"\n  - overall_score: {assessment.overall_score:.2f}"
                f"\n  - score_breakdown: relevance={breakdown.relevance:.2f}, "
                f"source_credibility={breakdown.source_credibility:.2f}, recency={breakdown.recency:.2f}, "
                f"completeness={breakdown.completeness:.2f}, corroboration={breakdown.corroboration:.2f}, "
                f"contradiction_penalty={breakdown.contradiction_penalty:.2f}, "
                f"extraction_quality={breakdown.extraction_quality:.2f}, actionability={breakdown.actionability:.2f}"
                f"\n  - strengths: {', '.join(assessment.strengths) or 'none'}"
                f"\n  - weaknesses: {', '.join(assessment.weaknesses) or 'none'}"
                f"\n  - flags: {', '.join(assessment.flags) or 'none'}"
                f"\n  - conflicts_with: {', '.join(assessment.conflicts_with) or 'none'}"
                f"\n  - supports: {', '.join(assessment.supports) or 'none'}"
            )
        evidence_lines_list.append(line)
    evidence_lines = "\n".join(evidence_lines_list)
    if not evidence_lines:
        evidence_lines = "No evidence ledger entries available."

    coverage_block = (
        f"Query Coverage: {coverage.query_coverage:.2f}\n"
        f"Criteria Coverage: {coverage.criteria_coverage:.2f}\n"
        f"Evidence Count: {coverage.evidence_count}\n"
        f"Coverage Notes: {', '.join(coverage.coverage_notes) or 'none'}"
        if coverage
        else "No coverage record available."
    )
    relations_block = (
        f"Support Links: {len(supports)}\n"
        f"Conflict Links: {len(conflicts)}\n"
        f"Conflict Summary: {', '.join(f'{item.left_source_id}->{item.right_source_id}({item.severity:.2f})' for item in conflicts[:5]) or 'none'}"
    )

    source_lines = "\n".join(
        f"- [{source.title}]({source.url}) | provider={source.provider} | source_type={source.source_type} | "
        f"score={source.score if source.score is not None else source.relevance:.2f}"
        for source in sources
    ) or "No source register entries available."

    review_block = render_review_feedback(review) if review else "No reviewer feedback available."
    recommendation_block = _render_recommendation(recommendation)
    governance_block = _render_governance(governance)

    return (
        f"# LangGraph AgentOps Studio Report\n\n"
        f"## Task\n{state['user_request']}\n\n"
        f"## Acceptance Criteria\n{acceptance_lines}\n\n"
        f"## Recommendation\n{recommendation_block}\n\n"
        f"## Governance Evaluation\n{governance_block}\n\n"
        f"## Findings\n{finding_lines}\n\n"
        f"## Evidence Coverage\n{coverage_block}\n\n"
        f"## Evidence Relations\n{relations_block}\n\n"
        f"## Evidence Ledger\n{evidence_lines}\n\n"
        f"## Source Register\n{source_lines}\n\n"
        f"## Reviewer Notes\n{review_block}\n\n"
        f"## Workflow Outcome\nStatus: `{state.get('status', 'unknown')}`\n"
    )


def _render_recommendation(recommendation: RecommendationRecord | None) -> str:
    if recommendation is None:
        return "No recommendation record available."
    return (
        f"Type: `{recommendation.recommendation_type}`\n\n"
        f"Summary: {recommendation.summary}\n\n"
        f"Rationale: {recommendation.rationale}\n\n"
        f"Confidence: {recommendation.confidence_level:.2f}\n"
        f"Supporting Evidence IDs: {', '.join(recommendation.supporting_evidence_ids) or 'none'}\n"
        f"Unresolved Questions: {', '.join(recommendation.unresolved_questions) or 'none'}\n"
        f"Residual Risks: {', '.join(recommendation.residual_risks) or 'none'}"
    )


def _render_governance(governance: GovernanceEvaluation | None) -> str:
    if governance is None:
        return "No governance evaluation record available."
    return (
        f"Requires Human Review: {governance.requires_human_review}\n"
        f"Overall Risk Score: {governance.overall_risk_score:.2f}\n"
        f"Triggered Policies: {', '.join(governance.triggered_policies) or 'none'}\n"
        f"Risk Summary: {governance.risk_summary}\n"
        f"Evidence Gaps: {', '.join(governance.evidence_gaps) or 'none'}\n"
        f"Contradiction Summary: {governance.contradiction_summary}\n"
        f"Required Human Action: {governance.required_human_action}"
    )


def render_review_feedback(review: ReviewFeedback | None) -> str:
    review = _coerce_review(review)
    if review is None:
        return "No review feedback available."
    questions = "\n".join(f"- {question}" for question in review.questions) or "- None"
    revisions = "\n".join(f"- {item}" for item in review.revision_requests) or "- None"
    risks = "\n".join(f"- {item}" for item in review.major_risks) or "- None"
    return (
        f"Verdict: `{review.verdict}`\n\n"
        f"Summary: {review.summary}\n\n"
        f"Questions:\n{questions}\n\n"
        f"Revision Requests:\n{revisions}\n\n"
        f"Major Risks:\n{risks}\n"
    )


def build_decision_record(state: AgentState) -> dict[str, Any]:
    evidence = _coerce_evidence(state)
    recommendation = _coerce_recommendation(state.get("recommendation"))
    governance = _coerce_governance(state.get("governance_evaluation"))
    approval = state.get("approval_decision")
    if approval is not None and not isinstance(approval, ApprovalDecision):
        approval = ApprovalDecision.model_validate(approval)

    evidence_ids = [record.evidence_id for record in evidence]
    confidence = recommendation.confidence_level if recommendation else (mean(record.confidence for record in evidence) if evidence else 0.0)
    risks = recommendation.residual_risks if recommendation else []
    if governance and governance.triggered_policies:
        risks = sorted(set(risks + [f"policy:{item}" for item in governance.triggered_policies]))

    decision = DecisionRecord(
        task_id=state["task_id"],
        recommendation=recommendation.summary if recommendation else "No recommendation available.",
        confidence=confidence,
        risks=risks,
        evidence_ids=evidence_ids,
        approval=approval,
    )
    return decision.model_dump()


def build_workflow_trace(state: AgentState) -> dict[str, Any]:
    return {
        "task_id": state["task_id"],
        "status": state.get("status"),
        "trace": [event.model_dump() for event in _coerce_trace(state)],
        "tool_calls": [record.model_dump() for record in _coerce_tool_calls(state)],
    }


def build_mermaid_diagram() -> str:
    return "\n".join(
        [
            "flowchart TD",
            "    START([START]) --> init[initialize_task]",
            "    init --> planner[planner_agent]",
            "    planner --> supervisor[supervisor]",
            "    supervisor --> research[research_pipeline]",
            "    research --> supervisor",
            "    supervisor --> analyst[analyst_agent]",
            "    supervisor -->|policy gate| hitl[human_review]",
            "    analyst --> reviewer[reviewer_agent]",
            "    reviewer -->|revise| analyst",
            "    reviewer -->|escalate| hitl[human_review]",
            "    reviewer -->|approve| executor[executor_agent]",
            "    hitl -->|approved| executor",
            "    hitl -->|rework| analyst",
            "    executor --> supervisor",
            "    supervisor --> END([END])",
        ]
    )


def build_task_summary_html(state: AgentState) -> str:
    review = _coerce_review(state.get("review_feedback"))
    review_summary = review.summary if review else "No reviewer summary."
    artifacts = "".join(f"<li>{artifact.name}: {artifact.path}</li>" for artifact in _coerce_artifacts(state))
    return (
        "<html><body>"
        f"<h1>Task Summary: {state['task_id']}</h1>"
        f"<p>Status: {state.get('status')}</p>"
        f"<p>Review Summary: {review_summary}</p>"
        f"<ul>{artifacts}</ul>"
        "</body></html>"
    )


def build_run_artifact(state: AgentState) -> dict[str, Any]:
    return to_jsonable(
        {
            "task_id": state["task_id"],
            "status": state.get("status"),
            "plan": state.get("plan", []),
            "coverage_record": state.get("coverage_record"),
            "evidence_assessments": state.get("evidence_assessments", []),
            "evidence_conflicts": state.get("evidence_conflicts", []),
            "evidence_supports": state.get("evidence_supports", []),
            "artifacts": _coerce_artifacts(state),
            "approval_decision": state.get("approval_decision"),
            "recommendation": state.get("recommendation"),
            "governance_evaluation": state.get("governance_evaluation"),
        }
    )
