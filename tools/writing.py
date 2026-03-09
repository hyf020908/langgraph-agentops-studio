from __future__ import annotations

import json
from statistics import mean
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from schemas.models import (
    ConflictRecord,
    CoverageRecord,
    EvidenceAssessment,
    EvidenceRecord,
    FindingRecord,
    GovernanceEvaluation,
    PlanStep,
    RecommendationRecord,
    SupportRecord,
)


class ReportWriterInput(BaseModel):
    user_request: str
    plan: list[PlanStep]
    findings: list[FindingRecord]
    ranked_evidence: list[EvidenceRecord]
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    evidence_conflicts: list[ConflictRecord] = Field(default_factory=list)
    evidence_supports: list[SupportRecord] = Field(default_factory=list)
    coverage_record: CoverageRecord | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    review_feedback_summary: str | None = None
    recommendation: RecommendationRecord | None = None
    governance_evaluation: GovernanceEvaluation | None = None


def write_report(
    user_request: str,
    plan: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    ranked_evidence: list[dict[str, Any]],
    evidence_assessments: list[dict[str, Any]] | None = None,
    evidence_conflicts: list[dict[str, Any]] | None = None,
    evidence_supports: list[dict[str, Any]] | None = None,
    coverage_record: dict[str, Any] | None = None,
    acceptance_criteria: list[str] | None = None,
    review_feedback_summary: str | None = None,
    recommendation: dict[str, Any] | None = None,
    governance_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_steps = [PlanStep.model_validate(item) for item in plan]
    finding_models = [FindingRecord.model_validate(item) for item in findings]
    evidence_models = [EvidenceRecord.model_validate(item) for item in ranked_evidence]
    assessment_models = [EvidenceAssessment.model_validate(item) for item in (evidence_assessments or [])]
    conflict_models = [ConflictRecord.model_validate(item) for item in (evidence_conflicts or [])]
    support_models = [SupportRecord.model_validate(item) for item in (evidence_supports or [])]
    coverage_model = CoverageRecord.model_validate(coverage_record) if coverage_record else None
    recommendation_model = RecommendationRecord.model_validate(recommendation) if recommendation else None
    governance_model = GovernanceEvaluation.model_validate(governance_evaluation) if governance_evaluation else None

    avg_confidence = mean(item.confidence for item in evidence_models) if evidence_models else 0.0
    plan_lines = "\n".join(f"- {step.step_id}: {step.objective} ({step.owner})" for step in plan_steps) or "- none"
    finding_lines = "\n".join(f"- {finding.theme}: {finding.insight}" for finding in finding_models) or "- none"
    evidence_lines_list: list[str] = []
    for record in evidence_models:
        citations = (
            ", ".join(
                f"{citation.title} ({citation.source}{' #' + citation.chunk_id if citation.chunk_id else ''})"
                for citation in record.citations
            )
            if record.citations
            else "none"
        )
        line = f"- {record.evidence_id}: {record.summary} | confidence={record.confidence:.2f} | citations: {citations}"
        if record.assessment:
            breakdown = record.assessment.score_breakdown
            line += (
                "\n  - assessment:"
                f" overall_score={record.assessment.overall_score:.2f},"
                f" relevance={breakdown.relevance:.2f},"
                f" source_credibility={breakdown.source_credibility:.2f},"
                f" recency={breakdown.recency:.2f},"
                f" completeness={breakdown.completeness:.2f},"
                f" corroboration={breakdown.corroboration:.2f},"
                f" contradiction_penalty={breakdown.contradiction_penalty:.2f},"
                f" extraction_quality={breakdown.extraction_quality:.2f},"
                f" actionability={breakdown.actionability:.2f}"
                f"\n  - strengths: {', '.join(record.assessment.strengths) or 'none'}"
                f"\n  - weaknesses: {', '.join(record.assessment.weaknesses) or 'none'}"
                f"\n  - flags: {', '.join(record.assessment.flags) or 'none'}"
                f"\n  - conflicts_with: {', '.join(record.assessment.conflicts_with) or 'none'}"
                f"\n  - supports: {', '.join(record.assessment.supports) or 'none'}"
            )
        evidence_lines_list.append(line)
    evidence_lines = "\n".join(evidence_lines_list) or "- none"
    acceptance_lines = "\n".join(f"- {criterion}" for criterion in acceptance_criteria or []) or "- none"
    coverage_block = (
        f"Query Coverage: {coverage_model.query_coverage:.2f}\n"
        f"Criteria Coverage: {coverage_model.criteria_coverage:.2f}\n"
        f"Evidence Count: {coverage_model.evidence_count}\n"
        f"Coverage Notes: {', '.join(coverage_model.coverage_notes) or 'none'}"
        if coverage_model
        else "No coverage record provided."
    )
    relations_block = (
        f"Support Links: {len(support_models)}\n"
        f"Conflict Links: {len(conflict_models)}\n"
        f"Assessment Records: {len(assessment_models)}"
    )

    if recommendation_model:
        recommendation_block = (
            f"Type: `{recommendation_model.recommendation_type}`\n"
            f"Confidence: {recommendation_model.confidence_level:.2f}\n"
            f"Summary: {recommendation_model.summary}\n\n"
            f"Rationale: {recommendation_model.rationale}\n\n"
            f"Supporting Evidence IDs: {', '.join(recommendation_model.supporting_evidence_ids) or 'none'}\n"
            f"Unresolved Questions: {', '.join(recommendation_model.unresolved_questions) or 'none'}\n"
            f"Residual Risks: {', '.join(recommendation_model.residual_risks) or 'none'}"
        )
    else:
        recommendation_block = (
            "Type: `insufficient_evidence`\n"
            "Confidence: 0.00\n"
            "Summary: Recommendation was not generated.\n\n"
            "Rationale: Evidence synthesis did not provide a recommendation payload."
        )

    governance_block = (
        (
            f"Requires Human Review: {governance_model.requires_human_review}\n"
            f"Overall Risk Score: {governance_model.overall_risk_score:.2f}\n"
            f"Triggered Policies: {', '.join(governance_model.triggered_policies) or 'none'}\n"
            f"Risk Summary: {governance_model.risk_summary}\n"
            f"Evidence Gaps: {', '.join(governance_model.evidence_gaps) or 'none'}\n"
            f"Contradiction Summary: {governance_model.contradiction_summary}\n"
            f"Required Human Action: {governance_model.required_human_action}"
        )
        if governance_model
        else "No governance evaluation payload was provided."
    )

    feedback_line = review_feedback_summary or "No reviewer feedback has been incorporated yet."
    report = (
        f"# Draft Report\n\n"
        f"## Task\n{user_request}\n\n"
        f"## Execution Plan\n{plan_lines}\n\n"
        f"## Acceptance Criteria\n{acceptance_lines}\n\n"
        f"## Evidence Summary\n{evidence_lines}\n\n"
        f"## Findings\n{finding_lines}\n\n"
        f"## Evidence Coverage\n{coverage_block}\n\n"
        f"## Evidence Relations\n{relations_block}\n\n"
        f"## Recommendation\n{recommendation_block}\n\n"
        f"## Governance Evaluation\n{governance_block}\n\n"
        f"## Confidence Snapshot\nAggregate evidence confidence: {avg_confidence:.2f}\n\n"
        f"## Reviewer-Requested Revision\n{feedback_line}\n"
    )
    return {"draft_report": report}


def build_report_writer_tool():
    @tool("report_writer_tool", args_schema=ReportWriterInput)
    def report_writer_tool(
        user_request: str,
        plan: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        ranked_evidence: list[dict[str, Any]],
        evidence_assessments: list[dict[str, Any]] | None = None,
        evidence_conflicts: list[dict[str, Any]] | None = None,
        evidence_supports: list[dict[str, Any]] | None = None,
        coverage_record: dict[str, Any] | None = None,
        acceptance_criteria: list[str] | None = None,
        review_feedback_summary: str | None = None,
        recommendation: dict[str, Any] | None = None,
        governance_evaluation: dict[str, Any] | None = None,
    ) -> str:
        """Write a structured markdown report from plan, findings, evidence, recommendation, and governance data."""
        return json.dumps(
            write_report(
                user_request=user_request,
                plan=plan,
                findings=findings,
                ranked_evidence=ranked_evidence,
                evidence_assessments=evidence_assessments,
                evidence_conflicts=evidence_conflicts,
                evidence_supports=evidence_supports,
                coverage_record=coverage_record,
                acceptance_criteria=acceptance_criteria,
                review_feedback_summary=review_feedback_summary,
                recommendation=recommendation,
                governance_evaluation=governance_evaluation,
            )
        )

    return report_writer_tool
