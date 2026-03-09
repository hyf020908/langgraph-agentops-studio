from __future__ import annotations

from schemas.models import (
    ConflictRecord,
    CoverageRecord,
    EvidenceAssessment,
    EvidenceScoreBreakdown,
    RecommendationRecord,
)
from services.config import GovernanceSettings
from services.governance import GovernanceService


def test_governance_service_triggers_policy_conditions() -> None:
    service = GovernanceService(
        GovernanceSettings(
            overall_risk_threshold=0.4,
            recommendation_confidence_threshold=0.7,
            evidence_completeness_threshold=0.65,
            contradiction_severity_threshold=0.5,
            unresolved_questions_threshold=1,
            high_stakes_task_categories=["architecture"],
            manual_approval_policy_by_task_type={"architecture": "required"},
        )
    )

    recommendation = RecommendationRecord(
        recommendation_type="conditional",
        summary="Proceed in staged increments.",
        rationale="Conflict pressure and coverage gaps require phased execution.",
        confidence_level=0.58,
        supporting_evidence_ids=["EVD-01", "EVD-02"],
        unresolved_questions=["Which dependency must be validated first?", "What rollback criterion is required?"],
        residual_risks=["risk:integration", "risk:operational"],
    )

    coverage = CoverageRecord(
        query_coverage=0.52,
        criteria_coverage=0.48,
        evidence_count=3,
        coverage_notes=["criteria coverage is below target"],
    )

    assessments = [
        EvidenceAssessment(
            source_id="SRC-1",
            overall_score=0.62,
            score_breakdown=EvidenceScoreBreakdown(
                relevance=0.7,
                source_credibility=0.68,
                recency=0.74,
                completeness=0.58,
                corroboration=0.45,
                contradiction_penalty=0.4,
                extraction_quality=0.75,
                actionability=0.62,
            ),
            strengths=["high query relevance"],
            weaknesses=["limited corroboration"],
            flags=["low_content_depth"],
            conflicts_with=["SRC-2"],
            supports=[],
        )
    ]

    conflicts = [
        ConflictRecord(
            left_source_id="SRC-1",
            right_source_id="SRC-2",
            severity=0.66,
            reason="opposing stance on rollout sequencing",
        )
    ]

    result = service.evaluate(
        recommendation=recommendation,
        coverage_record=coverage,
        evidence_assessments=assessments,
        conflicts=conflicts,
        task_type="architecture",
        user_request="Define architecture rollout policy",
    )

    assert result.requires_human_review is True
    assert "overall_risk_threshold" in result.triggered_policies
    assert "task_type_manual_approval" in result.triggered_policies
    assert result.overall_risk_score >= 0.0
