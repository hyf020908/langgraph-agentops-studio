from __future__ import annotations

from statistics import mean

from schemas.models import ConflictRecord, CoverageRecord, EvidenceAssessment, GovernanceEvaluation, RecommendationRecord
from services.config import GovernanceSettings


class GovernanceService:
    def __init__(self, policy: GovernanceSettings) -> None:
        self.policy = policy

    def evaluate(
        self,
        *,
        recommendation: RecommendationRecord,
        coverage_record: CoverageRecord | None,
        evidence_assessments: list[EvidenceAssessment],
        conflicts: list[ConflictRecord],
        task_type: str,
        user_request: str,
    ) -> GovernanceEvaluation:
        triggered: list[str] = []

        evidence_gaps = list(coverage_record.coverage_notes) if coverage_record else ["coverage record unavailable"]
        contradiction_peak = max((item.severity for item in conflicts), default=0.0)
        contradiction_summary = f"max_conflict_severity={contradiction_peak:.2f}, conflict_count={len(conflicts)}"

        if recommendation.confidence_level < self.policy.recommendation_confidence_threshold:
            triggered.append("recommendation_confidence_threshold")

        if coverage_record and coverage_record.criteria_coverage < self.policy.evidence_completeness_threshold:
            triggered.append("evidence_completeness_threshold")

        if contradiction_peak > self.policy.contradiction_severity_threshold:
            triggered.append("contradiction_severity_threshold")

        if len(recommendation.unresolved_questions) > self.policy.unresolved_questions_threshold:
            triggered.append("unresolved_questions_threshold")

        normalized_task_type = task_type.strip().lower() or "general"
        normalized_request = user_request.lower()
        if normalized_task_type in {item.lower() for item in self.policy.high_stakes_task_categories}:
            triggered.append("high_stakes_task_category")
        elif any(tag.lower() in normalized_request for tag in self.policy.high_stakes_task_categories):
            triggered.append("high_stakes_task_category")

        policy_value = self.policy.manual_approval_policy_by_task_type.get(normalized_task_type)
        if policy_value and policy_value.strip().lower() in {"required", "always", "true"}:
            triggered.append("task_type_manual_approval")

        overall_risk_score = self._estimate_risk_score(
            recommendation=recommendation,
            coverage_record=coverage_record,
            evidence_assessments=evidence_assessments,
            contradiction_peak=contradiction_peak,
        )
        if overall_risk_score >= max(self.policy.overall_risk_threshold, 0.0):
            triggered.append("overall_risk_threshold")

        requires_human_review = len(triggered) > 0
        required_action = (
            "Review governance triggers, validate unresolved risks, and provide explicit approval decision."
            if requires_human_review
            else "No manual gate required; continue automated execution."
        )

        return GovernanceEvaluation(
            requires_human_review=requires_human_review,
            triggered_policies=sorted(set(triggered)),
            risk_summary=(
                f"overall_risk_score={overall_risk_score:.2f}; "
                f"recommendation_confidence={recommendation.confidence_level:.2f}; "
                f"open_questions={len(recommendation.unresolved_questions)}"
            ),
            evidence_gaps=evidence_gaps,
            contradiction_summary=contradiction_summary,
            recommendation_confidence=recommendation.confidence_level,
            required_human_action=required_action,
            overall_risk_score=overall_risk_score,
        )

    @staticmethod
    def _estimate_risk_score(
        *,
        recommendation: RecommendationRecord,
        coverage_record: CoverageRecord | None,
        evidence_assessments: list[EvidenceAssessment],
        contradiction_peak: float,
    ) -> float:
        evidence_strength = mean(item.overall_score for item in evidence_assessments[:8]) if evidence_assessments else 0.0
        coverage_strength = (
            mean([coverage_record.query_coverage, coverage_record.criteria_coverage])
            if coverage_record is not None
            else 0.0
        )
        unresolved_factor = min(1.0, len(recommendation.residual_risks) / 8)

        risk = (
            0.28 * (1.0 - recommendation.confidence_level)
            + 0.22 * (1.0 - evidence_strength)
            + 0.2 * (1.0 - coverage_strength)
            + 0.18 * contradiction_peak
            + 0.12 * unresolved_factor
        )
        return max(0.0, min(1.0, risk))
