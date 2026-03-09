from __future__ import annotations

from statistics import mean

from schemas.models import CoverageRecord, EvidenceAssessment, EvidenceRecord, FindingRecord, RecommendationRecord


class RecommendationService:
    def synthesize(
        self,
        *,
        user_request: str,
        findings: list[FindingRecord],
        ranked_evidence: list[EvidenceRecord],
        evidence_assessments: list[EvidenceAssessment],
        coverage_record: CoverageRecord | None,
    ) -> RecommendationRecord:
        confidence = self._estimate_confidence(ranked_evidence, coverage_record)
        contradiction_pressure = self._contradiction_pressure(evidence_assessments)
        unresolved_risks = self._collect_risks(ranked_evidence, findings)
        open_questions = self._collect_open_questions(findings, coverage_record)

        evidence_ids = [item.evidence_id for item in ranked_evidence[:4]]
        coverage = coverage_record.query_coverage if coverage_record else 0.0

        if len(ranked_evidence) < 2 or confidence < 0.45 or coverage < 0.45:
            return RecommendationRecord(
                recommendation_type="insufficient_evidence",
                summary="The current evidence set is insufficient for a definitive recommendation.",
                rationale=(
                    "Evidence coverage and confidence remain below the required level for a single-direction decision. "
                    "Additional targeted research is required before selecting a final option."
                ),
                confidence_level=confidence,
                supporting_evidence_ids=evidence_ids,
                unresolved_questions=open_questions,
                residual_risks=unresolved_risks,
            )

        if contradiction_pressure >= 0.45 or len(open_questions) >= 2:
            return RecommendationRecord(
                recommendation_type="conditional",
                summary="Proceed with a staged recommendation conditioned on risk controls and validation milestones.",
                rationale=(
                    "Evidence supports progress, but contradiction levels and unresolved questions indicate that "
                    "a phased rollout with explicit validation gates is the safer decision path."
                ),
                confidence_level=confidence,
                supporting_evidence_ids=evidence_ids,
                unresolved_questions=open_questions,
                residual_risks=unresolved_risks,
            )

        major_themes = ", ".join(item.theme for item in findings[:3]) if findings else "evidence-backed priorities"
        return RecommendationRecord(
            recommendation_type="directional",
            summary="Proceed with the option set best supported by the highest-ranked evidence cluster.",
            rationale=(
                f"Top findings emphasize {major_themes}. The strongest evidence group is internally consistent, "
                "coverage is above threshold, and contradiction pressure is contained."
            ),
            confidence_level=confidence,
            supporting_evidence_ids=evidence_ids,
            unresolved_questions=open_questions,
            residual_risks=unresolved_risks,
        )

    @staticmethod
    def _estimate_confidence(ranked_evidence: list[EvidenceRecord], coverage_record: CoverageRecord | None) -> float:
        base = mean(item.confidence for item in ranked_evidence[:6]) if ranked_evidence else 0.0
        if coverage_record is None:
            return max(0.0, min(1.0, base * 0.85))
        coverage_weight = mean([coverage_record.query_coverage, coverage_record.criteria_coverage])
        return max(0.0, min(1.0, 0.75 * base + 0.25 * coverage_weight))

    @staticmethod
    def _contradiction_pressure(evidence_assessments: list[EvidenceAssessment]) -> float:
        if not evidence_assessments:
            return 0.0
        return mean(item.score_breakdown.contradiction_penalty for item in evidence_assessments)

    @staticmethod
    def _collect_risks(ranked_evidence: list[EvidenceRecord], findings: list[FindingRecord]) -> list[str]:
        risks = {flag for item in ranked_evidence for flag in item.risk_flags}
        risks.update(
            f"finding:{item.finding_id}:{item.risk_level}"
            for item in findings
            if item.risk_level in {"medium", "high"}
        )
        return sorted(risks)

    @staticmethod
    def _collect_open_questions(findings: list[FindingRecord], coverage_record: CoverageRecord | None) -> list[str]:
        questions: list[str] = []
        if coverage_record and coverage_record.query_coverage < 0.55:
            questions.append("Which critical task constraints remain under-covered by current evidence?")
        if coverage_record and coverage_record.criteria_coverage < 0.55:
            questions.append("Which acceptance criteria require additional validation evidence?")

        for finding in findings:
            if finding.risk_level == "high":
                questions.append(f"What mitigation is required for high-risk finding {finding.finding_id}?")

        unique: list[str] = []
        for item in questions:
            if item not in unique:
                unique.append(item)
        return unique[:5]
