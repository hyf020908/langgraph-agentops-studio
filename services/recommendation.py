from __future__ import annotations

# Recommendation synthesis.
# This service converts findings plus evidence quality signals into a single
# recommendation record that governance and export can evaluate consistently.

from statistics import mean

from schemas.models import CoverageRecord, EvidenceAssessment, EvidenceRecord, FindingRecord, RecommendationRecord
from services.config import ReportingSettings


class RecommendationService:
    def __init__(self, settings: ReportingSettings | None = None) -> None:
        self.settings = settings or ReportingSettings()

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

        finding_limit, evidence_limit, risk_limit = self._detail_limits()
        # Keep the machine-facing support ledger independent from presentation
        # settings; only the prose evidence projection below is detail-limited.
        supporting_evidence_ids = [item.evidence_id for item in ranked_evidence[:6]]
        coverage = coverage_record.query_coverage if coverage_record else 0.0
        key_findings = [
            (
                f"{item.finding_id} — {item.theme}: {item.insight} "
                f"Evidence linkage: {', '.join(item.evidence_ids) or 'not explicitly linked'}."
            )
            for item in findings[:finding_limit]
        ]
        evidence_basis = [
            f"{item.evidence_id} (confidence {item.confidence:.2f}): {item.summary}"
            for item in ranked_evidence[:evidence_limit]
        ]
        bounded_risks = unresolved_risks[:risk_limit]

        # Recommendation mode is determined by evidence sufficiency first,
        # disagreement/open questions second, and directional confidence last.
        if len(ranked_evidence) < 2 or confidence < 0.45 or coverage < 0.45:
            tradeoffs = [
                "Deferring an irreversible decision increases near-term research time, but avoids locking in a direction whose evidence coverage is below the decision threshold.",
                f"Current confidence is {confidence:.2f} and query coverage is {coverage:.2f}; speed should therefore be traded for targeted validation.",
            ]
            mitigations = self._risk_mitigations(bounded_risks, findings)
            next_actions = self._next_actions(open_questions, supporting_evidence_ids, staged=False)
            return RecommendationRecord(
                recommendation_type="insufficient_evidence",
                summary=(
                    "Do not make an irreversible final selection yet. The current evidence set supports narrowing the decision, "
                    f"but confidence ({confidence:.2f}) and coverage ({coverage:.2f}) are not strong enough for a defensible commitment."
                ),
                rationale=self._build_rationale(key_findings, evidence_basis, tradeoffs, mitigations, next_actions),
                confidence_level=confidence,
                supporting_evidence_ids=supporting_evidence_ids,
                unresolved_questions=open_questions,
                residual_risks=unresolved_risks,
                key_findings=key_findings,
                tradeoffs=tradeoffs,
                risk_mitigations=mitigations,
                next_actions=next_actions,
            )

        if contradiction_pressure >= 0.45 or len(open_questions) >= 2:
            tradeoffs = [
                "A staged path preserves delivery momentum while keeping reversal costs bounded until disputed assumptions are validated.",
                f"Contradiction pressure is {contradiction_pressure:.2f}; the leading direction has support, but competing evidence still changes the acceptable rollout risk.",
            ]
            mitigations = self._risk_mitigations(bounded_risks, findings)
            next_actions = self._next_actions(open_questions, supporting_evidence_ids, staged=True)
            return RecommendationRecord(
                recommendation_type="conditional",
                summary=(
                    "Proceed through a staged, reversible rollout rather than a full commitment. Evidence supports progress, "
                    "provided that validation milestones, named risk owners, and stop/go criteria are enforced."
                ),
                rationale=self._build_rationale(key_findings, evidence_basis, tradeoffs, mitigations, next_actions),
                confidence_level=confidence,
                supporting_evidence_ids=supporting_evidence_ids,
                unresolved_questions=open_questions,
                residual_risks=unresolved_risks,
                key_findings=key_findings,
                tradeoffs=tradeoffs,
                risk_mitigations=mitigations,
                next_actions=next_actions,
            )

        major_themes = ", ".join(item.theme for item in findings[:3]) if findings else "evidence-backed priorities"
        tradeoffs = [
            f"The leading direction best satisfies {major_themes}, while the principal alternative is to delay for more evidence at the cost of slower delivery.",
            f"Confidence is {confidence:.2f} with contained contradiction pressure ({contradiction_pressure:.2f}); residual risks still require monitored controls rather than being treated as resolved.",
        ]
        mitigations = self._risk_mitigations(bounded_risks, findings)
        next_actions = self._next_actions(open_questions, supporting_evidence_ids, staged=True)
        return RecommendationRecord(
            recommendation_type="directional",
            summary=(
                "Proceed with the direction supported by the highest-ranked evidence cluster, using measurable implementation "
                f"gates to preserve reversibility. The decision is supported by themes in {major_themes} and confidence of {confidence:.2f}."
            ),
            rationale=self._build_rationale(key_findings, evidence_basis, tradeoffs, mitigations, next_actions),
            confidence_level=confidence,
            supporting_evidence_ids=supporting_evidence_ids,
            unresolved_questions=open_questions,
            residual_risks=unresolved_risks,
            key_findings=key_findings,
            tradeoffs=tradeoffs,
            risk_mitigations=mitigations,
            next_actions=next_actions,
        )

    def _detail_limits(self) -> tuple[int, int, int]:
        if self.settings.detail_level == "compact":
            return (
                min(2, self.settings.max_findings_in_conclusion),
                min(2, self.settings.max_evidence_in_conclusion),
                min(2, self.settings.max_risks_in_conclusion),
            )
        if self.settings.detail_level == "standard":
            return (
                min(4, self.settings.max_findings_in_conclusion),
                min(4, self.settings.max_evidence_in_conclusion),
                min(3, self.settings.max_risks_in_conclusion),
            )
        return (
            self.settings.max_findings_in_conclusion,
            self.settings.max_evidence_in_conclusion,
            self.settings.max_risks_in_conclusion,
        )

    def _next_actions(self, questions: list[str], evidence_ids: list[str], *, staged: bool) -> list[str]:
        if not self.settings.include_next_actions:
            return []
        actions = [f"Resolve and document: {question}" for question in questions[:3]]
        if staged:
            actions.append("Define a bounded pilot with measurable success, stop, and rollback criteria before wider rollout.")
        actions.extend(
            [
                f"Revalidate the highest-weight evidence ({', '.join(evidence_ids) or 'none'}) at the next decision gate.",
                "Assign an accountable owner and due date to each residual risk and record the resulting control evidence.",
            ]
        )
        return actions[:5]

    @staticmethod
    def _risk_mitigations(risks: list[str], findings: list[FindingRecord]) -> list[str]:
        finding_by_id = {item.finding_id: item for item in findings}
        mitigations: list[str] = []
        for risk in risks:
            parts = risk.split(":")
            if len(parts) >= 3 and parts[0] == "finding" and parts[1] in finding_by_id:
                finding = finding_by_id[parts[1]]
                mitigations.append(
                    f"For {finding.finding_id} ({finding.theme}), define a named control owner, an observable leading indicator, and a rollback threshold before implementation."
                )
            else:
                mitigations.append(
                    f"Treat '{risk}' as an open control item: validate it in a bounded environment, record the result, and require owner sign-off before scale-up."
                )
        if not mitigations:
            mitigations.append(
                "Maintain evidence freshness and outcome monitoring; reopen the decision if confidence, coverage, or contradiction indicators cross governance thresholds."
            )
        return mitigations

    @staticmethod
    def _build_rationale(
        key_findings: list[str],
        evidence_basis: list[str],
        tradeoffs: list[str],
        mitigations: list[str],
        next_actions: list[str],
    ) -> str:
        sections = [
            "Key findings: " + (" ".join(key_findings) if key_findings else "No validated findings were produced."),
            "Evidence basis: " + (" ".join(evidence_basis) if evidence_basis else "No ranked evidence was available."),
            "Alternatives and trade-offs: " + " ".join(tradeoffs),
            "Risk mitigation: " + " ".join(mitigations),
        ]
        if next_actions:
            sections.append("Next actions: " + " ".join(next_actions))
        return "\n\n".join(sections)

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
