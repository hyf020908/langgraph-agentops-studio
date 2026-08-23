from __future__ import annotations

from artifacts.exporter import render_final_report
from schemas.models import CoverageRecord, EvidenceRecord, FindingRecord
from services.config import ReportingSettings
from services.recommendation import RecommendationService


def _evidence(index: int, *, risks: list[str] | None = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"EVD-{index:02d}",
        claim=f"Claim {index}",
        supporting_sources=[f"SRC-{index:02d}"],
        confidence=0.82,
        risk_flags=risks or [],
        summary=f"Evidence {index} supports a staged implementation with measurable controls.",
    )


def _finding(index: int, *, risk_level: str = "low") -> FindingRecord:
    return FindingRecord(
        finding_id=f"F{index}",
        theme=f"Theme {index}",
        insight=f"Finding {index} changes the implementation decision because it affects reliability and reversibility.",
        rationale=f"EVD-{index:02d} provides a traceable basis and identifies a delivery trade-off.",
        evidence_ids=[f"EVD-{index:02d}"],
        risk_level=risk_level,
    )


def test_detailed_recommendation_is_structured_and_keeps_governance_risks_complete() -> None:
    settings = ReportingSettings(
        detail_level="detailed",
        max_findings_in_conclusion=5,
        max_evidence_in_conclusion=6,
        max_risks_in_conclusion=1,
    )
    service = RecommendationService(settings)
    evidence = [
        _evidence(1, risks=["vendor-lock-in", "operational-complexity"]),
        _evidence(2, risks=["migration-risk"]),
        _evidence(3),
    ]
    findings = [_finding(1, risk_level="high"), _finding(2), _finding(3)]
    coverage = CoverageRecord(
        query_coverage=0.9,
        criteria_coverage=0.9,
        evidence_count=len(evidence),
    )

    recommendation = service.synthesize(
        user_request="Choose an implementation direction",
        findings=findings,
        ranked_evidence=evidence,
        evidence_assessments=[],
        coverage_record=coverage,
    )

    assert recommendation.recommendation_type == "directional"
    assert len(recommendation.key_findings) == 3
    assert recommendation.tradeoffs
    assert len(recommendation.risk_mitigations) == 1
    # Reporting limits must never truncate the governance-facing risk ledger.
    assert set(recommendation.residual_risks) == {
        "finding:F1:high",
        "migration-risk",
        "operational-complexity",
        "vendor-lock-in",
    }
    assert "Key findings:" in recommendation.rationale
    assert "Evidence basis:" in recommendation.rationale
    assert "Alternatives and trade-offs:" in recommendation.rationale
    assert "Risk mitigation:" in recommendation.rationale
    assert "Next actions:" in recommendation.rationale
    assert len(recommendation.rationale) > 600

    report = render_final_report(
        {
            "task_id": "task-report",
            "user_request": "Choose an implementation direction",
            "status": "completed",
            "findings": findings,
            "ranked_evidence": evidence,
            "coverage_record": coverage,
            "recommendation": recommendation,
        }
    )
    assert "### Key Findings Behind the Conclusion" in report
    assert "### Alternatives and Trade-offs" in report
    assert "### Risk Mitigations" in report
    assert "### Next Actions" in report


def test_compact_detail_level_only_changes_conclusion_projection() -> None:
    evidence = [_evidence(index) for index in range(1, 6)]
    findings = [_finding(index) for index in range(1, 6)]
    coverage = CoverageRecord(query_coverage=0.9, criteria_coverage=0.9, evidence_count=5)
    service = RecommendationService(
        ReportingSettings(
            detail_level="compact",
            max_findings_in_conclusion=5,
            max_evidence_in_conclusion=6,
            max_risks_in_conclusion=5,
        )
    )

    recommendation = service.synthesize(
        user_request="Choose an implementation direction",
        findings=findings,
        ranked_evidence=evidence,
        evidence_assessments=[],
        coverage_record=coverage,
    )

    assert len(recommendation.key_findings) == 2
    assert recommendation.supporting_evidence_ids == ["EVD-01", "EVD-02", "EVD-03", "EVD-04", "EVD-05"]
    evidence_section = recommendation.rationale.split("Evidence basis: ", 1)[1].split(
        "\n\nAlternatives and trade-offs:",
        1,
    )[0]
    assert "EVD-02" in evidence_section
    assert "EVD-03" not in evidence_section
