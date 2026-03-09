from __future__ import annotations

from schemas.models import SourceRecord
from services.evidence import EvidencePipeline


def test_evidence_pipeline_scores_multiple_dimensions() -> None:
    pipeline = EvidencePipeline()
    sources = [
        SourceRecord(
            source_id="SRC-1",
            provider="tavily",
            source_type="webpage",
            title="Workflow governance patterns",
            url="https://example.com/gov",
            source="https://example.com/gov",
            domain="example.com",
            snippet="Governance policies should gate high-risk decisions.",
            content="Governance policies should gate high-risk decisions with explicit approval criteria and escalation logic.",
            published_at="2026-01-15",
            credibility=0.81,
        ),
        SourceRecord(
            source_id="SRC-2",
            provider="exa",
            source_type="webpage",
            title="Evidence ranking methods",
            url="https://example.com/evidence",
            source="https://example.com/evidence",
            domain="example.com",
            snippet="Evidence ranking can combine relevance, credibility, recency, and contradiction analysis.",
            content="Evidence ranking can combine relevance, credibility, recency, contradiction analysis, and actionability.",
            published_at="2025-12-20",
            credibility=0.79,
        ),
    ]

    result = pipeline.assess(
        sources=sources,
        user_request="Design governance and evidence ranking for an agent workflow",
        acceptance_criteria=["Provide governance thresholds", "Rank evidence with explainable dimensions"],
    )

    assert len(result.ranked_evidence) == 2
    assert len(result.assessments) == 2
    assert result.coverage.query_coverage > 0.0
    assert result.coverage.criteria_coverage > 0.0
    assert result.assessments[0].score_breakdown.relevance >= 0.0
    assert result.assessments[0].score_breakdown.actionability >= 0.0
