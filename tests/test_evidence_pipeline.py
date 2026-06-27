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


def test_coverage_does_not_count_self_referential_claim_text() -> None:
    pipeline = EvidencePipeline()
    sources = [
        SourceRecord(
            source_id="SRC-1",
            provider="tavily",
            source_type="webpage",
            title="Unrelated source",
            url="https://example.com/unrelated",
            source="https://example.com/unrelated",
            domain="example.com",
            snippet="Bananas and oranges are common fruit in grocery stores.",
            content="Bananas and oranges are common fruit in grocery stores.",
            credibility=0.7,
        )
    ]

    result = pipeline.assess(
        sources=sources,
        user_request="core trading compliance disaster recovery approval",
        acceptance_criteria=["quantified RTO RPO comparison"],
    )

    assert result.coverage.query_coverage == 0.0
    assert result.coverage.criteria_coverage == 0.0


def test_relevance_uses_grounding_query_metadata() -> None:
    source = SourceRecord(
        source_id="SRC-1",
        provider="tavily",
        source_type="webpage",
        title="Trading platform migration",
        url="https://example.com/trading",
        source="https://example.com/trading",
        domain="example.com",
        snippet="Microservices migration for trading platforms requires audit controls.",
        metadata={"grounding_query": "trading platform microservices migration audit controls"},
    )

    score = EvidencePipeline.score_relevance_to_query(source, "核心交易系统迁移合规风险")

    assert score > 0.0


def test_multilingual_coverage_uses_grounding_queries_and_domain_terms() -> None:
    pipeline = EvidencePipeline()
    sources = [
        SourceRecord(
            source_id="SRC-1",
            provider="tavily",
            source_type="webpage",
            title="Trading system microservices migration compliance controls",
            url="https://example.com/trading",
            source="https://example.com/trading",
            domain="example.com",
            snippet="Trading system microservices migration requires compliance controls and audit logging.",
            content="Trading system microservices migration requires compliance controls and audit logging.",
            metadata={"grounding_query": "financial trading system microservices migration compliance risks data governance"},
        ),
        SourceRecord(
            source_id="SRC-2",
            provider="tavily",
            source_type="webpage",
            title="Disaster recovery RTO RPO comparison",
            url="https://example.com/dr",
            source="https://example.com/dr",
            domain="example.com",
            snippet="Disaster recovery architecture compares RTO RPO metrics and resilience.",
            content="Disaster recovery architecture compares RTO RPO metrics and resilience.",
            metadata={"grounding_query": "disaster recovery architecture microservices trading system RTO RPO comparison"},
        ),
    ]

    result = pipeline.assess(
        sources=sources,
        user_request="评估核心交易系统从单体架构迁移到微服务架构的数据合规风险和容灾能力",
        acceptance_criteria=["明确识别数据合规风险", "量化迁移前后容灾指标变化"],
    )

    assert result.coverage.query_coverage > 0.0
    assert result.coverage.criteria_coverage > 0.0
