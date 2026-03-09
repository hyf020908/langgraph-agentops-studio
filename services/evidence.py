from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Iterable

from schemas.models import (
    CitationRecord,
    ConflictRecord,
    CoverageRecord,
    EvidenceAssessment,
    EvidenceRecord,
    EvidenceScoreBreakdown,
    SourceRecord,
    SupportRecord,
)


@dataclass(slots=True)
class EvidencePipelineResult:
    ranked_evidence: list[EvidenceRecord]
    assessments: list[EvidenceAssessment]
    conflicts: list[ConflictRecord]
    supports: list[SupportRecord]
    coverage: CoverageRecord


class EvidencePipeline:
    def assess(
        self,
        *,
        sources: list[SourceRecord],
        user_request: str,
        acceptance_criteria: list[str] | None = None,
    ) -> EvidencePipelineResult:
        normalized = [self.normalize_source_record(item) for item in sources]
        conflicts, supports = self._detect_relations(normalized)

        assessments: list[EvidenceAssessment] = []
        for source in normalized:
            breakdown = self._build_score_breakdown(
                source=source,
                user_request=user_request,
                supports=supports,
                conflicts=conflicts,
            )
            overall_score = self._combine_score(breakdown)
            assessment = EvidenceAssessment(
                source_id=source.source_id,
                overall_score=overall_score,
                score_breakdown=breakdown,
                strengths=self._collect_strengths(breakdown),
                weaknesses=self._collect_weaknesses(breakdown),
                flags=self._collect_flags(source, breakdown),
                conflicts_with=[
                    record.right_source_id
                    for record in conflicts
                    if record.left_source_id == source.source_id
                ],
                supports=[
                    record.supports_source_id
                    for record in supports
                    if record.source_id == source.source_id
                ],
            )
            assessments.append(assessment)

        evidence_records = self._to_evidence_records(normalized, assessments, user_request)
        evidence_records.sort(key=lambda item: item.confidence, reverse=True)
        for index, item in enumerate(evidence_records, start=1):
            item.evidence_id = f"EVD-{index:02d}"

        coverage = self._build_coverage_record(
            user_request=user_request,
            acceptance_criteria=acceptance_criteria or [],
            evidence=evidence_records,
        )
        return EvidencePipelineResult(
            ranked_evidence=evidence_records,
            assessments=sorted(assessments, key=lambda item: item.overall_score, reverse=True),
            conflicts=conflicts,
            supports=supports,
            coverage=coverage,
        )

    @staticmethod
    def normalize_source_record(source: SourceRecord) -> SourceRecord:
        snippet = source.snippet.strip() if source.snippet else ""
        content = source.content.strip() if source.content else None
        if not snippet and content:
            snippet = content[:500]
        source.snippet = snippet
        source.content = content
        return source

    def _build_score_breakdown(
        self,
        *,
        source: SourceRecord,
        user_request: str,
        supports: list[SupportRecord],
        conflicts: list[ConflictRecord],
    ) -> EvidenceScoreBreakdown:
        relevance = self.score_relevance_to_query(source, user_request)
        source_credibility = self.score_source_quality(source)
        recency = self.score_recency(source.published_at)
        completeness = self.score_completeness(source)
        extraction_quality = self.score_extraction_quality(source)
        actionability = self.score_actionability(source, user_request)

        support_strength = [
            item.strength
            for item in supports
            if item.source_id == source.source_id or item.supports_source_id == source.source_id
        ]
        conflict_severity = [
            item.severity
            for item in conflicts
            if item.left_source_id == source.source_id or item.right_source_id == source.source_id
        ]
        corroboration = min(1.0, mean(support_strength) if support_strength else 0.0)
        contradiction_penalty = min(1.0, max(conflict_severity) if conflict_severity else 0.0)

        return EvidenceScoreBreakdown(
            relevance=relevance,
            source_credibility=source_credibility,
            recency=recency,
            completeness=completeness,
            corroboration=corroboration,
            contradiction_penalty=contradiction_penalty,
            extraction_quality=extraction_quality,
            actionability=actionability,
        )

    @staticmethod
    def score_relevance_to_query(source: SourceRecord, user_request: str) -> float:
        source_tokens = set(_tokenize(" ".join([source.title, source.snippet, source.content or ""])))
        query_tokens = set(_tokenize(user_request))
        if not source_tokens or not query_tokens:
            return 0.0
        overlap = len(source_tokens & query_tokens)
        return min(1.0, overlap / max(1, len(query_tokens)))

    @staticmethod
    def score_source_quality(source: SourceRecord) -> float:
        provider_weight = {
            "qdrant": 0.78,
            "tavily": 0.7,
            "exa": 0.74,
            "jina": 0.66,
        }
        baseline = provider_weight.get(str(source.provider), 0.6)
        return min(1.0, max(0.0, mean([baseline, source.credibility])))

    @staticmethod
    def score_recency(published_at: str) -> float:
        if not published_at or published_at == "unknown":
            return 0.45
        dt = _parse_datetime(published_at)
        if dt is None:
            return 0.45
        age_days = (datetime.now(UTC) - dt).days
        if age_days <= 30:
            return 0.95
        if age_days <= 90:
            return 0.85
        if age_days <= 180:
            return 0.72
        if age_days <= 365:
            return 0.6
        return 0.45

    @staticmethod
    def score_completeness(source: SourceRecord) -> float:
        text = source.content or source.snippet or ""
        length = len(text)
        if length >= 3000:
            return 0.95
        if length >= 1800:
            return 0.84
        if length >= 900:
            return 0.72
        if length >= 350:
            return 0.58
        return 0.4

    @staticmethod
    def score_extraction_quality(source: SourceRecord) -> float:
        text = source.content or source.snippet or ""
        if not text:
            return 0.0
        symbol_ratio = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(1, len(text))
        line_count = text.count("\n") + 1
        if symbol_ratio > 0.25:
            return 0.45
        if line_count < 2:
            return 0.58
        return 0.82

    @staticmethod
    def score_actionability(source: SourceRecord, user_request: str) -> float:
        text = " ".join([source.title, source.snippet, source.content or "", user_request]).lower()
        markers = ["should", "risk", "impact", "cost", "trade-off", "recommend", "decision", "mitigation"]
        hits = sum(1 for marker in markers if marker in text)
        return min(1.0, hits / len(markers) * 1.6)

    @staticmethod
    def _combine_score(breakdown: EvidenceScoreBreakdown) -> float:
        positive = (
            0.2 * breakdown.relevance
            + 0.16 * breakdown.source_credibility
            + 0.1 * breakdown.recency
            + 0.14 * breakdown.completeness
            + 0.12 * breakdown.corroboration
            + 0.12 * breakdown.extraction_quality
            + 0.16 * breakdown.actionability
        )
        penalty = 0.2 * breakdown.contradiction_penalty
        return max(0.0, min(1.0, positive - penalty))

    def _to_evidence_records(
        self,
        sources: list[SourceRecord],
        assessments: list[EvidenceAssessment],
        user_request: str,
    ) -> list[EvidenceRecord]:
        assessment_by_id = {item.source_id: item for item in assessments}
        evidence: list[EvidenceRecord] = []
        for source in sources:
            assessment = assessment_by_id[source.source_id]
            evidence.append(
                EvidenceRecord(
                    evidence_id=f"EVD-{len(evidence) + 1:02d}",
                    claim=f"{source.title} contributes to task '{user_request}'.",
                    supporting_sources=[source.source_id],
                    confidence=assessment.overall_score,
                    risk_flags=assessment.flags,
                    summary=source.snippet,
                    citations=[
                        CitationRecord(
                            source_id=source.source_id,
                            provider=source.provider,
                            title=source.title,
                            source=source.source or source.url,
                            url=source.url,
                            chunk_id=source.chunk_id,
                            score=source.score,
                            retrieved_at=source.retrieved_at,
                        )
                    ],
                    assessment=assessment,
                )
            )
        return evidence

    def _build_coverage_record(
        self,
        *,
        user_request: str,
        acceptance_criteria: list[str],
        evidence: list[EvidenceRecord],
    ) -> CoverageRecord:
        request_tokens = set(_tokenize(user_request))
        evidence_tokens = set()
        for item in evidence:
            evidence_tokens.update(_tokenize(item.summary))
            evidence_tokens.update(_tokenize(item.claim))

        query_coverage = len(request_tokens & evidence_tokens) / max(1, len(request_tokens)) if request_tokens else 0.0

        criteria_scores: list[float] = []
        for criterion in acceptance_criteria:
            criterion_tokens = set(_tokenize(criterion))
            if not criterion_tokens:
                continue
            overlap = len(criterion_tokens & evidence_tokens) / len(criterion_tokens)
            criteria_scores.append(overlap)
        criteria_coverage = mean(criteria_scores) if criteria_scores else query_coverage

        notes = []
        if query_coverage < 0.5:
            notes.append("query coverage is below target")
        if criteria_coverage < 0.5:
            notes.append("acceptance criteria coverage is below target")

        return CoverageRecord(
            query_coverage=min(1.0, max(0.0, query_coverage)),
            criteria_coverage=min(1.0, max(0.0, criteria_coverage)),
            evidence_count=len(evidence),
            coverage_notes=notes,
        )

    def _detect_relations(self, sources: list[SourceRecord]) -> tuple[list[ConflictRecord], list[SupportRecord]]:
        conflicts: list[ConflictRecord] = []
        supports: list[SupportRecord] = []

        for index, left in enumerate(sources):
            left_tokens = set(_tokenize(" ".join([left.title, left.snippet, left.content or ""])))
            for right in sources[index + 1 :]:
                right_tokens = set(_tokenize(" ".join([right.title, right.snippet, right.content or ""])))
                if not left_tokens or not right_tokens:
                    continue
                overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

                left_negative = _contains_negative_signal(left.snippet + " " + (left.content or ""))
                right_negative = _contains_negative_signal(right.snippet + " " + (right.content or ""))

                if overlap >= 0.22 and left_negative == right_negative:
                    supports.append(
                        SupportRecord(
                            source_id=left.source_id,
                            supports_source_id=right.source_id,
                            strength=min(1.0, overlap * 2.4),
                            reason="semantic overlap with aligned stance",
                        )
                    )
                    supports.append(
                        SupportRecord(
                            source_id=right.source_id,
                            supports_source_id=left.source_id,
                            strength=min(1.0, overlap * 2.4),
                            reason="semantic overlap with aligned stance",
                        )
                    )

                if overlap >= 0.15 and left_negative != right_negative:
                    severity = min(1.0, 0.45 + overlap)
                    conflicts.append(
                        ConflictRecord(
                            left_source_id=left.source_id,
                            right_source_id=right.source_id,
                            severity=severity,
                            reason="semantic overlap with opposing stance",
                        )
                    )
                    conflicts.append(
                        ConflictRecord(
                            left_source_id=right.source_id,
                            right_source_id=left.source_id,
                            severity=severity,
                            reason="semantic overlap with opposing stance",
                        )
                    )

        return conflicts, supports

    @staticmethod
    def _collect_strengths(breakdown: EvidenceScoreBreakdown) -> list[str]:
        strengths: list[str] = []
        if breakdown.relevance >= 0.65:
            strengths.append("high query relevance")
        if breakdown.source_credibility >= 0.7:
            strengths.append("credible source profile")
        if breakdown.completeness >= 0.7:
            strengths.append("deep content coverage")
        if breakdown.corroboration >= 0.55:
            strengths.append("corroborated by related evidence")
        if breakdown.actionability >= 0.6:
            strengths.append("decision-useful content")
        return strengths

    @staticmethod
    def _collect_weaknesses(breakdown: EvidenceScoreBreakdown) -> list[str]:
        weaknesses: list[str] = []
        if breakdown.recency < 0.55:
            weaknesses.append("limited recency")
        if breakdown.extraction_quality < 0.6:
            weaknesses.append("extraction quality is moderate")
        if breakdown.contradiction_penalty > 0.5:
            weaknesses.append("conflicts with other high-overlap evidence")
        if breakdown.relevance < 0.5:
            weaknesses.append("limited direct relevance")
        return weaknesses

    @staticmethod
    def _collect_flags(source: SourceRecord, breakdown: EvidenceScoreBreakdown) -> list[str]:
        flags: list[str] = []
        if breakdown.contradiction_penalty > 0.55:
            flags.append("high_conflict")
        if breakdown.completeness < 0.5:
            flags.append("low_content_depth")
        if source.source_type == "web_search":
            flags.append("requires_reader_validation")
        if breakdown.recency < 0.5:
            flags.append("stale_information")
        return flags


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if len(token) > 2]


def _contains_negative_signal(text: str) -> bool:
    lowered = text.lower()
    markers = ["risk", "issue", "limitation", "failed", "problem", "warning", "not recommended"]
    return any(marker in lowered for marker in markers)


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None

    variants = [
        text,
        text.replace("Z", "+00:00"),
    ]
    for item in variants:
        try:
            parsed = datetime.fromisoformat(item)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    for pattern in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, pattern).replace(tzinfo=UTC)
            return parsed
        except ValueError:
            continue
    return None


def mean_or_default(values: Iterable[float], default: float) -> float:
    nums = [value for value in values]
    return mean(nums) if nums else default
