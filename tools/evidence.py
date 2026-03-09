from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from schemas.models import SourceRecord
from services.evidence import EvidencePipeline


class EvidenceRankerInput(BaseModel):
    sources: list[SourceRecord]
    user_request: str
    acceptance_criteria: list[str] = Field(default_factory=list)


def rank_evidence(
    sources: list[dict],
    user_request: str,
    acceptance_criteria: list[str] | None = None,
) -> dict:
    source_models = [SourceRecord.model_validate(item) for item in sources]
    pipeline = EvidencePipeline()
    result = pipeline.assess(
        sources=source_models,
        user_request=user_request,
        acceptance_criteria=acceptance_criteria or [],
    )
    return {
        "ranked_evidence": [item.model_dump() for item in result.ranked_evidence],
        "evidence_assessments": [item.model_dump() for item in result.assessments],
        "conflicts": [item.model_dump() for item in result.conflicts],
        "supports": [item.model_dump() for item in result.supports],
        "coverage": result.coverage.model_dump(),
    }


def build_evidence_ranker_tool():
    @tool("evidence_ranker_tool", args_schema=EvidenceRankerInput)
    def evidence_ranker_tool(
        sources: list[dict],
        user_request: str,
        acceptance_criteria: list[str] | None = None,
    ) -> str:
        """Evaluate and rank normalized sources into evidence records with detailed assessments."""
        return json.dumps(
            rank_evidence(
                sources=sources,
                user_request=user_request,
                acceptance_criteria=acceptance_criteria,
            )
        )

    return evidence_ranker_tool
