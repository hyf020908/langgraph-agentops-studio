from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel

from schemas.models import ReviewFeedback, SearchResult, SourceRecord


class SourceParserInput(BaseModel):
    results: list[SearchResult]


class ReviewFormatterInput(BaseModel):
    review_feedback: ReviewFeedback
    task_id: str


def parse_sources(results: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in results:
        result = SearchResult.model_validate(raw)
        score = result.score if result.score is not None else result.credibility
        relevance = max(0.0, min(0.99, float(score)))
        source_locator = result.source or result.url

        sections = [
            f"Provider: {result.provider}",
            f"Source type: {result.source_type}",
            f"Source locator: {source_locator}",
            f"Chunk ID: {result.chunk_id or 'n/a'}",
            f"Evidence snippet: {result.snippet}",
        ]

        records.append(
            SourceRecord(
                source_id=result.source_id,
                provider=result.provider,
                source_type=result.source_type,
                title=result.title,
                url=result.url,
                domain=result.domain,
                snippet=result.snippet,
                content=result.content,
                source=source_locator,
                chunk_id=result.chunk_id,
                score=result.score,
                parsed_sections=sections,
                published_at=result.published_at,
                retrieved_at=result.retrieved_at,
                credibility=result.credibility,
                relevance=relevance,
                metadata=result.metadata,
            ).model_dump()
        )
    return {"sources": records}


def format_review_feedback(review_feedback: dict[str, Any], task_id: str) -> dict[str, Any]:
    review = ReviewFeedback.model_validate(review_feedback)
    question_lines = [f"- {question}" for question in review.questions] or ["- None"]
    revision_lines = [f"- {item}" for item in review.revision_requests] or ["- None"]
    risk_lines = [f"- {risk}" for risk in review.major_risks] or ["- None"]
    lines = [
        f"# Review Feedback for {task_id}",
        "",
        f"Verdict: `{review.verdict}`",
        f"Score: {review.score:.2f}",
        "",
        "## Summary",
        review.summary,
        "",
        "## Questions",
        *question_lines,
        "",
        "## Revision Requests",
        *revision_lines,
        "",
        "## Major Risks",
        *risk_lines,
    ]
    return {"content": "\n".join(lines)}


def build_source_parser_tool():
    @tool("source_parser_tool", args_schema=SourceParserInput)
    def source_parser_tool(results: list[dict[str, Any]]) -> str:
        """Normalize retrieval/search/reader results into structured source records."""
        return json.dumps(parse_sources(results))

    return source_parser_tool


def build_review_formatter_tool():
    @tool("review_formatter_tool", args_schema=ReviewFormatterInput)
    def review_formatter_tool(review_feedback: dict[str, Any], task_id: str) -> str:
        """Convert reviewer feedback into markdown suitable for export."""
        return json.dumps(format_review_feedback(review_feedback=review_feedback, task_id=task_id))

    return review_formatter_tool
