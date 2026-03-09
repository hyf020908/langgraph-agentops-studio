from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from services.runtime import AgentRuntime
from tools.evidence import build_evidence_ranker_tool
from tools.parsers import build_review_formatter_tool, build_source_parser_tool
from tools.research_grounding import build_research_grounding_tool
from tools.retrieval import build_retrieval_tool
from tools.storage import build_artifact_export_tool, build_local_storage_tool
from tools.tracing import build_trace_logger_tool
from tools.web_reader import build_web_reader_tool
from tools.web_search import build_web_search_tool
from tools.writing import build_report_writer_tool


@dataclass(slots=True)
class ToolRegistry:
    research_grounding_tool: BaseTool
    retrieval_tool: BaseTool
    web_search_tool: BaseTool
    web_reader_tool: BaseTool
    source_parser_tool: BaseTool
    evidence_ranker_tool: BaseTool
    report_writer_tool: BaseTool
    review_formatter_tool: BaseTool
    artifact_export_tool: BaseTool
    trace_logger_tool: BaseTool
    local_storage_tool: BaseTool

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = getattr(self, tool_name).invoke(payload)
        return json.loads(raw) if isinstance(raw, str) else raw


def build_tool_registry(runtime: AgentRuntime) -> ToolRegistry:
    return ToolRegistry(
        research_grounding_tool=build_research_grounding_tool(runtime),
        retrieval_tool=build_retrieval_tool(runtime),
        web_search_tool=build_web_search_tool(runtime),
        web_reader_tool=build_web_reader_tool(runtime),
        source_parser_tool=build_source_parser_tool(),
        evidence_ranker_tool=build_evidence_ranker_tool(),
        report_writer_tool=build_report_writer_tool(),
        review_formatter_tool=build_review_formatter_tool(),
        artifact_export_tool=build_artifact_export_tool(runtime),
        trace_logger_tool=build_trace_logger_tool(),
        local_storage_tool=build_local_storage_tool(runtime),
    )
