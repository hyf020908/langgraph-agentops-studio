from __future__ import annotations

# Tool registry for graph nodes.
# This module is the boundary between LangGraph orchestration and tool
# implementations. Nodes ask for tools by semantic role, while the registry
# hides the concrete construction and JSON decoding details.

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from langchain_core.tools import BaseTool

from schemas.models import ToolCallRecord
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
    # The registry keeps typed references so the graph can either pass the tools
    # into `ToolNode` or invoke them synchronously from normal Python code.
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
    _tool_calls: ContextVar[tuple[ToolCallRecord, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool_calls = ContextVar(f"tool_calls_{id(self)}", default=())

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # LangChain tools return strings by default; callers in this project work
        # with decoded dictionaries to keep state updates explicit and typed.
        timestamp = datetime.now(UTC).isoformat()
        started = monotonic()
        node = _DIRECT_TOOL_NODES.get(tool_name, "unknown")
        try:
            raw = getattr(self, tool_name).invoke(payload)
            result = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            self._append_tool_call(
                ToolCallRecord(
                    timestamp=timestamp,
                    node=node,
                    tool_name=tool_name,
                    status="error",
                    input_payload=bound_tool_payload(payload),
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"duration_ms": round((monotonic() - started) * 1000, 2)},
                )
            )
            raise

        preview = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        self._append_tool_call(
            ToolCallRecord(
                timestamp=timestamp,
                node=node,
                tool_name=tool_name,
                status="success",
                input_payload=bound_tool_payload(payload),
                output_preview=_bounded_text(preview, 2000),
                metadata={
                    "duration_ms": round((monotonic() - started) * 1000, 2),
                    "output_chars": len(preview),
                    "output_truncated": len(preview) > 2000,
                },
            )
        )
        return result

    def drain_tool_call_history(self) -> list[ToolCallRecord]:
        records = list(self._tool_calls.get())
        self._tool_calls.set(())
        return records

    def _append_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.set((*self._tool_calls.get(), record))


def build_tool_registry(runtime: AgentRuntime) -> ToolRegistry:
    # Tools are thin adapters over services/export helpers; the runtime owns the
    # actual provider clients and storage backends they depend on.
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


def drain_tool_call_history(tools: Any) -> list[ToolCallRecord]:
    """Drain registry records while remaining compatible with lightweight test doubles."""
    drain = getattr(tools, "drain_tool_call_history", None)
    if drain is None:
        return []
    return [
        item if isinstance(item, ToolCallRecord) else ToolCallRecord.model_validate(item)
        for item in drain()
    ]


_DIRECT_TOOL_NODES = {
    "report_writer_tool": "analyst_agent",
    "review_formatter_tool": "reviewer_agent",
    "artifact_export_tool": "executor_agent",
    "local_storage_tool": "executor_agent",
}


def _bounded_text(value: Any, limit: int = 400) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def bound_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an audit-friendly tool input without checkpointing full documents."""
    return {str(key): _bounded_value(value, depth=0) for key, value in list(payload.items())[:20]}


def _bounded_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_text(value, 500)
    if depth >= 2:
        if isinstance(value, dict):
            return {"type": "object", "item_count": len(value), "keys": list(map(str, value.keys()))[:12]}
        if isinstance(value, (list, tuple)):
            return {"type": "array", "item_count": len(value)}
        return _bounded_text(value, 200)
    if isinstance(value, dict):
        return {
            str(key): _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:16]
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "array",
            "item_count": len(value),
            "preview": [_bounded_value(item, depth=depth + 1) for item in list(value)[:3]],
        }
    model_dump = getattr(value, "model_dump", None)
    if model_dump is not None:
        return _bounded_value(model_dump(), depth=depth + 1)
    return _bounded_text(value, 200)
