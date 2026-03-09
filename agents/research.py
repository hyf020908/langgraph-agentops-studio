from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from schemas.models import (
    ConflictRecord,
    CoverageRecord,
    ErrorInfo,
    EvidenceAssessment,
    EvidenceRecord,
    SearchResult,
    SourceRecord,
    SupportRecord,
    ToolCallRecord,
    TraceEvent,
)
from services.runtime import AgentRuntime


def _tool_messages(messages, name: str) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage) and getattr(message, "name", None) == name]


def _dedupe_key(item: dict) -> str:
    return f"{item.get('provider', 'unknown')}|{item.get('source_type', 'unknown')}|{item.get('url', '')}|{item.get('chunk_id', '')}"


def build_research_briefing_node(runtime: AgentRuntime):
    def research_briefing(state):
        queries = state.get("search_queries", [])[: runtime.settings.max_search_results]
        retry_count = state.get("retry_count", 0)
        if retry_count > 0:
            queries = [f"{query} implementation risks and mitigations" for query in queries]

        tool_calls = [
            {
                "name": "trace_logger_tool",
                "args": {
                    "node": "research_pipeline",
                    "message": "Dispatching research grounding queries.",
                    "metadata": {
                        "queries": queries,
                        "enable_web_search": runtime.settings.web_grounding.enable_web_search,
                        "enable_vector_rag": runtime.settings.web_grounding.enable_vector_rag,
                        "mode": runtime.settings.web_grounding.mode,
                    },
                },
                "id": "trace-research-briefing",
                "type": "tool_call",
            }
        ]

        for index, query in enumerate(queries, start=1):
            tool_calls.append(
                {
                    "name": "research_grounding_tool",
                    "args": {
                        "query": query,
                        "top_k": runtime.settings.web_grounding.web_search_top_k,
                    },
                    "id": f"grounding-{index}",
                    "type": "tool_call",
                }
            )

        trace = runtime.trace(
            node="research_briefing",
            status="dispatching_tools",
            message="Prepared research grounding tool calls.",
            metadata={
                "query_count": len(queries),
                "retry_count": retry_count,
            },
        )
        return {
            "messages": [AIMessage(content="Dispatching research grounding tools.", tool_calls=tool_calls)],
            "sender": "research_agent",
            "status": "researching",
            "execution_trace": [trace],
        }

    return research_briefing


def build_parse_sources_node(runtime: AgentRuntime):
    def parse_sources(state):
        grounding_messages = _tool_messages(state.get("messages", []), "research_grounding_tool")

        aggregated_results = []
        for message in grounding_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            aggregated_results.extend(payload.get("results", []))

        deduped = {_dedupe_key(item): SearchResult.model_validate(item).model_dump() for item in aggregated_results}

        tool_calls = [
            {
                "name": "source_parser_tool",
                "args": {"results": list(deduped.values())},
                "id": "source-parser-1",
                "type": "tool_call",
            }
        ]
        trace = runtime.trace(
            node="parse_sources",
            status="dispatching_tools",
            message="Prepared source parser tool call from grounding outputs.",
            metadata={
                "grounding_message_count": len(grounding_messages),
                "deduped_result_count": len(deduped),
            },
        )
        return {
            "messages": [AIMessage(content="Normalizing source metadata.", tool_calls=tool_calls)],
            "sender": "research_agent",
            "execution_trace": [trace],
        }

    return parse_sources


def build_rank_evidence_node(runtime: AgentRuntime):
    def rank_evidence(state):
        parser_messages = _tool_messages(state.get("messages", []), "source_parser_tool")
        parsed_sources = []
        for message in parser_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            parsed_sources.extend(payload.get("sources", []))
        deduped = {_dedupe_key(item): SourceRecord.model_validate(item).model_dump() for item in parsed_sources}
        tool_calls = [
            {
                "name": "evidence_ranker_tool",
                "args": {
                    "sources": list(deduped.values()),
                    "user_request": state["user_request"],
                    "acceptance_criteria": state.get("acceptance_criteria", []),
                },
                "id": "evidence-ranker-1",
                "type": "tool_call",
            }
        ]
        trace = runtime.trace(
            node="rank_evidence",
            status="dispatching_tools",
            message="Prepared evidence ranker tool call.",
            metadata={"source_count": len(deduped)},
        )
        return {
            "messages": [AIMessage(content="Ranking evidence candidates.", tool_calls=tool_calls)],
            "sender": "research_agent",
            "execution_trace": [trace],
        }

    return rank_evidence


def build_collect_research_node(runtime: AgentRuntime):
    def collect_research(state):
        parser_messages = _tool_messages(state.get("messages", []), "source_parser_tool")
        ranker_messages = _tool_messages(state.get("messages", []), "evidence_ranker_tool")
        grounding_messages = _tool_messages(state.get("messages", []), "research_grounding_tool")
        trace_messages = _tool_messages(state.get("messages", []), "trace_logger_tool")

        source_records = []
        for message in parser_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            source_records.extend(payload.get("sources", []))

        evidence_records = []
        assessment_records = []
        conflict_records = []
        support_records = []
        coverage_record = None
        for message in ranker_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            evidence_records.extend(payload.get("ranked_evidence", []))
            assessment_records.extend(payload.get("evidence_assessments", []))
            conflict_records.extend(payload.get("conflicts", []))
            support_records.extend(payload.get("supports", []))
            if payload.get("coverage"):
                coverage_record = payload.get("coverage")

        trace_events = []
        for message in trace_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            if payload.get("trace_event"):
                trace_events.append(TraceEvent.model_validate(payload["trace_event"]))

        tool_history = []
        for message in grounding_messages:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            query = payload.get("query", "")
            result_count = len(payload.get("results", []))
            providers = payload.get("providers", {})
            tool_history.append(
                ToolCallRecord(
                    tool_name="research_grounding_tool",
                    status="success",
                    input_payload={"query": query},
                    output_preview=(
                        f"Merged {result_count} grounded result(s). "
                        f"search={providers.get('web_search')} reader={providers.get('web_reader')}"
                    ),
                )
            )

        tool_history.extend(
            [
                ToolCallRecord(
                    tool_name="source_parser_tool",
                    status="success",
                    input_payload={"result_count": len(source_records)},
                    output_preview="Normalized grounded results into source records.",
                ),
                ToolCallRecord(
                    tool_name="evidence_ranker_tool",
                    status="success" if evidence_records else "error",
                    input_payload={"source_count": len(source_records)},
                    output_preview="Ranked source records into evidence statements." if evidence_records else "",
                    error=None if evidence_records else "No evidence produced by ranker.",
                ),
            ]
        )
        if trace_events:
            tool_history.append(
                ToolCallRecord(
                    tool_name="trace_logger_tool",
                    status="success",
                    input_payload={"node": "research_pipeline"},
                    output_preview="Recorded research dispatch trace event.",
                )
            )

        error_info = None
        status = "research_complete"
        retry_count = state.get("retry_count", 0)
        if not evidence_records:
            retry_count += 1
            status = "research_failed"
            error_info = ErrorInfo(
                stage="research_pipeline",
                message="No ranked evidence generated from grounding outputs.",
                recoverable=retry_count <= runtime.settings.max_retries,
                detail={"retry_count": retry_count},
            )

        source_models = [SourceRecord.model_validate(item) for item in source_records]
        evidence_models = [EvidenceRecord.model_validate(item) for item in evidence_records]
        assessment_models = [EvidenceAssessment.model_validate(item) for item in assessment_records]
        conflict_models = [ConflictRecord.model_validate(item) for item in conflict_records]
        support_models = [SupportRecord.model_validate(item) for item in support_records]
        coverage_model = CoverageRecord.model_validate(coverage_record) if coverage_record else None

        trace = runtime.trace(
            node="collect_research",
            status=status,
            message="Collected research artifacts from tool messages.",
            metadata={
                "source_count": len(source_models),
                "evidence_count": len(evidence_models),
                "conflict_count": len(conflict_models),
            },
        )
        return {
            "retrieved_sources": source_models,
            "retrieved_chunks": source_models,
            "ranked_evidence": evidence_models,
            "evidence_assessments": assessment_models,
            "evidence_conflicts": conflict_models,
            "evidence_supports": support_models,
            "coverage_record": coverage_model,
            "tool_call_history": tool_history,
            "execution_trace": trace_events + [trace],
            "retry_count": retry_count,
            "error_info": error_info,
            "status": status,
            "sender": "research_agent",
        }

    return collect_research
