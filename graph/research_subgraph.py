from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.research import (
    build_collect_research_node,
    build_parse_sources_node,
    build_rank_evidence_node,
    build_research_briefing_node,
)
from schemas.state import AgentState
from services.runtime import AgentRuntime
from tools.factory import ToolRegistry


def build_research_subgraph(runtime: AgentRuntime, tools: ToolRegistry):
    graph = StateGraph(AgentState)
    graph.add_node("research_briefing", build_research_briefing_node(runtime))
    graph.add_node(
        "research_tools",
        ToolNode([tools.research_grounding_tool, tools.trace_logger_tool]),
    )
    graph.add_node("parse_sources", build_parse_sources_node(runtime))
    graph.add_node("parser_tools", ToolNode([tools.source_parser_tool]))
    graph.add_node("rank_evidence", build_rank_evidence_node(runtime))
    graph.add_node("ranking_tools", ToolNode([tools.evidence_ranker_tool]))
    graph.add_node("collect_research", build_collect_research_node(runtime))

    graph.add_edge(START, "research_briefing")
    graph.add_edge("research_briefing", "research_tools")
    graph.add_edge("research_tools", "parse_sources")
    graph.add_edge("parse_sources", "parser_tools")
    graph.add_edge("parser_tools", "rank_evidence")
    graph.add_edge("rank_evidence", "ranking_tools")
    graph.add_edge("ranking_tools", "collect_research")
    graph.add_edge("collect_research", END)

    return graph.compile(name="research_pipeline")
