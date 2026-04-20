from __future__ import annotations

# Tool wrapper for hybrid grounding.
# This is the interface the research subgraph uses when it wants retrieval,
# web search, and optional webpage reading merged behind one tool call.

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.runtime import AgentRuntime


class ResearchGroundingInput(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


def build_research_grounding_tool(runtime: AgentRuntime):
    @tool("research_grounding_tool", args_schema=ResearchGroundingInput)
    def research_grounding_tool(query: str, top_k: int = 5) -> str:
        """Run hybrid grounding that combines vector RAG, web search, and webpage reading."""
        # The service returns richer stats/provider metadata than the graph
        # stores directly; that detail is preserved in the tool payload.
        report = runtime.grounding.ground_query(query=query)
        results = report.get("results", [])[:top_k]
        return json.dumps(
            {
                "query": query,
                "results": results,
                "stats": report.get("stats", {}),
                "providers": report.get("providers", {}),
            }
        )

    return research_grounding_tool
