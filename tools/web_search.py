from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.runtime import AgentRuntime


class WebSearchInput(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)


def build_web_search_tool(runtime: AgentRuntime):
    @tool("web_search_tool", args_schema=WebSearchInput)
    def web_search_tool(query: str, max_results: int = 5) -> str:
        """Run real web search through the configured provider (Tavily or Exa)."""
        if runtime.web_search_provider is None:
            raise RuntimeError("Web search is disabled. Set ENABLE_WEB_SEARCH=true and configure a provider.")
        results = runtime.web_search_provider.search(query=query, max_results=max_results)
        return json.dumps(
            {
                "query": query,
                "provider": runtime.web_search_provider.name,
                "results": [item.model_dump() for item in results],
            }
        )

    return web_search_tool
