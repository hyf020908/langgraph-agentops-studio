from __future__ import annotations

# Direct retrieval tool.
# This tool is not part of the default research subgraph path, but it exposes
# vector-store lookup behind the same tool interface used elsewhere.

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.runtime import AgentRuntime


class RetrievalInput(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=20)
    perspective: str = "technical"


def build_retrieval_tool(runtime: AgentRuntime):
    @tool("retrieval_tool", args_schema=RetrievalInput)
    def retrieval_tool(query: str, top_k: int = 4, perspective: str = "technical") -> str:
        """Retrieve top-k chunks from the configured vector store with source metadata."""
        _ = perspective  # reserved for future retrieval profiles
        results = runtime.retrieval.search(query=query, top_k=top_k)
        return json.dumps(
            {
                "query": query,
                "results": results,
                "provider": runtime.vector_store.provider_name,
            }
        )

    return retrieval_tool
