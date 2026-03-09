from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel

from services.runtime import AgentRuntime


class IngestionInput(BaseModel):
    source_dir: str | None = None
    recreate_collection: bool = False


def build_ingestion_tool(runtime: AgentRuntime):
    @tool("ingestion_tool", args_schema=IngestionInput)
    def ingestion_tool(source_dir: str | None = None, recreate_collection: bool = False) -> str:
        """Ingest local documents into the configured vector store for retrieval."""
        report = runtime.retrieval.ingest_directory(
            source_dir=source_dir,
            recreate_collection=recreate_collection,
        )
        return json.dumps(report)

    return ingestion_tool
