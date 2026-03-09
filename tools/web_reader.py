from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.runtime import AgentRuntime


class WebReaderInput(BaseModel):
    urls: list[str] = Field(default_factory=list)


def build_web_reader_tool(runtime: AgentRuntime):
    @tool("web_reader_tool", args_schema=WebReaderInput)
    def web_reader_tool(urls: list[str]) -> str:
        """Read and clean webpage content through the configured reader provider (Jina or Exa Contents)."""
        if runtime.web_reader_provider is None:
            raise RuntimeError("Web reader is disabled or not configured.")
        payload = runtime.web_reader_provider.read_urls(urls)
        return json.dumps(
            {
                "provider": runtime.web_reader_provider.name,
                "results": {url: page.model_dump() for url, page in payload.items()},
            }
        )

    return web_reader_tool
