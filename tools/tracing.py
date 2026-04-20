from __future__ import annotations

# Trace-event tool.
# Most trace events are emitted directly by nodes, but the research subgraph
# also records dispatch metadata through a tool so the event appears in the same
# message-driven collection path as other research outputs.

import json
from datetime import UTC, datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TraceLoggerInput(BaseModel):
    node: str
    message: str
    metadata: dict = Field(default_factory=dict)


def build_trace_logger_tool():
    @tool("trace_logger_tool", args_schema=TraceLoggerInput)
    def trace_logger_tool(node: str, message: str, metadata: dict | None = None) -> str:
        """Generate a structured trace event payload."""
        return json.dumps(
            {
                "trace_event": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "node": node,
                    "status": "observed",
                    "message": message,
                    "metadata": metadata or {},
                }
            }
        )

    return trace_logger_tool
