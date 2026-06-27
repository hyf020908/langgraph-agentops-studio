from __future__ import annotations

from datetime import UTC, datetime

from agents.executor import build_executor_node
from schemas.models import TraceEvent


class _Runtime:
    def trace(self, node: str, status: str, message: str, metadata: dict | None = None) -> TraceEvent:
        return TraceEvent(
            timestamp=datetime.now(UTC).isoformat(),
            node=node,
            status=status,
            message=message,
            metadata=metadata or {},
        )


class _Tools:
    def __init__(self) -> None:
        self.export_state = None
        self.snapshot_state = None

    def invoke(self, tool_name: str, payload: dict):
        if tool_name == "artifact_export_tool":
            self.export_state = payload["state_payload"]
            return {
                "artifacts": [
                    {"name": "final_report.md", "path": "runs/task/final_report.md", "media_type": "text/markdown"}
                ]
            }
        if tool_name == "local_storage_tool":
            self.snapshot_state = payload["payload"]
            return {"path": "runs/task/state_snapshot.json"}
        raise AssertionError(f"Unexpected tool: {tool_name}")


def test_executor_exports_completed_state_and_snapshots_artifacts() -> None:
    tools = _Tools()
    node = build_executor_node(_Runtime(), tools)

    result = node(
        {
            "task_id": "task",
            "status": "approved_for_export",
            "execution_trace": [],
            "artifacts": [],
        }
    )

    assert tools.export_state["status"] == "completed"
    assert tools.snapshot_state["status"] == "completed"
    assert tools.snapshot_state["artifacts"][0]["name"] == "final_report.md"
    assert result["status"] == "completed"
    assert result["artifacts"][0].name == "final_report.md"
