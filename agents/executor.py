from __future__ import annotations

from schemas.models import ArtifactRecord
from services.runtime import AgentRuntime
from services.serialization import to_jsonable
from tools.factory import ToolRegistry


def build_executor_node(runtime: AgentRuntime, tools: ToolRegistry):
    def executor_agent(state):
        export_payload = tools.invoke(
            "artifact_export_tool",
            {
                "task_id": state["task_id"],
                "state_payload": to_jsonable(state),
            },
        )
        tools.invoke(
            "local_storage_tool",
            {
                "task_id": state["task_id"],
                "file_name": "state_snapshot.json",
                "payload": to_jsonable(state),
            },
        )
        artifacts = [ArtifactRecord.model_validate(item) for item in export_payload["artifacts"]]
        trace = runtime.trace(
            node="executor_agent",
            status="completed",
            message="Exported final artifacts and persisted local snapshot.",
            metadata={"artifact_count": len(artifacts)},
        )
        return {
            "artifacts": artifacts,
            "status": "completed",
            "sender": "executor_agent",
            "execution_trace": [trace],
            "error_info": None,
        }

    return executor_agent

