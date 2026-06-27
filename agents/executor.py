from __future__ import annotations

# Final export node.
# The executor persists the run's user-facing artifacts and a raw JSON snapshot
# after the supervisor determines the workflow has enough information to finish.

from schemas.models import ArtifactRecord
from services.runtime import AgentRuntime
from services.serialization import to_jsonable
from tools.factory import ToolRegistry


def build_executor_node(runtime: AgentRuntime, tools: ToolRegistry):
    def executor_agent(state):
        # Export and snapshot are separate on purpose: one produces curated
        # outputs for readers, the other preserves the full serializable state.
        trace = runtime.trace(
            node="executor_agent",
            status="completed",
            message="Exported final artifacts and persisted local snapshot.",
            metadata={"artifact_count": 0},
        )
        export_state = {
            **state,
            "artifacts": [],
            "status": "completed",
            "sender": "executor_agent",
            "execution_trace": state.get("execution_trace", []) + [trace],
            "error_info": None,
        }
        export_payload = tools.invoke(
            "artifact_export_tool",
            {
                "task_id": state["task_id"],
                "state_payload": to_jsonable(export_state),
            },
        )
        artifacts = [ArtifactRecord.model_validate(item) for item in export_payload["artifacts"]]
        final_trace = trace.model_copy(update={"metadata": {"artifact_count": len(artifacts)}})
        snapshot_state = {
            **export_state,
            "artifacts": artifacts,
            "execution_trace": state.get("execution_trace", []) + [final_trace],
        }
        tools.invoke(
            "local_storage_tool",
            {
                "task_id": state["task_id"],
                "file_name": "state_snapshot.json",
                "payload": to_jsonable(snapshot_state),
            },
        )
        return {
            "artifacts": artifacts,
            "status": "completed",
            "sender": "executor_agent",
            "execution_trace": [final_trace],
            "error_info": None,
        }

    return executor_agent
