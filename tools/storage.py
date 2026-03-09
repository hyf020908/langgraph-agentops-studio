from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from artifacts.exporter import (
    build_decision_record,
    build_mermaid_diagram,
    build_run_artifact,
    build_task_summary_html,
    build_workflow_trace,
    render_final_report,
    render_review_feedback,
)
from schemas.models import ApprovalDecision, ReviewFeedback
from services.runtime import AgentRuntime


class ArtifactExportInput(BaseModel):
    task_id: str
    state_payload: dict[str, Any]


class LocalStorageInput(BaseModel):
    task_id: str
    file_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


def build_artifact_export_tool(runtime: AgentRuntime):
    @tool("artifact_export_tool", args_schema=ArtifactExportInput)
    def artifact_export_tool(task_id: str, state_payload: dict[str, Any]) -> str:
        """Render and persist markdown, JSON, Mermaid, and HTML artifacts for a run."""
        state = dict(state_payload)
        if state.get("review_feedback"):
            state["review_feedback"] = ReviewFeedback.model_validate(state["review_feedback"])
        if state.get("approval_decision"):
            state["approval_decision"] = ApprovalDecision.model_validate(state["approval_decision"])
        final_report_path = runtime.storage.write_text(task_id, "final_report.md", render_final_report(state))
        decision_path = runtime.storage.write_json(task_id, "decision_record.json", build_decision_record(state))
        trace_path = runtime.storage.write_json(task_id, "workflow_trace.json", build_workflow_trace(state))
        review_path = runtime.storage.write_text(
            task_id,
            "review_feedback.md",
            render_review_feedback(state.get("review_feedback")),
        )
        diagram_path = runtime.storage.write_text(task_id, "diagram.mmd", build_mermaid_diagram())
        summary_path = runtime.storage.write_text(task_id, "task_summary.html", build_task_summary_html(state))
        run_artifact_path = runtime.storage.write_json(task_id, "run_artifact.json", build_run_artifact(state))
        payload = {
            "artifacts": [
                {"name": "final_report.md", "path": str(final_report_path), "media_type": "text/markdown"},
                {"name": "decision_record.json", "path": str(decision_path), "media_type": "application/json"},
                {"name": "workflow_trace.json", "path": str(trace_path), "media_type": "application/json"},
                {"name": "review_feedback.md", "path": str(review_path), "media_type": "text/markdown"},
                {"name": "diagram.mmd", "path": str(diagram_path), "media_type": "text/plain"},
                {"name": "task_summary.html", "path": str(summary_path), "media_type": "text/html"},
                {"name": "run_artifact.json", "path": str(run_artifact_path), "media_type": "application/json"},
            ]
        }
        return json.dumps(payload)

    return artifact_export_tool


def build_local_storage_tool(runtime: AgentRuntime):
    @tool("local_storage_tool", args_schema=LocalStorageInput)
    def local_storage_tool(task_id: str, file_name: str, payload: dict[str, Any]) -> str:
        """Persist an arbitrary JSON snapshot for local replay and auditing."""
        path = runtime.storage.write_json(task_id, file_name, payload)
        return json.dumps({"path": str(path)})

    return local_storage_tool

