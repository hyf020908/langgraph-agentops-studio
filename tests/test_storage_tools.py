from __future__ import annotations

import json

from services.storage import LocalArtifactStore
from tools.storage import build_artifact_export_tool


class _Runtime:
    def __init__(self, root) -> None:
        self.storage = LocalArtifactStore(root)


def test_artifact_export_rewrites_files_with_final_artifacts(tmp_path) -> None:
    tool = build_artifact_export_tool(_Runtime(tmp_path))
    state = {
        "task_id": "task",
        "user_request": "Assess migration",
        "status": "completed",
        "execution_trace": [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "node": "executor_agent",
                "status": "completed",
                "message": "Exported final artifacts and persisted local snapshot.",
                "metadata": {"artifact_count": 0},
            }
        ],
        "tool_call_history": [],
        "artifacts": [],
    }

    response = json.loads(tool.invoke({"task_id": "task", "state_payload": state}))
    run_dir = tmp_path / "task"
    trace = json.loads((run_dir / "workflow_trace.json").read_text(encoding="utf-8"))
    run_artifact = json.loads((run_dir / "run_artifact.json").read_text(encoding="utf-8"))
    summary = (run_dir / "task_summary.html").read_text(encoding="utf-8")

    assert len(response["artifacts"]) == 7
    assert trace["trace"][-1]["metadata"]["artifact_count"] == 7
    assert len(run_artifact["artifacts"]) == 7
    assert "final_report.md" in summary
