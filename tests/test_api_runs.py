from __future__ import annotations

from fastapi.testclient import TestClient

import app.api as api_module
from app.runner import RunNotFoundError
from schemas.models import ModelCallRecord, RunResponse, TraceEvent


class _InspectRunner:
    def inspect(self, task_id: str):
        return {"task_id": task_id, "status": "planned"}, None

    def summarize(self, state: dict, interrupt_payload: dict | None = None) -> RunResponse:
        return RunResponse(
            task_id=state["task_id"],
            status=state["status"],
            approval_required=False,
            execution_trace=[
                TraceEvent(
                    timestamp="2026-01-01T00:00:00Z",
                    node="planner_agent",
                    status="completed",
                    message="planned",
                )
            ],
            model_call_history=[
                ModelCallRecord(
                    node="planner_agent",
                    provider="test-provider",
                    model="test-model",
                    status="success",
                    output_preview='{"plan":[]}',
                )
            ],
            next_nodes=["research_pipeline"],
        )


class _MissingRunner:
    def inspect(self, task_id: str):
        raise RunNotFoundError(f"Run '{task_id}' has not produced a checkpoint yet.")


def test_get_run_returns_checkpoint_observability(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_runner", _InspectRunner())
    client = TestClient(api_module.app)

    response = client.get("/runs/task-live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_trace"][0]["node"] == "planner_agent"
    assert payload["model_call_history"][0]["output_preview"] == '{"plan":[]}'
    assert payload["next_nodes"] == ["research_pipeline"]


def test_get_run_returns_404_before_first_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_runner", _MissingRunner())
    client = TestClient(api_module.app)

    response = client.get("/runs/not-visible")

    assert response.status_code == 404
    assert "has not produced a checkpoint" in response.json()["detail"]
