from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.types import Command

from graph.builder import build_agent_graph
from schemas.models import RunResponse
from schemas.state import initial_state
from services.runtime import AgentRuntime, build_runtime


class WorkflowRunner:
    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self.runtime = runtime or build_runtime()
        self.graph = build_agent_graph(self.runtime)

    def start(
        self,
        task: str,
        task_id: str | None = None,
        auto_approve: bool = False,
        task_type: str = "general",
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = initial_state(user_request=task, task_id=task_id, task_type=task_type)
        config = {"configurable": {"thread_id": state["task_id"]}}
        result = self.graph.invoke(state, config=config)
        interrupt_payload = self._extract_interrupt_payload(result)
        while interrupt_payload is not None and auto_approve:
            decision = {
                "approved": True,
                "reviewer": "cli-auto-approver",
                "rationale": "Auto-approved for local demonstration run.",
            }
            result = self.graph.invoke(Command(resume=decision), config=config)
            interrupt_payload = self._extract_interrupt_payload(result)
        return result, interrupt_payload

    def continue_run(self, task_id: str, approved: bool, reviewer: str, rationale: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        config = {"configurable": {"thread_id": task_id}}
        result = self.graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "reviewer": reviewer,
                    "rationale": rationale,
                }
            ),
            config=config,
        )
        return result, self._extract_interrupt_payload(result)

    def summarize(self, state: dict[str, Any], interrupt_payload: dict[str, Any] | None = None) -> RunResponse:
        artifact_paths = [artifact.path for artifact in state.get("artifacts", [])]
        review_summary = state.get("review_feedback").summary if state.get("review_feedback") else None
        return RunResponse(
            task_id=state["task_id"],
            status=state.get("status", "unknown"),
            approval_required=interrupt_payload is not None,
            approval_payload=interrupt_payload,
            artifact_paths=artifact_paths,
            review_summary=review_summary,
        )

    @staticmethod
    def read_task_from_example(example_path: str) -> str:
        path = Path(example_path)
        payload = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            import json

            parsed = json.loads(payload)
            return parsed["task"]
        return payload.strip()

    @staticmethod
    def _extract_interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
        if "__interrupt__" not in result:
            return None
        interrupts = result["__interrupt__"]
        if not interrupts:
            return None
        payload = getattr(interrupts[0], "value", interrupts[0])
        return payload if isinstance(payload, dict) else {"payload": payload}
