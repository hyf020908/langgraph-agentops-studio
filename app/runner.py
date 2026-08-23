from __future__ import annotations

# Programmatic workflow entrypoint.
# `WorkflowRunner` is the thin application layer used by the CLI and FastAPI
# adapter. It owns graph compilation, run start/continue semantics, and
# translation between raw graph state and API-friendly summaries.

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command

from artifacts.exporter import render_final_report
from graph.builder import build_agent_graph
from schemas.models import ModelCallRecord, RunResponse, ToolCallRecord, TraceEvent
from schemas.state import initial_state
from services.runtime import AgentRuntime, build_runtime


class RunNotFoundError(LookupError):
    """Raised when a checkpoint thread has not produced any state yet."""


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
        # Every run gets a checkpoint thread keyed by `task_id` so interrupts can
        # be resumed later by CLI or API callers.
        state = initial_state(user_request=task, task_id=task_id, task_type=task_type)
        config = {"configurable": {"thread_id": state["task_id"]}}
        result = self.graph.invoke(state, config=config)
        interrupt_payload = self._extract_interrupt_payload(result)
        while interrupt_payload is not None and auto_approve:
            # Auto-approval exists only for local/demo flows; normal production
            # control resumes from the stored checkpoint with a human decision.
            decision = {
                "approved": True,
                "reviewer": "cli-auto-approver",
                "rationale": "Auto-approved for local demonstration run.",
            }
            result = self.graph.invoke(Command(resume=decision), config=config)
            interrupt_payload = self._extract_interrupt_payload(result)
        return result, interrupt_payload

    def continue_run(self, task_id: str, approved: bool, reviewer: str, rationale: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        # Resume feeds the approval payload back into the interrupted `human_review`
        # node; LangGraph restores the rest of the checkpointed state for us.
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

    def inspect(self, task_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Read the latest checkpoint without advancing the LangGraph thread."""
        config = {"configurable": {"thread_id": task_id}}
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict) or not values or values.get("task_id") != task_id:
            raise RunNotFoundError(f"Run '{task_id}' has not produced a checkpoint yet.")

        state = dict(values)
        state["_next_nodes"] = list(getattr(snapshot, "next", ()) or ())
        checkpoint_timestamp = getattr(snapshot, "created_at", None) or datetime.now(UTC).isoformat()
        state["_checkpoint_trace_events"] = [
            TraceEvent(
                timestamp=checkpoint_timestamp,
                node=str(getattr(task, "name", "langgraph_task")),
                status="error",
                message="LangGraph task execution failed.",
                metadata={"error": str(getattr(task, "error"))},
            )
            for task in getattr(snapshot, "tasks", ()) or ()
            if getattr(task, "error", None)
        ]
        interrupt_payload = self._extract_snapshot_interrupt_payload(snapshot)
        return state, interrupt_payload

    def summarize(self, state: dict[str, Any], interrupt_payload: dict[str, Any] | None = None) -> RunResponse:
        # The response retains the compact lifecycle fields while exposing the
        # bounded, audit-safe observability ledgers needed by the frontend.
        artifact_paths = [
            str(artifact.get("path", "")) if isinstance(artifact, dict) else str(artifact.path)
            for artifact in state.get("artifacts", [])
            if (artifact.get("path") if isinstance(artifact, dict) else getattr(artifact, "path", None))
        ]
        review = state.get("review_feedback")
        review_summary = (
            str(review.get("summary", "")) if isinstance(review, dict) else str(review.summary)
        ) if review else None
        try:
            raw_trace = self.runtime.context_compaction.merge_archived_trace(state)
        except Exception:
            raw_trace = state.get("execution_trace", [])
        raw_trace = list(raw_trace) + list(state.get("_checkpoint_trace_events", []))
        execution_trace = [
            item if isinstance(item, TraceEvent) else TraceEvent.model_validate(item)
            for item in raw_trace
        ]
        tool_calls = [
            item if isinstance(item, ToolCallRecord) else ToolCallRecord.model_validate(item)
            for item in state.get("tool_call_history", [])
        ]
        model_calls = [
            item if isinstance(item, ModelCallRecord) else ModelCallRecord.model_validate(item)
            for item in state.get("model_call_history", [])
        ]
        draft_report = state.get("draft_report") or None
        final_report = None
        if state.get("status") == "completed":
            final_report = render_final_report(state)
        return RunResponse(
            task_id=state["task_id"],
            status=state.get("status", "unknown"),
            approval_required=interrupt_payload is not None,
            approval_payload=interrupt_payload,
            artifact_paths=artifact_paths,
            review_summary=review_summary,
            execution_trace=execution_trace,
            tool_call_history=tool_calls,
            model_call_history=model_calls,
            draft_report=draft_report,
            final_report=final_report,
            next_nodes=list(state.get("_next_nodes", [])),
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
        # LangGraph stores interrupts under `__interrupt__`; callers only need
        # the first payload because this workflow pauses at a single human gate.
        if "__interrupt__" not in result:
            return None
        interrupts = result["__interrupt__"]
        if not interrupts:
            return None
        payload = getattr(interrupts[0], "value", interrupts[0])
        return payload if isinstance(payload, dict) else {"payload": payload}

    @staticmethod
    def _extract_snapshot_interrupt_payload(snapshot: Any) -> dict[str, Any] | None:
        interrupts = list(getattr(snapshot, "interrupts", ()) or ())
        # LangGraph versions in the supported range may expose pending
        # interrupts only on PregelTask entries rather than on StateSnapshot.
        for task in getattr(snapshot, "tasks", ()) or ():
            interrupts.extend(getattr(task, "interrupts", ()) or ())
        if not interrupts:
            return None
        payload = getattr(interrupts[0], "value", interrupts[0])
        return payload if isinstance(payload, dict) else {"payload": payload}
