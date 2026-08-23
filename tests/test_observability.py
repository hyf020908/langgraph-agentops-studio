from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agents.research import _collect_tool_call_records
from app.runner import RunNotFoundError, WorkflowRunner
from graph.builder import build_agent_graph
from schemas.models import ModelCallRecord, ToolCallRecord, TraceEvent
from schemas.state import initial_state
from services.config import LLMSettings, Settings
from services.llm import ProviderReasoningEngine
from tools.factory import ToolRegistry


class _Provider:
    name = "test-provider"

    def __init__(self, response: str) -> None:
        self.response = response
        self.settings = LLMSettings(model="trace-model")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response


class _TraceOnlyRuntime:
    """Minimal runtime used to exercise real LangGraph checkpoint semantics."""

    def __init__(self) -> None:
        self.settings = Settings(checkpoint_mode="memory")
        self.reasoning = ProviderReasoningEngine(_Provider(_planner_response()))

    @staticmethod
    def trace(node: str, status: str, message: str, metadata: dict | None = None) -> TraceEvent:
        return TraceEvent(
            timestamp=datetime.now(UTC).isoformat(),
            node=node,
            status=status,
            message=message,
            metadata=metadata or {},
        )


def _planner_response() -> str:
    return json.dumps(
        {
            "plan": [
                {
                    "step_id": f"P{index}",
                    "objective": f"Objective {index}",
                    "owner": "planner_agent",
                    "done_definition": f"Done {index}",
                    "dependencies": [],
                }
                for index in range(1, 5)
            ],
            "acceptance_criteria": ["A", "B", "C"],
            "search_queries": ["Q1", "Q2", "Q3"],
        }
    )


def test_reasoning_engine_records_real_model_input_and_output() -> None:
    engine = ProviderReasoningEngine(_Provider(_planner_response()))

    engine.plan_task("Trace this request")
    records = engine.drain_model_call_history()

    assert len(records) == 1
    record = records[0]
    assert record.node == "planner_agent"
    assert record.provider == "test-provider"
    assert record.model == "trace-model"
    assert record.status == "success"
    assert "Trace this request" in record.input_preview
    assert '"plan"' in record.output_preview
    assert record.metadata["operation"] == "plan_task"
    assert record.metadata["output_chars"] == len(_planner_response())
    assert engine.drain_model_call_history() == []


def test_real_langgraph_checkpoint_contains_planner_trace_and_model_call() -> None:
    runtime = _TraceOnlyRuntime()
    graph = build_agent_graph(runtime)
    task_id = "task-real-checkpoint"
    config = {"configurable": {"thread_id": task_id}}

    result = graph.invoke(
        initial_state("Trace the real graph", task_id=task_id),
        config=config,
        interrupt_after=["planner_agent"],
    )
    snapshot = graph.get_state(config)

    assert result["status"] == "planned"
    assert snapshot.values["task_id"] == task_id
    assert snapshot.next == ("supervisor",)
    assert snapshot.values["model_call_history"][0].node == "planner_agent"
    assert any(event.node == "planner_agent" for event in snapshot.values["execution_trace"])


def test_reasoning_engine_records_invalid_model_output_as_error() -> None:
    engine = ProviderReasoningEngine(_Provider("not-json"))

    with pytest.raises(RuntimeError, match="not valid JSON"):
        engine.plan_task("Bad output")

    record = engine.drain_model_call_history()[0]
    assert record.status == "error"
    assert record.output_preview == "not-json"
    assert record.error is not None


def test_tool_node_messages_become_actual_tool_call_records() -> None:
    messages = [
        AIMessage(
            content="dispatch",
            tool_calls=[
                {
                    "name": "research_grounding_tool",
                    "args": {"query": "langgraph tracing", "top_k": 3},
                    "id": "grounding-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"query":"langgraph tracing","results":[{"source_id":"S1"}]}',
            name="research_grounding_tool",
            tool_call_id="grounding-1",
        ),
    ]

    records = _collect_tool_call_records(messages)

    assert len(records) == 1
    assert records[0].node == "research_tools"
    assert records[0].input_payload == {"query": "langgraph tracing", "top_k": 3}
    assert '"source_id":"S1"' in records[0].output_preview
    assert records[0].metadata["tool_call_id"] == "grounding-1"


def test_tool_node_records_bound_large_nested_inputs() -> None:
    messages = [
        AIMessage(
            content="dispatch",
            tool_calls=[
                {
                    "name": "source_parser_tool",
                    "args": {"results": [{"content": "x" * 4000, "metadata": {"rank": 1}}]},
                    "id": "parser-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"sources":[]}',
            name="source_parser_tool",
            tool_call_id="parser-1",
        ),
    ]

    record = _collect_tool_call_records(messages)[0]

    assert record.input_payload["results"]["item_count"] == 1
    assert record.input_payload["results"]["preview"][0]["content"].endswith("…")
    assert len(record.input_payload["results"]["preview"][0]["content"]) <= 501


def test_direct_tool_registry_invocation_records_actual_result() -> None:
    @tool("echo_tool")
    def echo_tool(value: str) -> str:
        """Echo a value as JSON."""
        return json.dumps({"echo": value})

    registry = ToolRegistry(
        research_grounding_tool=echo_tool,
        retrieval_tool=echo_tool,
        web_search_tool=echo_tool,
        web_reader_tool=echo_tool,
        source_parser_tool=echo_tool,
        evidence_ranker_tool=echo_tool,
        report_writer_tool=echo_tool,
        review_formatter_tool=echo_tool,
        artifact_export_tool=echo_tool,
        trace_logger_tool=echo_tool,
        local_storage_tool=echo_tool,
    )

    result = registry.invoke("report_writer_tool", {"value": "actual output"})
    records = registry.drain_tool_call_history()

    assert result == {"echo": "actual output"}
    assert len(records) == 1
    assert records[0].node == "analyst_agent"
    assert records[0].input_payload == {"value": "actual output"}
    assert records[0].output_preview == '{"echo": "actual output"}'
    assert records[0].metadata["duration_ms"] >= 0


class _Snapshot:
    def __init__(
        self,
        values: dict,
        *,
        next_nodes: tuple[str, ...] = (),
        interrupts: tuple = (),
        tasks: tuple = (),
    ) -> None:
        self.values = values
        self.next = next_nodes
        self.interrupts = interrupts
        self.tasks = tasks


class _Graph:
    def __init__(self, snapshot: _Snapshot) -> None:
        self.snapshot = snapshot
        self.config = None

    def get_state(self, config: dict) -> _Snapshot:
        self.config = config
        return self.snapshot


class _Compaction:
    @staticmethod
    def merge_archived_trace(state: dict) -> list:
        return list(state.get("execution_trace", []))


def _runner(snapshot: _Snapshot) -> WorkflowRunner:
    runner = WorkflowRunner.__new__(WorkflowRunner)
    runner.runtime = SimpleNamespace(context_compaction=_Compaction())
    runner.graph = _Graph(snapshot)
    return runner


def test_runner_inspect_reads_checkpoint_and_exposes_observability() -> None:
    state = {
        "task_id": "task-live",
        "user_request": "Inspect me",
        "status": "planned",
        "draft_report": "# Draft\nDetails",
        "artifacts": [],
        "execution_trace": [
            TraceEvent(timestamp="2026-01-01T00:00:00Z", node="planner_agent", status="completed", message="done")
        ],
        "tool_call_history": [
            ToolCallRecord(node="research_tools", tool_name="research_grounding_tool", status="success")
        ],
        "model_call_history": [
            ModelCallRecord(
                node="planner_agent",
                provider="test-provider",
                model="trace-model",
                status="success",
            )
        ],
    }
    runner = _runner(_Snapshot(state, next_nodes=("research_pipeline",)))

    inspected, interrupt_payload = runner.inspect("task-live")
    response = runner.summarize(inspected, interrupt_payload)

    assert runner.graph.config == {"configurable": {"thread_id": "task-live"}}
    assert response.task_id == "task-live"
    assert response.next_nodes == ["research_pipeline"]
    assert response.execution_trace[0].node == "planner_agent"
    assert response.tool_call_history[0].tool_name == "research_grounding_tool"
    assert response.model_call_history[0].output_preview == ""
    assert response.draft_report == "# Draft\nDetails"
    assert response.final_report is None


def test_runner_inspect_rejects_unknown_checkpoint() -> None:
    runner = _runner(_Snapshot({}))

    with pytest.raises(RunNotFoundError, match="has not produced a checkpoint"):
        runner.inspect("missing")


def test_runner_inspect_recovers_interrupt_from_snapshot_tasks() -> None:
    interrupt = SimpleNamespace(value={"task_id": "task-hitl", "risk_summary": "Review required"})
    task = SimpleNamespace(interrupts=(interrupt,))
    state = {"task_id": "task-hitl", "user_request": "Review me", "status": "review_escalate"}
    runner = _runner(_Snapshot(state, tasks=(task,)))

    inspected, interrupt_payload = runner.inspect("task-hitl")
    response = runner.summarize(inspected, interrupt_payload)

    assert response.approval_required is True
    assert response.approval_payload == {"task_id": "task-hitl", "risk_summary": "Review required"}


def test_runner_returns_full_markdown_only_for_completed_state() -> None:
    state = {
        "task_id": "task-complete",
        "user_request": "Produce the final report",
        "status": "completed",
        "draft_report": "# Draft\nSuperseded",
    }
    runner = _runner(_Snapshot(state))

    response = runner.summarize(state)

    assert response.draft_report == "# Draft\nSuperseded"
    assert response.final_report is not None
    assert response.final_report.startswith("# LangGraph AgentOps Studio Report")
    assert "## Workflow Outcome\nStatus: `completed`" in response.final_report


def test_runner_surfaces_checkpoint_task_errors_as_trace_events() -> None:
    task = SimpleNamespace(name="analyst_agent", error="provider timeout", interrupts=())
    state = {"task_id": "task-error", "user_request": "Fail visibly", "status": "routing_to_analyst_agent"}
    runner = _runner(_Snapshot(state, tasks=(task,)))

    inspected, interrupt_payload = runner.inspect("task-error")
    response = runner.summarize(inspected, interrupt_payload)

    assert response.execution_trace[-1].node == "analyst_agent"
    assert response.execution_trace[-1].status == "error"
    assert response.execution_trace[-1].metadata["error"] == "provider timeout"
