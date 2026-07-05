from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from schemas.models import (
    CitationRecord,
    ConflictRecord,
    ContextArtifactPointer,
    EvidenceRecord,
    PlanStep,
    ReviewFeedback,
    TraceEvent,
)
from services.config import ContextCompactionSettings
from services.context_compaction import ContextBudgeter, ContextCompactionService
from services.storage import LocalArtifactStore


class FakeEmbeddings:
    name = "fake"
    dimensions = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
                vectors.append([0.99, 0.01])
            elif "gamma" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FailingStore(LocalArtifactStore):
    def write_json(self, task_id: str, name: str, payload):
        raise OSError("write failed")


def _service(tmp_path: Path, settings: ContextCompactionSettings | None = None) -> ContextCompactionService:
    return ContextCompactionService(
        settings=settings or ContextCompactionSettings(max_estimated_tokens=1000),
        storage=LocalArtifactStore(tmp_path),
        embeddings=FakeEmbeddings(),
    )


def _evidence(evidence_id: str, source_id: str, marker: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        claim=f"{marker} claim",
        supporting_sources=[source_id],
        confidence=0.8,
        risk_flags=[f"risk-{source_id}"],
        summary=f"{marker} summary",
        citations=[
            CitationRecord(
                source_id=source_id,
                provider="memory",
                title=f"Source {source_id}",
                source=f"https://example.com/{source_id}",
            )
        ],
    )


def test_protected_context_survives_compaction(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    plan = [PlanStep(step_id="P1", objective="Assess", owner="planner", done_definition="Done")]
    state = {
        "task_id": "task",
        "user_request": "Keep this request exact",
        "task_type": "architecture",
        "plan": plan,
        "acceptance_criteria": ["must preserve criteria"],
        "search_queries": ["active query"],
        "messages": [],
        "retrieved_sources": [{"source_id": "SRC-1"}],
        "ranked_evidence": [_evidence("EVD-1", "SRC-1", "alpha")],
        "evidence_assessments": [],
        "evidence_conflicts": [],
        "evidence_supports": [],
        "coverage_record": {"query_coverage": 1.0},
        "execution_trace": [],
        "context_artifacts": [],
    }

    update = svc.compact_research_context(state)
    merged = {**state, **{key: value for key, value in update.items() if key != "messages"}}

    assert merged["user_request"] == state["user_request"]
    assert merged["task_type"] == state["task_type"]
    assert merged["plan"] == state["plan"]
    assert merged["acceptance_criteria"] == state["acceptance_criteria"]


def test_recent_user_intent_is_kept_verbatim_when_old_conversation_is_summarized(tmp_path: Path) -> None:
    svc = _service(
        tmp_path,
        ContextCompactionSettings(max_estimated_tokens=1000, recent_user_messages=2, recent_messages=2),
    )
    messages = [
        HumanMessage(content="old intent " + "x" * 1200),
        AIMessage(content="old answer " + "y" * 1200),
        HumanMessage(content="recent intent one"),
        HumanMessage(content="recent intent two"),
    ]
    state = {
        "task_id": "task",
        "user_request": "Summarize old chat",
        "acceptance_criteria": ["keep hard constraint"],
        "messages": messages,
    }

    summary, pointer = svc.summarize_old_conversation_if_needed(state, "high")

    assert summary
    assert pointer is not None
    assert [item.content for item in svc._recent_human_messages(messages)] == [
        "recent intent one",
        "recent intent two",
    ]
    assert messages[-2].content == "recent intent one"
    assert messages[-1].content == "recent intent two"


def test_tool_lifecycle_offloads_only_materialized_outputs_and_fails_open(tmp_path: Path) -> None:
    ai = AIMessage(
        content="call",
        tool_calls=[{"name": "research_grounding_tool", "args": {}, "id": "grounding-1"}],
        id="ai-1",
    )
    tool = ToolMessage(
        content=json.dumps({"results": [{"source_id": "SRC-1"}]}),
        name="research_grounding_tool",
        tool_call_id="grounding-1",
        id="tool-1",
    )
    unmaterialized = {"task_id": "task", "messages": [ai, tool], "ranked_evidence": []}
    svc = _service(tmp_path)

    update, pointers, removed = svc.offload_consumed_tool_outputs(unmaterialized)

    assert update == {}
    assert pointers == []
    assert removed == 0

    materialized = {
        **unmaterialized,
        "retrieved_sources": [{"source_id": "SRC-1"}],
        "ranked_evidence": [_evidence("EVD-1", "SRC-1", "alpha")],
        "evidence_assessments": [],
        "coverage_record": {"query_coverage": 1.0},
    }
    update, pointers, removed = svc.offload_consumed_tool_outputs(materialized)

    assert removed == 1
    assert pointers[0].artifact_type == "raw_tool_output"
    assert update["messages"]
    assert (tmp_path / "task" / pointers[0].path).exists()

    failing = ContextCompactionService(
        settings=ContextCompactionSettings(max_estimated_tokens=1000),
        storage=FailingStore(tmp_path / "failing"),
        embeddings=FakeEmbeddings(),
    )
    update, pointers, removed = failing.offload_consumed_tool_outputs(materialized)

    assert update == {}
    assert pointers == []
    assert removed == 0


def test_semantic_dedupe_preserves_conflicting_evidence_lineage(tmp_path: Path) -> None:
    svc = _service(tmp_path, ContextCompactionSettings(max_estimated_tokens=1000, semantic_dedupe_threshold=0.9))
    evidence = [
        _evidence("EVD-1", "SRC-1", "alpha"),
        _evidence("EVD-2", "SRC-2", "beta"),
        _evidence("EVD-3", "SRC-3", "gamma"),
    ]
    conflicts = [ConflictRecord(left_source_id="SRC-1", right_source_id="SRC-3", severity=0.9, reason="opposes")]

    digests, compaction_map = svc.compact_evidence(evidence, conflicts, [])

    assert len(digests) == 2
    merged = next(item for item in digests if set(item.evidence_ids) == {"EVD-1", "EVD-2"})
    separate = next(item for item in digests if item.evidence_ids == ["EVD-3"])
    assert set(merged.source_ids) == {"SRC-1", "SRC-2"}
    assert {citation.source_id for citation in merged.citations} == {"SRC-1", "SRC-2"}
    assert separate.conflict_refs == ["SRC-1->SRC-3"]
    assert compaction_map[0].evidence_ids


def test_review_delta_tracks_resolved_open_and_new_issues(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    first = ReviewFeedback(
        verdict="revise",
        summary="first review",
        revision_requests=["Add cost comparison", "Clarify risk controls"],
        major_risks=["Compliance gap"],
    )
    ledger, _, _ = svc.update_revision_ledger({}, first)
    second = ReviewFeedback(
        verdict="revise",
        summary="second review",
        revision_requests=["Clarify risk controls", "Add rollout owner"],
        major_risks=[],
    )

    ledger, _, _ = svc.update_revision_ledger({"revision_ledger": ledger}, second)

    assert "Add cost comparison" in ledger.resolved_issues
    assert "Clarify risk controls" in ledger.open_issues
    assert "Add rollout owner" in ledger.new_issues
    assert ledger.iteration == 2


def test_pressure_levels_are_configurable() -> None:
    budgeter = ContextBudgeter(
        ContextCompactionSettings(
            max_estimated_tokens=1000,
            light_pressure_ratio=0.5,
            high_pressure_ratio=0.7,
            critical_pressure_ratio=0.9,
        )
    )

    assert budgeter.pressure_level(400) == "low"
    assert budgeter.pressure_level(600) == "medium"
    assert budgeter.pressure_level(800) == "high"
    assert budgeter.pressure_level(950) == "critical"


def test_rehydration_reads_known_context_artifacts_and_rejects_path_escape(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    raw_pointer = ContextArtifactPointer(
        artifact_type="raw_tool_output",
        path="context/raw_tool_outputs/tool.json",
    )
    trace_pointer = ContextArtifactPointer(
        artifact_type="execution_trace_archive",
        path="context/execution_trace_archive.json",
    )
    svc.storage.write_json("task", raw_pointer.path, {"tool_name": "research_grounding_tool"})
    svc.storage.write_json(
        "task",
        trace_pointer.path,
        {"trace": [TraceEvent(timestamp="2026-01-01T00:00:00Z", node="n", status="s", message="m").model_dump()]},
    )

    assert svc.rehydrate_context_artifact("task", raw_pointer)["tool_name"] == "research_grounding_tool"
    assert svc.rehydrate_context_artifact("task", trace_pointer)["trace"][0]["node"] == "n"

    with pytest.raises(ValueError):
        svc.rehydrate_context_artifact(
            "task",
            ContextArtifactPointer(artifact_type="raw_tool_output", path="../outside.json"),
        )
