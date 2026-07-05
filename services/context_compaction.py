from __future__ import annotations

# Context compaction is an active-context optimization, not a replacement for
# the audit ledger. Raw state stays recoverable through artifacts and lineage.

import json
import math
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from schemas.models import (
    CitationRecord,
    ConflictRecord,
    ContextArtifactPointer,
    ContextCompactionEvent,
    ContextCompactionStats,
    EvidenceCompactionMap,
    EvidenceDigest,
    EvidenceRecord,
    RevisionLedger,
    ReviewFeedback,
    SupportRecord,
    TraceEvent,
)
from schemas.state import replace_records
from services.config import ContextCompactionSettings
from services.embeddings import BaseEmbeddingProvider
from services.llm import BaseLLMProvider
from services.serialization import dumps, to_jsonable
from services.storage import LocalArtifactStore


PROTECTED_FIELDS = [
    "user_request",
    "task_type",
    "plan",
    "acceptance_criteria",
    "search_queries",
    "governance_evaluation",
    "human_approval_required",
    "approval_decision",
]

RAW_TOOL_NAMES = {
    "research_grounding_tool",
    "source_parser_tool",
    "evidence_ranker_tool",
    "trace_logger_tool",
}


class ContextBudgeter:
    def __init__(self, settings: ContextCompactionSettings) -> None:
        self.settings = settings

    def estimate_tokens(self, payload: Any) -> int:
        # This is a deterministic pressure estimate, not a provider tokenizer.
        # It treats CJK characters conservatively and groups Latin text by word.
        try:
            text = dumps(to_jsonable(payload), indent=0)
        except Exception:
            text = str(payload)
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
        other = max(0, len(text) - cjk)
        return max(1, int(cjk * 1.2 + latin_words * 1.3 + other / 4))

    def pressure_level(self, estimated_tokens: int) -> str:
        ratio = estimated_tokens / max(1, self.settings.max_estimated_tokens)
        if ratio >= self.settings.critical_pressure_ratio:
            return "critical"
        if ratio >= self.settings.high_pressure_ratio:
            return "high"
        if ratio >= self.settings.light_pressure_ratio:
            return "medium"
        return "low"


class ContextPolicy:
    def protected_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return {field: to_jsonable(state.get(field)) for field in PROTECTED_FIELDS}

    def validate(self, before: dict[str, Any], after_state: dict[str, Any]) -> bool:
        return before == self.protected_snapshot(after_state)


class ContextCompactionService:
    def __init__(
        self,
        *,
        settings: ContextCompactionSettings,
        storage: LocalArtifactStore,
        embeddings: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider | None = None,
        logger: Any | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.embeddings = embeddings
        self.llm_provider = llm_provider
        self.logger = logger
        self.budgeter = ContextBudgeter(settings)
        self.policy = ContextPolicy()

    def compact_research_context(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.enabled:
            return {}

        protected_before = self.policy.protected_snapshot(state)
        before_tokens = self.budgeter.estimate_tokens(self._pressure_payload(state))
        pressure = self.budgeter.pressure_level(before_tokens)
        new_pointers: list[ContextArtifactPointer] = []
        artifact_paths: list[str] = []
        update: dict[str, Any] = {}
        fallback_reason = None
        tool_messages_offloaded = 0

        try:
            compacted, compaction_map = self.compact_evidence(
                state.get("ranked_evidence", []),
                state.get("evidence_conflicts", []),
                state.get("evidence_supports", []),
            )
            update["compacted_evidence"] = compacted
            update["evidence_compaction_map"] = compaction_map

            if self.settings.tool_offload_enabled:
                tool_update, pointers, removed_count = self.offload_consumed_tool_outputs(state)
                update.update(tool_update)
                new_pointers.extend(pointers)
                tool_messages_offloaded = removed_count

            summary, summary_pointer = self.summarize_old_conversation_if_needed(state, pressure)
            if summary:
                update["conversation_summary"] = summary
            if summary_pointer:
                new_pointers.append(summary_pointer)

            trace_update, trace_pointer = self.archive_trace_if_needed(state)
            if trace_pointer:
                new_pointers.append(trace_pointer)

            if new_pointers:
                existing = [
                    item if isinstance(item, ContextArtifactPointer) else ContextArtifactPointer.model_validate(item)
                    for item in state.get("context_artifacts", [])
                ]
                update["context_artifacts"] = _dedupe_pointers(existing + new_pointers)
                artifact_paths = [item.path for item in new_pointers]

            after_state = {**state, **{key: value for key, value in update.items() if key != "messages"}}
            if not self.policy.validate(protected_before, after_state):
                fallback_reason = "protected_context_validation_failed"
                update = {}
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            self._log_exception("Context compaction failed; continuing with raw context.", exc)
            update = {}

        after_payload = {**self._pressure_payload(state), **to_jsonable(update)}
        after_tokens = self.budgeter.estimate_tokens(after_payload) if not fallback_reason else before_tokens
        stats = ContextCompactionStats(
            trigger_reason="research_phase_boundary",
            phase="research",
            pressure_level=pressure,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=after_tokens,
            tool_messages_offloaded=tool_messages_offloaded if not fallback_reason else 0,
            evidence_count_before=len(state.get("ranked_evidence", [])),
            evidence_count_after=len(update.get("compacted_evidence", [])) if not fallback_reason else 0,
            conflicts_preserved=len(state.get("evidence_conflicts", [])),
            review_open_issue_count=len((state.get("revision_ledger") or {}).get("open_issues", []))
            if isinstance(state.get("revision_ledger"), dict)
            else len(state.get("revision_ledger").open_issues)
            if state.get("revision_ledger")
            else 0,
            artifact_paths=artifact_paths if not fallback_reason else [],
            fallback_reason=fallback_reason,
        )
        event = ContextCompactionEvent(
            trigger_reason=stats.trigger_reason,
            phase=stats.phase,
            stats=stats,
            protected_fields=PROTECTED_FIELDS,
        )
        trace = self._trace_from_event(event)
        update["context_compaction_history"] = [event]
        if "execution_trace" in update and isinstance(update["execution_trace"], dict):
            update["execution_trace"]["items"].append(trace)
        else:
            update["execution_trace"] = [trace]
        return update

    def compact_evidence(
        self,
        ranked_evidence: list[EvidenceRecord | dict[str, Any]],
        conflicts: list[ConflictRecord | dict[str, Any]],
        supports: list[SupportRecord | dict[str, Any]],
    ) -> tuple[list[EvidenceDigest], list[EvidenceCompactionMap]]:
        evidence = [
            item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
            for item in ranked_evidence
        ]
        conflict_models = [
            item if isinstance(item, ConflictRecord) else ConflictRecord.model_validate(item)
            for item in conflicts
        ]
        support_models = [
            item if isinstance(item, SupportRecord) else SupportRecord.model_validate(item)
            for item in supports
        ]
        if not evidence:
            return [], []

        vectors = self._embed_evidence(evidence)
        clusters: list[list[int]] = []
        for index, record in enumerate(evidence):
            target_cluster = None
            for cluster in clusters:
                representative = evidence[cluster[0]]
                similar = self._is_similar(index, cluster[0], record, representative, vectors)
                if similar and not self._has_conflict(record, [evidence[item] for item in cluster], conflict_models):
                    target_cluster = cluster
                    break
            if target_cluster is None:
                clusters.append([index])
            else:
                target_cluster.append(index)

        digests = [
            self._build_digest(f"DIG-{index:02d}", [evidence[item] for item in cluster], conflict_models, support_models)
            for index, cluster in enumerate(clusters, start=1)
        ]
        maps = [
            EvidenceCompactionMap(
                digest_id=digest.digest_id,
                evidence_ids=digest.evidence_ids,
                source_ids=digest.source_ids,
            )
            for digest in digests
        ]
        return digests, maps

    def offload_consumed_tool_outputs(
        self,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[ContextArtifactPointer], int]:
        if not self._research_outputs_materialized(state):
            return {}, [], 0

        messages = list(state.get("messages", []))
        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage) and getattr(message, "name", None) in RAW_TOOL_NAMES
        ]
        if not tool_messages:
            return {}, [], 0

        pointers: list[ContextArtifactPointer] = []
        offloaded_call_ids: set[str] = set()
        for index, message in enumerate(tool_messages, start=1):
            call_id = str(getattr(message, "tool_call_id", "") or f"tool-{index}")
            name = str(getattr(message, "name", "tool"))
            file_name = f"context/raw_tool_outputs/{_safe_slug(name)}_{_safe_slug(call_id)}.json"
            payload = {
                "tool_name": name,
                "tool_call_id": call_id,
                "message_id": getattr(message, "id", None),
                "content": _decode_content(message.content),
                "created_at": datetime.now(UTC).isoformat(),
            }
            try:
                self.storage.write_json(state["task_id"], file_name, payload)
            except Exception as exc:
                self._log_exception("Raw tool output offload failed; retaining message.", exc)
                continue
            pointers.append(
                ContextArtifactPointer(
                    artifact_type="raw_tool_output",
                    path=file_name,
                    description=f"Raw output from {name}.",
                    item_count=1,
                    metadata={"tool_name": name, "tool_call_id": call_id},
                )
            )
            offloaded_call_ids.add(call_id)

        removals = self._safe_tool_message_removals(messages, offloaded_call_ids)
        update = {"messages": removals} if removals else {}
        return update, pointers, len(pointers)

    def summarize_old_conversation_if_needed(
        self,
        state: dict[str, Any],
        pressure: str,
    ) -> tuple[str | None, ContextArtifactPointer | None]:
        messages = list(state.get("messages", []))
        if pressure not in {"high", "critical"} or len(messages) <= self.settings.recent_messages:
            return None, None

        recent_user_ids = {
            id(message)
            for message in self._recent_human_messages(messages)
        }
        old_messages = messages[: -self.settings.recent_messages]
        summary_lines = [
            "Task goal: " + str(state.get("user_request", "")),
            "Hard constraints: " + "; ".join(state.get("acceptance_criteria", []) or ["none"]),
            "Decisions: " + _message_excerpt(old_messages, {"ai"}),
            "Unresolved issues: "
            + (
                "; ".join(_coerce_ledger(state.get("revision_ledger")).open_issues)
                if state.get("revision_ledger")
                else "none"
            ),
            "Relevant errors: "
            + (
                str(state.get("error_info").message)
                if state.get("error_info") and hasattr(state.get("error_info"), "message")
                else "none"
            ),
        ]
        for message in old_messages:
            if isinstance(message, HumanMessage) and id(message) in recent_user_ids:
                continue
        summary = "\n".join(summary_lines)
        file_name = "context/conversation_summary.json"
        try:
            self.storage.write_json(
                state["task_id"],
                file_name,
                {"summary": summary, "created_at": datetime.now(UTC).isoformat()},
            )
        except Exception as exc:
            self._log_exception("Conversation summary write failed; keeping raw messages.", exc)
            return None, None
        return summary, ContextArtifactPointer(
            artifact_type="conversation_summary",
            path=file_name,
            description="Deterministic summary of old non-recent conversation context.",
            item_count=len(old_messages),
        )

    def archive_trace_if_needed(self, state: dict[str, Any]) -> tuple[dict[str, Any], ContextArtifactPointer | None]:
        trace = [
            item if isinstance(item, TraceEvent) else TraceEvent.model_validate(item)
            for item in state.get("execution_trace", [])
        ]
        tail_size = self.settings.trace_tail_size
        if len(trace) <= tail_size:
            return {}, None

        archive_name = "context/execution_trace_archive.json"
        archived = trace[:-tail_size]
        try:
            try:
                existing_payload = self.storage.read_json(state["task_id"], archive_name)
                existing = [TraceEvent.model_validate(item) for item in existing_payload.get("trace", [])]
            except FileNotFoundError:
                existing = []
            merged = _dedupe_trace(existing + archived)
            self.storage.write_json(
                state["task_id"],
                archive_name,
                {"trace": [item.model_dump() for item in merged], "updated_at": datetime.now(UTC).isoformat()},
            )
        except Exception as exc:
            self._log_exception("Trace archive failed; retaining full active trace.", exc)
            return {}, None

        pointer = ContextArtifactPointer(
            artifact_type="execution_trace_archive",
            path=archive_name,
            description="Archived execution trace events outside active state.",
            item_count=len(merged),
        )
        return {"execution_trace": replace_records(trace[-tail_size:])}, pointer

    def update_revision_ledger(
        self,
        state: dict[str, Any],
        feedback: ReviewFeedback,
    ) -> tuple[RevisionLedger, ContextCompactionEvent, TraceEvent]:
        previous = _coerce_ledger(state.get("revision_ledger"))
        current_issues = _unique(feedback.revision_requests + feedback.questions + feedback.major_risks)
        resolved = [issue for issue in previous.open_issues if not _has_similar(issue, current_issues)]
        still_open = [issue for issue in previous.open_issues if _has_similar(issue, current_issues)]
        new = [issue for issue in current_issues if not _has_similar(issue, previous.open_issues)]
        ledger = RevisionLedger(
            resolved_issues=_unique(previous.resolved_issues + resolved),
            open_issues=_unique(still_open + current_issues),
            new_issues=new,
            latest_revision_requests=list(feedback.revision_requests),
            latest_major_risks=list(feedback.major_risks),
            iteration=previous.iteration + 1,
        )
        stats = ContextCompactionStats(
            trigger_reason="reviewer_feedback",
            phase="reviewer",
            pressure_level="low",
            estimated_tokens_before=self.budgeter.estimate_tokens(previous),
            estimated_tokens_after=self.budgeter.estimate_tokens(ledger),
            review_open_issue_count=len(ledger.open_issues),
        )
        event = ContextCompactionEvent(
            trigger_reason=stats.trigger_reason,
            phase=stats.phase,
            stats=stats,
            protected_fields=PROTECTED_FIELDS,
        )
        return ledger, event, self._trace_from_event(event)

    @staticmethod
    def project_planner_context(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_request": state.get("user_request", ""),
            "task_type": state.get("task_type", "general"),
        }

    @staticmethod
    def project_research_context(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_request": state.get("user_request", ""),
            "task_type": state.get("task_type", "general"),
            "search_queries": list(state.get("search_queries", [])),
            "acceptance_criteria": list(state.get("acceptance_criteria", [])),
            "plan": state.get("plan", []),
        }

    def project_analyst_evidence(self, state: dict[str, Any]) -> list[EvidenceRecord]:
        digests = [
            item if isinstance(item, EvidenceDigest) else EvidenceDigest.model_validate(item)
            for item in state.get("compacted_evidence", [])
        ]
        if not digests:
            return [
                item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
                for item in state.get("ranked_evidence", [])
            ]
        return [_digest_to_evidence(item) for item in digests]

    def project_reviewer_evidence(self, state: dict[str, Any]) -> list[EvidenceRecord]:
        return self.project_analyst_evidence(state)

    def merge_archived_trace(self, state: dict[str, Any]) -> list[TraceEvent]:
        active = [
            item if isinstance(item, TraceEvent) else TraceEvent.model_validate(item)
            for item in state.get("execution_trace", [])
        ]
        archived: list[TraceEvent] = []
        for pointer in state.get("context_artifacts", []):
            pointer_model = pointer if isinstance(pointer, ContextArtifactPointer) else ContextArtifactPointer.model_validate(pointer)
            if pointer_model.artifact_type != "execution_trace_archive":
                continue
            payload = self.rehydrate_context_artifact(state["task_id"], pointer_model)
            archived.extend(TraceEvent.model_validate(item) for item in payload.get("trace", []))
        return _dedupe_trace(archived + active)

    def rehydrate_context_artifact(self, task_id: str, pointer: ContextArtifactPointer | dict[str, Any]) -> Any:
        pointer_model = pointer if isinstance(pointer, ContextArtifactPointer) else ContextArtifactPointer.model_validate(pointer)
        if pointer_model.artifact_type in {"raw_tool_output", "execution_trace_archive", "conversation_summary", "context_manifest"}:
            return self.storage.read_json(task_id, pointer_model.path)
        raise ValueError(f"Unsupported context artifact type: {pointer_model.artifact_type}")

    def _embed_evidence(self, evidence: list[EvidenceRecord]) -> list[list[float]] | None:
        texts = [f"{item.claim}\n{item.summary}" for item in evidence]
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as exc:
            self._log_exception("Evidence embedding failed; semantic dedupe will use exact text only.", exc)
            return None
        return vectors if len(vectors) == len(evidence) else None

    def _is_similar(
        self,
        left_index: int,
        right_index: int,
        left: EvidenceRecord,
        right: EvidenceRecord,
        vectors: list[list[float]] | None,
    ) -> bool:
        if vectors:
            return _cosine(vectors[left_index], vectors[right_index]) >= self.settings.semantic_dedupe_threshold
        return _normalize_text(left.claim + " " + left.summary) == _normalize_text(right.claim + " " + right.summary)

    @staticmethod
    def _has_conflict(
        candidate: EvidenceRecord,
        cluster: list[EvidenceRecord],
        conflicts: list[ConflictRecord],
    ) -> bool:
        candidate_sources = set(candidate.supporting_sources)
        cluster_sources = {source for item in cluster for source in item.supporting_sources}
        for conflict in conflicts:
            pair = {conflict.left_source_id, conflict.right_source_id}
            if pair & candidate_sources and pair & cluster_sources and not pair <= candidate_sources and not pair <= cluster_sources:
                return True
        return False

    @staticmethod
    def _build_digest(
        digest_id: str,
        records: list[EvidenceRecord],
        conflicts: list[ConflictRecord],
        supports: list[SupportRecord],
    ) -> EvidenceDigest:
        primary = max(records, key=lambda item: item.confidence)
        source_ids = _unique([source for record in records for source in record.supporting_sources])
        evidence_ids = _unique([record.evidence_id for record in records])
        citations = _dedupe_citations([citation for record in records for citation in record.citations])
        risk_flags = _unique([flag for record in records for flag in record.risk_flags])
        conflict_refs = [
            f"{item.left_source_id}->{item.right_source_id}"
            for item in conflicts
            if item.left_source_id in source_ids or item.right_source_id in source_ids
        ]
        support_refs = [
            f"{item.source_id}->{item.supports_source_id}"
            for item in supports
            if item.source_id in source_ids or item.supports_source_id in source_ids
        ]
        return EvidenceDigest(
            digest_id=digest_id,
            claim=primary.claim,
            summary=primary.summary,
            source_ids=source_ids,
            evidence_ids=evidence_ids,
            citations=citations,
            confidence=min(record.confidence for record in records),
            risk_flags=risk_flags,
            conflict_refs=_unique(conflict_refs),
            support_refs=_unique(support_refs),
        )

    @staticmethod
    def _research_outputs_materialized(state: dict[str, Any]) -> bool:
        return bool(
            state.get("retrieved_sources")
            and state.get("ranked_evidence")
            and state.get("evidence_assessments") is not None
            and state.get("coverage_record") is not None
        )

    @staticmethod
    def _safe_tool_message_removals(messages: list[BaseMessage], offloaded_call_ids: set[str]) -> list[RemoveMessage]:
        if not offloaded_call_ids:
            return []
        tool_by_call_id = {
            str(getattr(message, "tool_call_id", "")): message
            for message in messages
            if isinstance(message, ToolMessage)
        }
        removal_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            tool_calls = getattr(message, "tool_calls", None) or []
            call_ids = {str(call.get("id", "")) for call in tool_calls if call.get("id")}
            if not call_ids or not call_ids <= offloaded_call_ids:
                continue
            related_tools = [tool_by_call_id.get(call_id) for call_id in call_ids]
            if getattr(message, "id", None) and all(item is not None and getattr(item, "id", None) for item in related_tools):
                removal_ids.add(str(message.id))
                removal_ids.update(str(item.id) for item in related_tools if item is not None)
        return [RemoveMessage(id=message_id) for message_id in sorted(removal_ids)]

    def _recent_human_messages(self, messages: list[BaseMessage]) -> list[HumanMessage]:
        humans = [message for message in messages if isinstance(message, HumanMessage)]
        return humans[-self.settings.recent_user_messages :]

    @staticmethod
    def _pressure_payload(state: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "user_request",
            "task_type",
            "messages",
            "plan",
            "acceptance_criteria",
            "search_queries",
            "ranked_evidence",
            "compacted_evidence",
            "reviewer_history",
            "revision_ledger",
            "execution_trace",
            "context_artifacts",
        ]
        return {key: state.get(key) for key in keys}

    @staticmethod
    def _trace_from_event(event: ContextCompactionEvent) -> TraceEvent:
        stats = event.stats
        return TraceEvent(
            timestamp=event.created_at,
            node="context_compaction",
            status="fallback" if stats.fallback_reason else "completed",
            message=f"Context compaction processed {event.phase} context.",
            metadata=stats.model_dump(),
        )

    def _log_exception(self, message: str, exc: Exception) -> None:
        if self.logger:
            self.logger.warning("%s %s: %s", message, type(exc).__name__, exc)


def _digest_to_evidence(digest: EvidenceDigest) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=digest.digest_id,
        claim=digest.claim,
        supporting_sources=digest.source_ids,
        confidence=digest.confidence,
        risk_flags=digest.risk_flags,
        summary=f"{digest.summary}\nLineage evidence IDs: {', '.join(digest.evidence_ids)}",
        citations=digest.citations,
    )


def _decode_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80] or "item"


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", value.lower()))


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", value.lower()))


def _has_similar(issue: str, candidates: list[str]) -> bool:
    issue_norm = _normalize_text(issue)
    issue_tokens = _tokens(issue)
    for candidate in candidates:
        if issue_norm == _normalize_text(candidate):
            return True
        candidate_tokens = _tokens(candidate)
        if issue_tokens and candidate_tokens:
            overlap = len(issue_tokens & candidate_tokens) / max(1, len(issue_tokens | candidate_tokens))
            if overlap >= 0.72:
                return True
    return False


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _coerce_ledger(value: RevisionLedger | dict[str, Any] | None) -> RevisionLedger:
    if value is None:
        return RevisionLedger()
    return value if isinstance(value, RevisionLedger) else RevisionLedger.model_validate(value)


def _dedupe_citations(citations: list[CitationRecord]) -> list[CitationRecord]:
    seen: set[tuple[str, str | None]] = set()
    result: list[CitationRecord] = []
    for citation in citations:
        key = (citation.source_id, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def _dedupe_pointers(pointers: list[ContextArtifactPointer]) -> list[ContextArtifactPointer]:
    deduped: dict[tuple[str, str], ContextArtifactPointer] = {}
    for pointer in pointers:
        deduped[(pointer.artifact_type, pointer.path)] = pointer
    return list(deduped.values())


def _dedupe_trace(trace: list[TraceEvent]) -> list[TraceEvent]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[TraceEvent] = []
    for event in trace:
        key = (event.timestamp, event.node, event.status, event.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return sorted(result, key=lambda item: item.timestamp)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _message_excerpt(messages: list[BaseMessage], types: set[str]) -> str:
    excerpts = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if message.type in types:
            text = str(message.content).replace("\n", " ").strip()
            if text:
                excerpts.append(text[:180])
    return "; ".join(excerpts[:5]) or "none"
