# Context Compaction

LangGraph AgentOps Studio keeps full audit state while reducing the active context used by downstream nodes. Compaction is an optimization layer: it must not change the workflow contract, routing semantics, HITL behavior, or final audit artifacts.

## Strategy

Context is handled by type instead of applying one generic summary:

- System and task constraints are protected.
- Recent user intent is kept verbatim.
- Raw tool outputs are offloaded only after they are parsed into structured state.
- Retrieved evidence is semantically deduplicated and distilled into structured digests.
- Reviewer history is represented as a revision delta for the next analyst pass.
- Execution trace can be archived outside active state and rehydrated on export.
- Old conversation can be summarized only under pressure.

## Protected Context

`ContextPolicy` protects `user_request`, `task_type`, `plan`, `acceptance_criteria`, `search_queries`, governance evaluation, human review requirement, and approval decision. The compaction service snapshots these fields before compaction and validates them afterward. If validation fails, the update is discarded and the original context continues.

## Tool Output Lifecycle

The research subgraph now runs:

```text
research_briefing
-> research_tools
-> parse_sources
-> parser_tools
-> rank_evidence
-> ranking_tools
-> collect_research
-> compact_research_context
```

`collect_research` materializes durable state fields such as `retrieved_sources`, `ranked_evidence`, `evidence_assessments`, `evidence_conflicts`, `evidence_supports`, and `coverage_record`. Only after that boundary can raw `ToolMessage` payloads from research grounding, parsing, ranking, and tracing be written under `runs/<task_id>/context/raw_tool_outputs/`.

If artifact writing succeeds and the message pair has stable IDs, the active message context can receive `RemoveMessage` records for the consumed AI/tool pair. If writing fails, or the tool output has not been materialized yet, no eviction happens.

## Evidence Digests

The full `ranked_evidence` ledger remains in state and final artifacts. `compacted_evidence` is an active-context view made of `EvidenceDigest` records:

- `claim`
- `summary`
- `source_ids`
- `evidence_ids`
- `citations`
- `confidence`
- `risk_flags`
- `conflict_refs`
- `support_refs`

Semantic deduplication uses the configured embedding provider on each evidence claim and summary. Cosine similarity must meet `CONTEXT_SEMANTIC_DEDUPE_THRESHOLD`. If embeddings fail, compaction falls back to exact normalized text matching.

## Contradiction Guard

Evidence with similar embeddings is not merged when existing `evidence_conflicts` links connect the candidate source with a source already in the cluster. Support and conflict references are copied into the digest so lineage remains explainable.

## Review Delta

Reviewer feedback still appends to `reviewer_history`. The active revision context is `revision_ledger`, which tracks:

- `resolved_issues`
- `open_issues`
- `new_issues`
- `latest_revision_requests`
- `latest_major_risks`
- `iteration`

Issue matching is deterministic and lightweight: normalized exact matching plus token overlap. Reviewer routing, max revision checks, escalation, approval, and HITL behavior are unchanged.

## Trace Offloading

When active trace length exceeds `CONTEXT_TRACE_TAIL_SIZE`, older events are written to:

```text
runs/<task_id>/context/execution_trace_archive.json
```

Active state keeps the recent tail plus a `ContextArtifactPointer`. The executor rehydrates archived trace events and merges them with the active tail before exporting `workflow_trace.json`, preserving chronological order and avoiding duplicates.

## Old Conversation Summary

Old conversation summary is triggered only under high or critical token pressure and only when message count exceeds the configured recent-message window. Recent HumanMessage content is preserved verbatim. The deterministic summary captures task goal, hard constraints, decisions, unresolved issues, and relevant errors.

## Phase Boundary and Token Pressure

Compaction triggers come from both workflow semantics and estimated token pressure:

- Research completion triggers evidence digest creation and tool-output lifecycle handling.
- Reviewer feedback triggers revision ledger updates.
- Trace length triggers archive/offload.
- High or critical pressure can trigger old conversation summary.

`ContextBudgeter` uses a deterministic estimate rather than a provider tokenizer. CJK characters, Latin words, and remaining characters are counted separately to produce a conservative approximation. Pressure levels are `low`, `medium`, `high`, and `critical`.

## Role-Aware Projection

`ContextCompactionService` exposes role projections:

- Planner: `user_request`, `task_type`
- Research: `user_request`, `task_type`, `search_queries`, `acceptance_criteria`, `plan`
- Analyst: compacted evidence when available, otherwise full ranked evidence
- Reviewer: compacted evidence when available, otherwise full ranked evidence

The original state remains the source of truth.

## Rehydration

`ContextArtifactPointer` records artifact type, relative path, creation time, item count, and metadata. `LocalArtifactStore.read_json` and `read_text` only resolve paths inside the current run directory, preventing path traversal.

Supported rehydration paths include raw tool outputs and execution trace archives.

## Metrics

Every compaction event records structured stats in `context_compaction_history` and `execution_trace`:

- trigger reason
- phase
- pressure level
- estimated tokens before and after
- tool messages offloaded
- evidence count before and after
- conflicts preserved
- review open issue count
- artifact paths
- fallback reason

The executor exports `context_compaction.json` and `context_manifest.json` alongside the existing artifacts.

## Failure Behavior

Compaction is fail-open. Embedding failure, artifact write failure, summary failure, rehydration failure, protected-context validation failure, or token-estimation errors must not stop the workflow. The original context remains active and the fallback reason is recorded.

## Configuration

Configuration is available in `config/defaults.yaml` and can be overridden through `.env`:

```env
CONTEXT_COMPACTION_ENABLED=true
CONTEXT_MAX_ESTIMATED_TOKENS=24000
CONTEXT_LIGHT_PRESSURE_RATIO=0.60
CONTEXT_HIGH_PRESSURE_RATIO=0.75
CONTEXT_CRITICAL_PRESSURE_RATIO=0.85
CONTEXT_RECENT_USER_MESSAGES=2
CONTEXT_RECENT_MESSAGES=8
CONTEXT_TRACE_TAIL_SIZE=25
CONTEXT_SEMANTIC_DEDUPE_THRESHOLD=0.90
CONTEXT_TOOL_OFFLOAD_ENABLED=true
```

Disabling `CONTEXT_COMPACTION_ENABLED` keeps behavior closest to the uncompressed workflow while leaving the original artifact and routing paths intact.
