# LangGraph AgentOps Studio Architecture

## Graph Nodes

The top-level graph contains these nodes:

- `initialize_task`
- `planner_agent`
- `supervisor`
- `research_pipeline` (subgraph)
- `analyst_agent`
- `reviewer_agent`
- `human_review`
- `executor_agent`

## Research Subgraph

`research_pipeline` runs:

1. research query dispatch
2. `research_grounding_tool`
3. `source_parser_tool`
4. `evidence_ranker_tool`
5. state collection and trace updates

## Service Layer

- `services/retrieval.py`: vector ingestion and retrieval.
- `services/web_search.py`: Tavily and Exa search adapters.
- `services/web_reader.py`: Jina and Exa content adapters.
- `services/web_grounding.py`: merge, dedupe, and rank across vector and web channels.
- `services/evidence.py`: evidence assessment, conflict/support detection, and coverage scoring.
- `services/recommendation.py`: evidence-driven recommendation synthesis.
- `services/governance.py`: policy evaluation for human review gating.

## State Contract

`schemas/state.py` carries structured fields across nodes, including:

- plan and acceptance criteria
- retrieved sources and ranked evidence
- evidence assessments, conflicts, supports, and coverage
- findings and recommendation
- governance evaluation and human approval decisions
- tool/model call histories, execution trace, and exported artifacts

## Checkpoint-backed progress inspection

`GET /runs/{task_id}` reads the latest `StateSnapshot` from the compiled
LangGraph checkpointer. It exposes the same `RunResponse` contract as create and
continue calls, including node traces, actual tool calls, actual provider model
calls and outputs, current/final reports, and `next_nodes`. See
[`docs/observability.md`](observability.md) for the polling and field contract.

## Decision Path

- The analyst creates findings and recommendation from ranked evidence.
- Governance policy evaluation computes trigger conditions and risk summary.
- Reviewer consumes governance output and recommendation confidence.
- Human approval is requested when governance or reviewer outcomes require manual signoff.
- Executor exports the final artifact set for audit and downstream consumption.
