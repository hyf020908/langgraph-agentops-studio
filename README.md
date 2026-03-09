# LangGraph AgentOps Studio

LangGraph AgentOps Studio is a LangGraph-native multi-agent workflow project for research, evidence evaluation, recommendation synthesis, governance checks, and artifact export.

## Core Capabilities

- Graph orchestration with `StateGraph`, `ToolNode`, `Command`, `interrupt`, and checkpointing.
- Multi-agent flow: planner, research pipeline, analyst, reviewer, human approval gate, executor, and supervisor.
- Provider-backed LLM and embeddings (`openai`, `deepseek`, `openai_compatible`).
- Hybrid grounding that combines vector retrieval and web grounding (Tavily + Jina, or Exa search + content).
- Evidence pipeline with structured scoring dimensions, conflict/support detection, and coverage tracking.
- Evidence-driven recommendation synthesis with confidence, unresolved questions, and residual risk outputs.
- Policy-driven governance evaluation with explicit trigger conditions for human review.
- Structured artifacts (`final_report.md`, `decision_record.json`, `workflow_trace.json`, `run_artifact.json`, and supporting files).

## Workflow Shape

```mermaid
flowchart TD
    START([START]) --> init[initialize_task]
    init --> planner[planner_agent]
    planner --> supervisor[supervisor]
    supervisor --> research[research_pipeline]
    research --> supervisor
    supervisor --> analyst[analyst_agent]
    supervisor -->|policy gate| hitl[human_review]
    analyst --> reviewer[reviewer_agent]
    reviewer -->|revise| analyst
    reviewer -->|escalate| hitl
    reviewer -->|approve| executor[executor_agent]
    hitl -->|approved| executor
    hitl -->|rework| analyst
    executor --> supervisor
    supervisor --> END([END])
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Key Configuration

- Project metadata: `APP_NAME`, `LOG_LEVEL`, `OUTPUT_ROOT`, `CHECKPOINT_MODE`
- LLM: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`
- Embeddings: `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`
- Vector retrieval: `VECTOR_DB_PROVIDER`, `QDRANT_URL`, `QDRANT_COLLECTION`, `RAG_*`
- Web grounding: `WEB_SEARCH_MODE`, `ENABLE_WEB_SEARCH`, `ENABLE_VECTOR_RAG`, `WEB_SEARCH_TOP_K`, `WEB_READER_TOP_K`
- Tavily: `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH`, `TAVILY_TOPIC`, `TAVILY_MAX_RESULTS`
- Jina Reader: `JINA_API_KEY`, `JINA_READER_BASE_URL`, `JINA_READER_TIMEOUT`
- Exa: `EXA_API_KEY`, `EXA_BASE_URL`, `EXA_SEARCH_TYPE`, `EXA_MAX_RESULTS`
- Governance policy: `GOVERNANCE_*` and `RISK_THRESHOLD_FOR_HUMAN_REVIEW`

## Build the Vector Index

```bash
python app/ingest.py --source-dir examples/knowledge_base --recreate-collection
```

## Run from CLI

```bash
python app/main.py \
  --task "Evaluate orchestration patterns for a regulated platform migration plan." \
  --task-type architecture \
  --auto-approve
```

## Run API

```bash
uvicorn app.api:app --reload
```

Create run:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"task":"Evaluate governance controls for a multi-agent platform.","task_type":"architecture","auto_approve":true}'
```

Continue after approval interrupt:

```bash
curl -X POST http://127.0.0.1:8000/runs/<task_id>/continue \
  -H "Content-Type: application/json" \
  -d '{"approved":true,"reviewer":"ops-reviewer","rationale":"Policy checks satisfied."}'
```

Inspect active providers:

```bash
curl http://127.0.0.1:8000/providers
```

## Testing

Pure logic tests:

```bash
pytest tests/test_config.py tests/test_evidence_pipeline.py tests/test_governance.py tests/test_grounding_merge.py
```

Integration smoke tests (network + provider credentials):

```bash
RUN_REAL_WEB_GROUNDING=1 pytest tests/integration/test_real_web_grounding.py
RUN_REAL_PROVIDER_RUN=1 pytest tests/integration/test_real_provider_run.py
```

Integration tests are env-gated and skip when required keys are not present.

## Documentation

- `docs/architecture.md`
- `docs/providers.md`
- `docs/rag.md`
- `docs/web_search.md`
- `docs/evidence.md`
- `docs/governance.md`
- `docs/testing.md`

## License

MIT (`LICENSE`).
