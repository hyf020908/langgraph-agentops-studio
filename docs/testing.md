# Testing Strategy

## Layer 1: Pure Logic Tests

These tests run without external network calls and validate deterministic logic:

- `tests/test_config.py`
- `tests/test_vectorstore.py` (fake clients; no vector database services required)
- `tests/test_observability.py` (real in-memory LangGraph checkpoint plus bounded call ledgers)
- `tests/test_api_runs.py`
- `tests/test_recommendation_detail.py`
- `tests/test_reporting_config.py`
- `tests/test_evidence_pipeline.py`
- `tests/test_governance.py`
- `tests/test_grounding_merge.py`

Run:

```bash
pytest tests/test_config.py tests/test_vectorstore.py tests/test_evidence_pipeline.py tests/test_governance.py tests/test_grounding_merge.py
```

Run the complete backend regression suite with `pytest -q`. For the frontend,
`npm run build` runs strict TypeScript checking before the Vite production
build; production preview should also be checked at desktop and mobile widths
after visual changes.

## Concurrency Acceptance

`tests/test_concurrency_acceptance.py` exercises:

- 64 simultaneous asynchronous queue submissions and durable job records.
- 16 complete concurrent workflows through the HTTP API with each of memory and SQLite checkpoints, including structured compression, message eviction, approval, export, and checkpoint reopening.
- Native asynchronous graph execution with the memory checkpointer. HTTP requests use the thread pool for the synchronous SQLite checkpointer.
- Concurrent ingestion during in-memory vector search.
- 32 trace archive merges on 16 threads and 64 artifact merges across four processes.
- Overload rejection, health availability, capacity release after failure, and cancellation of a queue wait without losing the durable job.
- Preservation of summary-only issues and complete long-message content under a bounded compression call budget.

The workflow tests use real graph nodes, reducers, checkpoints, filesystem writes, and API concurrency. External model, embedding, and grounding calls use local providers. These tests establish correctness for the covered concurrency levels on one local volume; broker delivery and external provider behavior require the environment-gated integration tests below.

Run with the dependency snapshot in `requirements.lock`:

```bash
pytest tests/test_concurrency_acceptance.py tests/test_context_compaction.py -q
pytest -q
```

## Layer 2: Integration Smoke Tests

These tests execute provider-backed code paths and are env-gated:

- `tests/integration/test_real_web_grounding.py`
- `tests/integration/test_real_provider_run.py`

Run:

```bash
RUN_REAL_WEB_GROUNDING=1 pytest tests/integration/test_real_web_grounding.py
RUN_REAL_PROVIDER_RUN=1 pytest tests/integration/test_real_provider_run.py
```

If required keys are missing, integration tests skip automatically.

## Required Environment for Integration

Provider run:

- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`

Web grounding:

- Tavily path: `TAVILY_API_KEY`
- or Exa path: `EXA_API_KEY`
