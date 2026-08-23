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
