# Provider Configuration

## LLM Providers

`LLM_PROVIDER` supports:

- `openai`
- `deepseek`
- `openai_compatible`

Key variables:

- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_TIMEOUT`
- `LLM_EXTRA_HEADERS_JSON`
- `LLM_EXTRA_KWARGS_JSON`

## Embedding Providers

`EMBEDDING_PROVIDER` supports:

- `openai`
- `deepseek`
- `openai_compatible`

Key variables:

- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_DIMENSIONS`
- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_TIMEOUT`
- `EMBEDDING_EXTRA_HEADERS_JSON`
- `EMBEDDING_EXTRA_KWARGS_JSON`

## Vector Store Providers

`VECTOR_DB_PROVIDER` supports:

- `qdrant`
- `memory`

Qdrant variables:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `QDRANT_LOCAL_PATH`

## Web Grounding Providers

Search:

- Tavily (`tavily`)
- Exa Search (`exa`)

Reader/content:

- Jina Reader (`jina`)
- Exa Contents (`exa`)

Mode selector:

- `WEB_SEARCH_MODE=auto`
- `WEB_SEARCH_MODE=tavily_jina`
- `WEB_SEARCH_MODE=exa`

## Governance Configuration

Policy controls are loaded from `GOVERNANCE_*` variables and `risk_threshold_for_human_review`:

- overall risk threshold
- recommendation confidence threshold
- evidence completeness threshold
- contradiction severity threshold
- unresolved question count threshold
- high-stakes category list
- manual approval policy by task type
