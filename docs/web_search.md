# Web Search and Reader Integration

## Supported Modes

- `WEB_SEARCH_MODE=tavily_jina`
- `WEB_SEARCH_MODE=exa`
- `WEB_SEARCH_MODE=auto`

## Tavily + Jina

Required:

- `ENABLE_WEB_SEARCH=true`
- `TAVILY_API_KEY=<key>`

Optional:

- `JINA_API_KEY=<key>`
- `JINA_READER_BASE_URL`
- `JINA_READER_TIMEOUT`

## Exa Search + Contents

Required:

- `ENABLE_WEB_SEARCH=true`
- `EXA_API_KEY=<key>`

Optional:

- `EXA_BASE_URL`
- `EXA_SEARCH_TYPE`
- `EXA_MAX_RESULTS`
- `EXA_USE_CONTENTS`

## Grounding Sequence

1. search provider returns URL candidates
2. reader/content provider enriches selected URLs
3. normalized records are merged with vector retrieval output
4. evidence pipeline scores and ranks the merged sources

## Provider Selection

In `auto` mode:

- Tavily + Jina is selected when `TAVILY_API_KEY` is present
- Exa stack is selected when `EXA_API_KEY` is present

Explicit provider overrides are also available through:

- `WEB_SEARCH_PROVIDER`
- `WEB_READER_PROVIDER`
