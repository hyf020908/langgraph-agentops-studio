# LangGraph AgentOps Studio Web Frontend

This is the Web Frontend for LangGraph AgentOps Studio. It provides a Vite, React, and TypeScript console for launching workflow runs, checking backend/provider status, viewing returned run snapshots, inspecting artifact paths, and continuing human review gates.

## Install Dependencies

```bash
npm install
```

## Configure Environment

```bash
cp .env.example .env
```

Set the backend API base URL:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Do not place LLM, embedding, Tavily, Exa, or other provider API keys in the frontend environment. Provider credentials belong in the backend `.env`.

## Start Development Server

Start the backend from the repository root:

```bash
uvicorn app.api:app --reload
```

Start the frontend from this directory:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Build Production Assets

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Backend Connection

The frontend calls these existing backend endpoints:

- `GET /health`
- `GET /providers`
- `POST /runs`
- `POST /runs/{task_id}/continue`

The current backend does not provide a separate run status polling endpoint. The UI displays the latest response returned by create run or continue run.

## Troubleshooting

- Frontend cannot connect to the API: confirm `VITE_API_BASE_URL` points to the running FastAPI server.
- Browser reports a CORS error: confirm the backend `CORS_ORIGINS` includes `http://127.0.0.1:5173` or `http://localhost:5173`.
- Creating a run fails: check backend LLM, embedding, and web search credentials.
- No Tavily or Exa key is available: set `ENABLE_WEB_SEARCH=false` in the backend environment.
- A run is paused for review: use the Continue review panel to submit reviewer, rationale, and an approve or reject decision.
