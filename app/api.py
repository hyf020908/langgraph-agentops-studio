from __future__ import annotations

# FastAPI surface for the workflow.
# The API mirrors the runner's lifecycle: create a run, resume an interrupted
# review gate, inspect provider wiring, and ingest knowledge documents.

import os
import sys
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from app.runner import RunNotFoundError, WorkflowRunner
from schemas.models import ContinueRequest, IngestRequest, IngestResponse, RunRequest, RunResponse


def cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="LangGraph AgentOps Studio API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_runner: WorkflowRunner | None = None
logger = logging.getLogger("agentops.api")


def get_runner() -> WorkflowRunner:
    # Reuse one runner instance so provider clients, vector store handles, and
    # compiled graph state are initialized once per process.
    global _runner
    if _runner is None:
        _runner = WorkflowRunner()
    return _runner


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/providers")
def providers() -> dict[str, str | None]:
    runner = get_runner()
    return {
        "llm": runner.runtime.llm_provider.name,
        "embedding": runner.runtime.embedding_provider.name,
        "vector_db": runner.runtime.vector_store.provider_name,
        "web_search": runner.runtime.web_search_provider.name if runner.runtime.web_search_provider else None,
        "web_reader": runner.runtime.web_reader_provider.name if runner.runtime.web_reader_provider else None,
    }


@app.post("/runs", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    try:
        runner = get_runner()
        state, interrupt_payload = runner.start(
            task=request.task,
            task_id=request.task_id,
            auto_approve=request.auto_approve,
            task_type=request.task_type,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run creation failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.summarize(state, interrupt_payload)


@app.get("/runs/{task_id}", response_model=RunResponse)
def inspect_run(task_id: str) -> RunResponse:
    """Return the latest checkpoint so clients can poll a run in progress."""
    try:
        runner = get_runner()
        state, interrupt_payload = runner.inspect(task_id)
    except RunNotFoundError as exc:
        # A client may poll immediately after starting POST /runs, before the
        # first checkpoint is committed. Treating this as 404 keeps unknown and
        # not-yet-visible thread IDs explicit and retryable.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run inspection failed for task_id=%s.", task_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.summarize(state, interrupt_payload)


@app.post("/runs/{task_id}/continue", response_model=RunResponse)
def continue_run(task_id: str, request: ContinueRequest) -> RunResponse:
    try:
        runner = get_runner()
        state, interrupt_payload = runner.continue_run(
            task_id=task_id,
            approved=request.approved,
            reviewer=request.reviewer,
            rationale=request.rationale,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run continuation failed for task_id=%s.", task_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.summarize(state, interrupt_payload)


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        runner = get_runner()
        # Ingestion is intentionally exposed as a separate API call so the
        # retrieval corpus can be refreshed independently of workflow runs.
        report = runner.runtime.retrieval.ingest_directory(
            source_dir=request.source_dir,
            recreate_collection=request.recreate_collection,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Document ingestion failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse.model_validate(report)
