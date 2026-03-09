from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.runner import WorkflowRunner
from schemas.models import ContinueRequest, IngestRequest, IngestResponse, RunRequest, RunResponse


app = FastAPI(title="LangGraph AgentOps Studio API", version="1.0.0")
_runner: WorkflowRunner | None = None


def get_runner() -> WorkflowRunner:
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.summarize(state, interrupt_payload)


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        runner = get_runner()
        report = runner.runtime.retrieval.ingest_directory(
            source_dir=request.source_dir,
            recreate_collection=request.recreate_collection,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse.model_validate(report)
