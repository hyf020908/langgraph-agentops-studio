from __future__ import annotations

# FastAPI surface for the workflow.
# The API mirrors the runner's lifecycle: create a run, resume an interrupted
# review gate, inspect provider wiring, and ingest knowledge documents.

import os
import sys
import logging
import re
import asyncio
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from uuid import uuid4
from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, Path as PathParameter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from services.concurrency import CapacityExceeded, RunConflictError
from services.resilience import ProviderUnavailable
from services.broker import queue_enabled
from services.jobs import JobStore
from services.config import load_settings
from services.mcp import MCPError, MCPService
from schemas.mcp import MCPResourceRequest, MCPServerConfig, MCPToolCallRequest


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from app.runner import RunNotFoundError, WorkflowRunner
from schemas.models import ContinueRequest, IngestRequest, IngestResponse, RunRequest, RunResponse


def cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app):
    global _runner
    if queue_enabled():
        if load_settings().checkpoint_mode != "sqlite":
            raise RuntimeError("RabbitMQ mode requires CHECKPOINT_MODE=sqlite.")
        vector = load_settings().vector_db
        if vector.provider == "qdrant" and not vector.qdrant_url:
            raise RuntimeError("RabbitMQ mode requires server Qdrant; set QDRANT_URL.")
        get_jobs()
    yield
    if _runner is not None:
        _runner.close()
        _runner = None


app = FastAPI(title="LangGraph AgentOps Studio API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_runner: WorkflowRunner | None = None
_runner_lock = Lock()
_jobs: JobStore | None = None
_jobs_lock = Lock()
_active_requests = 0
_active_writes = 0
_active_ingests = 0
_request_lock = Lock()
_request_count = 0
_rejected_count = 0
_request_seconds = 0.0
logger = logging.getLogger("agentops.api")


def get_jobs() -> JobStore:
    global _jobs
    with _jobs_lock:
        if _jobs is None:
            _jobs = JobStore(load_settings().output_root, max_pending=int(os.getenv("MAX_PENDING_JOBS", "1000")))
    return _jobs


@app.middleware("http")
async def admission_and_trace(request, call_next):
    # Reject overload before Starlette queues synchronous endpoints in its pool.
    global _active_requests, _active_writes, _active_ingests, _request_count, _rejected_count, _request_seconds
    started = monotonic()
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied_id) else uuid4().hex
    write = request.method == "POST"
    ingest = request.url.path == "/ingest" and write
    health_request = request.url.path == "/health"
    with _request_lock:
        overloaded = not health_request and (
            _active_requests >= (128 if queue_enabled() else 32)
            or (write and _active_writes >= (96 if queue_enabled() else load_settings().max_concurrent_runs))
            or (ingest and _active_ingests >= 1)
        )
        if not overloaded:
            _active_requests += 1
            _active_writes += int(write)
            _active_ingests += int(ingest)
    if overloaded:
        with _request_lock:
            _rejected_count += 1
        return JSONResponse({"detail": "Service capacity exhausted. Retry later."}, status_code=503,
                            headers={"Retry-After": "2", "X-Request-ID": request_id})
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("request_id=%s method=%s status=%s duration_ms=%.2f",
                    request_id, request.method, response.status_code, (monotonic() - started) * 1000)
        return response
    finally:
        with _request_lock:
            _request_count += 1
            _request_seconds += monotonic() - started
            _active_requests -= 1
            _active_writes -= int(write)
            _active_ingests -= int(ingest)


def api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MCPError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, RunNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RunConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (CapacityExceeded, ProviderUnavailable)):
        return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "2"})
    # Provider errors can include credentials, internal URLs and response bodies.
    return HTTPException(status_code=500, detail="Operation failed. Contact the operator with X-Request-ID.")


def get_runner() -> WorkflowRunner:
    # Reuse one runner instance so provider clients, vector store handles, and
    # compiled graph state are initialized once per process.
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = WorkflowRunner()
    return _runner


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    with _request_lock:
        body = (f"agentops_http_requests_total {_request_count}\n"
                f"agentops_http_rejected_total {_rejected_count}\n"
                f"agentops_http_active {_active_requests}\n"
                f"agentops_http_duration_seconds_sum {_request_seconds}\n")
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        runner = get_runner()
        runner.graph.get_state({"configurable": {"thread_id": "readiness-probe"}})
    except Exception as exc:
        logger.exception("Readiness check failed.")
        raise HTTPException(status_code=503, detail="Runtime is not ready.") from exc
    return {"status": "ready"}


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


def _create_run(request: RunRequest, response: Response) -> RunResponse:
    try:
        if request.collaboration is not None:
            for agent in request.collaboration.agents:
                for server in agent.mcp_servers:
                    get_mcp_service().get_server(server, active=True)
        if queue_enabled():
            task_id = request.task_id or f"task-{uuid4().hex}"
            job = get_jobs().enqueue(task_id, "start", {**request.model_dump(), "task_id": task_id})
            response.status_code = 202
            return RunResponse(task_id=task_id, status="queued", queue_status="queued", queue_job_id=job["id"], approval_required=False)
        runner = get_runner()
        options = {"collaboration": request.collaboration} if request.collaboration is not None else {}
        state, interrupt_payload = runner.start(
            task=request.task,
            task_id=request.task_id,
            auto_approve=request.auto_approve,
            task_type=request.task_type,
            **options,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run creation failed.")
        raise api_error(exc) from exc
    return runner.summarize(state, interrupt_payload)


async def wait_for_job(task_id: str, response: Response, job_id: str) -> RunResponse:
    deadline = monotonic() + 900
    while monotonic() < deadline:
        job = await run_in_threadpool(get_jobs().get, job_id)
        if job is not None and job["result"]:
            response.status_code = 200
            return RunResponse.model_validate_json(job["result"])
        await asyncio.sleep(1)
    raise HTTPException(status_code=504, detail="Task is still queued or running; inspect it using GET /runs/{task_id}.")


@app.post("/runs", response_model=RunResponse)
async def create_run(request: RunRequest, response: Response, wait: bool = True) -> RunResponse:
    result = await run_in_threadpool(_create_run, request, response)
    if queue_enabled() and wait:
        return await wait_for_job(result.task_id, response, result.queue_job_id)
    return result


@app.get("/runs/{task_id}", response_model=RunResponse)
def inspect_run(task_id: Annotated[str, PathParameter(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]) -> RunResponse:
    """Return the latest checkpoint so clients can poll a run in progress."""
    try:
        if queue_enabled():
            job = get_jobs().latest(task_id)
            if job is not None:
                if job["result"]:
                    return RunResponse.model_validate_json(job["result"]).model_copy(update={"queue_status": job["status"], "queue_job_id": job["id"]})
                if job["status"] == "queued":
                    return RunResponse(task_id=task_id, status="queued", queue_status="queued", queue_job_id=job["id"], approval_required=False)
        runner = get_runner()
        state, interrupt_payload = runner.inspect(task_id)
    except RunNotFoundError as exc:
        # A client may poll immediately after starting POST /runs, before the
        # first checkpoint is committed. Treating this as 404 keeps unknown and
        # not-yet-visible thread IDs explicit and retryable.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run inspection failed for task_id=%s.", task_id)
        raise api_error(exc) from exc
    return runner.summarize(state, interrupt_payload)


def _continue_run(task_id: str, request: ContinueRequest, response: Response) -> RunResponse:
    try:
        runner = get_runner()
        if queue_enabled():
            _, interrupt = runner.inspect(task_id)
            if interrupt is None:
                raise RunConflictError("Run is not awaiting approval.")
            snapshot = runner.graph.get_state({"configurable": {"thread_id": task_id}})
            checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
            job = get_jobs().enqueue(task_id, "continue", {**request.model_dump(), "task_id": task_id, "checkpoint_id": checkpoint_id})
            response.status_code = 202
            return RunResponse(task_id=task_id, status="queued", queue_status="queued", queue_job_id=job["id"], approval_required=False)
        state, interrupt_payload = runner.continue_run(
            task_id=task_id,
            approved=request.approved,
            reviewer=request.reviewer,
            rationale=request.rationale,
        )
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Run continuation failed for task_id=%s.", task_id)
        raise api_error(exc) from exc
    return runner.summarize(state, interrupt_payload)


@app.post("/runs/{task_id}/continue", response_model=RunResponse)
async def continue_run(task_id: Annotated[str, PathParameter(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")], request: ContinueRequest, response: Response, wait: bool = True) -> RunResponse:
    result = await run_in_threadpool(_continue_run, task_id, request, response)
    if queue_enabled() and wait:
        return await wait_for_job(result.task_id, response, result.queue_job_id)
    return result


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        runner = get_runner()
        # Ingestion is intentionally exposed as a separate API call so the
        # retrieval corpus can be refreshed independently of workflow runs.
        allowed_root = Path(runner.runtime.settings.rag.source_dir).resolve()
        source = Path(request.source_dir or allowed_root).resolve()
        if source != allowed_root and allowed_root not in source.parents:
            raise HTTPException(status_code=422, detail="source_dir must be inside RAG_SOURCE_DIR.")
        if request.recreate_collection:
            raise HTTPException(status_code=409, detail="Collection recreation requires an offline maintenance operation.")
        report = runner.runtime.retrieval.ingest_directory(
            source_dir=str(source),
            recreate_collection=request.recreate_collection,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        logger.exception("Document ingestion failed.")
        raise api_error(exc) from exc
    return IngestResponse.model_validate(report)


def get_mcp_service() -> MCPService:
    settings = load_settings()
    return MCPService(settings.mcp, settings.output_root)


@app.exception_handler(MCPError)
async def mcp_error_handler(request, exc: MCPError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)


@app.get("/mcp/servers", response_model=list[MCPServerConfig])
def list_mcp_servers():
    return get_mcp_service().list_servers()


@app.post("/mcp/servers", response_model=MCPServerConfig, status_code=201)
def create_mcp_server(request: MCPServerConfig):
    return get_mcp_service().save_server(request)


@app.get("/mcp/servers/{name}", response_model=MCPServerConfig)
def get_mcp_server(name: str):
    return get_mcp_service().get_server(name)


@app.put("/mcp/servers/{name}", response_model=MCPServerConfig)
def update_mcp_server(name: str, request: MCPServerConfig):
    if name != request.name:
        raise HTTPException(status_code=422, detail="Server name must match the URL.")
    return get_mcp_service().save_server(request, replace=True)


@app.delete("/mcp/servers/{name}", status_code=204)
def delete_mcp_server(name: str):
    get_mcp_service().delete_server(name)
    return Response(status_code=204)


@app.post("/mcp/servers/{name}/test")
async def test_mcp_server(name: str):
    return await get_mcp_service().test_server(name)


@app.get("/mcp/servers/{name}/tools")
async def list_mcp_tools(name: str):
    return await get_mcp_service().list_tools(name)


@app.post("/mcp/servers/{name}/tools/{tool_name}/call")
async def call_mcp_tool(name: str, tool_name: str, request: MCPToolCallRequest):
    return await get_mcp_service().call_tool(name, tool_name, request.arguments)


@app.get("/mcp/servers/{name}/resources")
async def list_mcp_resources(name: str):
    return await get_mcp_service().list_resources(name)


@app.post("/mcp/servers/{name}/resources/read")
async def read_mcp_resource(name: str, request: MCPResourceRequest):
    return await get_mcp_service().read_resource(name, request.uri)
