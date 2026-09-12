from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

import app.api as api
from services.broker import publish_pending
from services.checkpoint import build_checkpointer
from services.concurrency import CapacityExceeded, RunAdmission, RunConflictError
from services.config import Settings, RAGSettings
from services.hybrid import BM25Index, Reranker, reciprocal_rank_fusion
from services.jobs import JobStore
from services.resilience import ProviderGuard, ProviderUnavailable
from services.storage import LocalArtifactStore


def test_admission_bounds_parallel_work_and_releases_after_failure():
    admission = RunAdmission(1)
    entered, release = Event(), Event()
    def work():
        with admission.claim("first"):
            entered.set()
            assert release.wait(5)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(work)
        assert entered.wait(5)
        try:
            with pytest.raises(CapacityExceeded), admission.claim("second"):
                pass
            with pytest.raises(RunConflictError), admission.claim("first"):
                pass
        finally:
            release.set()
        future.result()
    with pytest.raises(ValueError), admission.claim("second"):
        raise ValueError("failed")
    with admission.claim("second"):
        pass


def test_api_queue_submission_is_durable_and_duplicate_is_conflict(tmp_path, monkeypatch):
    store = JobStore(str(tmp_path))
    monkeypatch.setenv("TASK_QUEUE_MODE", "rabbitmq")
    monkeypatch.setattr(api, "_jobs", store)
    client = TestClient(api.app)
    request = {"task": "Assess the platform", "task_id": "queued-one"}
    response = client.post("/runs?wait=false", json=request)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert client.get("/runs/queued-one").json()["status"] == "queued"
    assert client.post("/runs?wait=false", json=request).status_code == 409
    assert JobStore(str(tmp_path)).latest("queued-one")["status"] == "queued"


def test_api_wait_compatibility_and_error_redaction(tmp_path, monkeypatch):
    store = JobStore(str(tmp_path))
    monkeypatch.setenv("TASK_QUEUE_MODE", "rabbitmq")
    monkeypatch.setattr(api, "_jobs", store)
    enqueue = store.enqueue
    def completed(*args):
        job = enqueue(*args)
        store.finish(job["id"], {"task_id": job["task_id"], "status": "completed", "approval_required": False})
        return job
    monkeypatch.setattr(store, "enqueue", completed)
    client = TestClient(api.app)
    assert client.post("/runs", json={"task": "test", "task_id": "wait"}).json()["status"] == "completed"
    monkeypatch.setenv("TASK_QUEUE_MODE", "inline")
    def broken():
        raise RuntimeError("secret-api-key")
    monkeypatch.setattr(api, "get_runner", broken)
    response = client.post("/runs", json={"task": "test"})
    assert response.status_code == 500
    assert "secret-api-key" not in response.text
    assert response.headers["X-Request-ID"]


def test_health_survives_overload(monkeypatch):
    monkeypatch.setattr(api, "_active_requests", 1000)
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    response = client.get("/providers")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"


@pytest.mark.parametrize("identifier", ["../escape", "/tmp/escape", "a/b", "..", ""])
def test_artifact_task_id_cannot_escape_root(tmp_path, identifier):
    with pytest.raises(ValueError):
        LocalArtifactStore(tmp_path).write_json(identifier, "output.json", {})


def test_atomic_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    path = store.write_json("run", "data.json", {"old": True})
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("services.storage.os.replace", fail)
    with pytest.raises(OSError):
        store.write_json("run", "data.json", {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(path.parent.iterdir()) == [path]


def test_sqlite_graph_survives_reopen(tmp_path):
    class State(TypedDict):
        value: int
    builder = StateGraph(State)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    settings = Settings(checkpoint_mode="sqlite", output_root=str(tmp_path))
    config = {"configurable": {"thread_id": "persist"}}
    saver = build_checkpointer(settings)
    graph = builder.compile(checkpointer=saver)
    assert graph.invoke({"value": 1}, config)["value"] == 2
    saver.conn.close()
    reopened = build_checkpointer(settings)
    try:
        assert builder.compile(checkpointer=reopened).get_state(config).values["value"] == 2
    finally:
        reopened.conn.close()


def test_outbox_retries_unconfirmed_publish(tmp_path):
    store = JobStore(str(tmp_path))
    job = store.enqueue("outbox", "start", {"task": "test"})
    class Channel:
        def basic_publish(self, **kwargs):
            raise ConnectionError("confirm lost")
    with pytest.raises(ConnectionError):
        publish_pending(store, Channel(), "runs")
    assert store.get(job["id"])["published_at"] is None
    sent = []
    channel = SimpleNamespace(basic_publish=lambda **kwargs: sent.append(kwargs))
    publish_pending(store, channel, "runs")
    assert sent[0]["mandatory"] is True
    assert sent[0]["properties"].delivery_mode == 2
    assert store.get(job["id"])["published_at"] is not None


def test_backlog_and_concurrent_enqueue_are_atomic(tmp_path):
    store = JobStore(str(tmp_path), max_pending=1)
    def create(_):
        try:
            return store.enqueue("same", "start", {})
        except (RunConflictError, CapacityExceeded):
            return None
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(create, range(20)))
    assert sum(result is not None for result in results) == 1
    with pytest.raises(CapacityExceeded):
        store.enqueue("another", "start", {})


def test_circuit_opens_and_allows_one_recovery_probe():
    from openai import APIConnectionError
    guard = ProviderGuard(failure_threshold=1, cooldown=100)
    with pytest.raises(APIConnectionError), guard.call():
        raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
    with pytest.raises(ProviderUnavailable), guard.call():
        pass
    guard.cooldown = 0
    with guard.call():
        with pytest.raises(ProviderUnavailable), guard.call():
            pass
    with guard.call():
        pass


def chunk(identifier, text):
    return {"point_id": identifier, "text": text, "payload": {"text": text, "chunk_id": identifier}}


def test_bm25_persists_handles_chinese_and_replaces_duplicates(tmp_path):
    path = str(tmp_path / "lexical.sqlite")
    index = BM25Index(path)
    index.upsert([chunk("a", "RabbitMQ 消息队列 故障恢复"), chunk("b", "vector search embedding")])
    assert BM25Index(path).search("消息队列", 5)[0]["id"] == "a"
    index.upsert([chunk("a", "PostgreSQL 事务")])
    assert index.search("RabbitMQ", 5) == []
    assert len(index.search("事务", 5)) == 1
    assert index.search('" OR * ()', 5) == []


def test_rrf_ignores_score_scales_and_duplicate_hits():
    a = {"id": "a", "score": 1000, "payload": {}}
    b = {"id": "b", "score": .01, "payload": {}}
    result = reciprocal_rank_fusion([[a, a, b], [b]])
    assert result[0]["id"] == "b"
    assert result[1]["score"] == pytest.approx(1 / 61)


def test_http_rerank_and_invalid_response_fallback():
    settings = RAGSettings(rerank_provider="http", rerank_url="https://rerank.example/v1/rerank")
    reranker = Reranker(settings)
    reranker.client.close()
    hits = [{"id": "a", "score": .1, "payload": {"text": "first"}}, {"id": "b", "score": .05, "payload": {"text": "second"}}]
    reranker.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": [{"index": 1, "relevance_score": .9}]})))
    try:
        assert reranker.rerank("query", hits, 1)[0]["id"] == "b"
    finally:
        reranker.close()
    reranker.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": [{"index": 999, "relevance_score": .9}]})))
    try:
        assert reranker.rerank("query", hits, 1)[0]["id"] == "a"
    finally:
        reranker.close()


def test_hybrid_returns_bm25_when_embeddings_fail(tmp_path):
    from services.retrieval import RetrievalService
    def unavailable(*args):
        raise TimeoutError("embedding unavailable")
    retrieval = RetrievalService(rag_settings=RAGSettings(lexical_path=str(tmp_path / "bm25.sqlite")),
        embeddings=SimpleNamespace(embed_query=unavailable), vector_store=SimpleNamespace(provider_name="test"))
    retrieval.lexical.upsert([chunk("a", "RabbitMQ durable queue")])
    try:
        results = retrieval.search("RabbitMQ")
        assert results[0]["content"] == "RabbitMQ durable queue"
        assert results[0]["metadata"]["retrieval_ranks"] == {"1": 1}
    finally:
        retrieval.reranker.close()


def test_worker_redelivery_does_not_repeat_completed_execution(tmp_path):
    from app.worker import execute_job
    from schemas.models import RunResponse
    store = JobStore(str(tmp_path))
    job = store.enqueue("dedupe", "start", {"task": "test", "task_id": "dedupe"})
    calls = []
    def start(**kwargs):
        calls.append(kwargs)
        return {"task_id": "dedupe", "status": "completed"}, None
    runner = SimpleNamespace(
        graph=SimpleNamespace(get_state=lambda config: SimpleNamespace(values={})),
        start=start,
        summarize=lambda state, interrupt: RunResponse(**state, approval_required=False),
    )
    assert execute_job(store, runner, job["id"]) is True
    assert execute_job(store, runner, job["id"]) is True
    assert len(calls) == 1
    assert store.get(job["id"])["status"] == "done"


def test_worker_recovers_checkpoint_after_process_loss(tmp_path):
    from app.worker import execute_job
    from schemas.models import RunResponse
    store = JobStore(str(tmp_path))
    job = store.enqueue("recover", "start", {"task": "test", "task_id": "recover"})
    store.running(job["id"])
    calls = []
    state = {"task_id": "recover", "status": "completed"}
    def invoke(value, config):
        calls.append(value)
        return state
    runner = SimpleNamespace(
        graph=SimpleNamespace(get_state=lambda config: SimpleNamespace(values=state, next=("node",)), invoke=invoke),
        inspect=lambda task_id: (state, None),
        _extract_interrupt_payload=lambda state: None,
        summarize=lambda state, interrupt: RunResponse(**state, approval_required=False),
    )
    assert execute_job(store, runner, job["id"])
    assert calls == [None]
    assert store.get(job["id"])["attempts"] == 2


def test_worker_never_applies_old_approval_to_new_checkpoint(tmp_path):
    from app.worker import execute_job
    from schemas.models import RunResponse
    store = JobStore(str(tmp_path))
    job = store.enqueue("approval", "continue", {"task_id": "approval", "checkpoint_id": "old", "approved": True})
    state = {"task_id": "approval", "status": "review_escalate"}
    runner = SimpleNamespace(
        graph=SimpleNamespace(get_state=lambda config: SimpleNamespace(values=state, next=("human_review",), config={"configurable": {"checkpoint_id": "new"}})),
        inspect=lambda task_id: (state, {"approval": "needed"}),
        summarize=lambda state, interrupt: RunResponse(**state, approval_required=True),
    )
    assert execute_job(store, runner, job["id"])
    assert json.loads(store.get(job["id"])["result"])["approval_required"]


def test_failed_worker_job_is_durable_and_not_reexecuted(tmp_path):
    from app.worker import execute_job
    store = JobStore(str(tmp_path))
    job = store.enqueue("failure", "start", {"task_id": "failure"})
    def broken(config):
        raise RuntimeError("private provider diagnostic")
    runner = SimpleNamespace(graph=SimpleNamespace(get_state=broken))
    assert execute_job(store, runner, job["id"]) is False
    assert execute_job(store, runner, job["id"]) is False
    persisted = store.get(job["id"])
    assert persisted["status"] == "failed"
    assert persisted["attempts"] == 1
    assert "private provider diagnostic" not in persisted["result"]
