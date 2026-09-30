from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import sleep

import httpx
import pytest

import app.api as api
import services.vectorstore as vectorstore
from schemas.models import TraceEvent
from services.config import ContextCompactionSettings, VectorDBSettings
from services.context_compaction import ContextCompactionService
from services.jobs import JobStore
from services.storage import LocalArtifactStore


class _Embeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _compaction(root, storage=None):
    return ContextCompactionService(
        settings=ContextCompactionSettings(trace_tail_size=1),
        storage=storage or LocalArtifactStore(root), embeddings=_Embeddings(),
    )


def test_concurrent_trace_archives_preserve_all_events(tmp_path):
    class DelayedStore(LocalArtifactStore):
        def write_json(self, task_id, name, payload):
            sleep(0.01)
            return super().write_json(task_id, name, payload)
    svc = _compaction(tmp_path, DelayedStore(tmp_path))
    def archive(index):
        state = {"task_id": "shared", "execution_trace": [
            TraceEvent(timestamp=str(index), node="research", status="completed", message=str(index)),
            TraceEvent(timestamp="tail", node="tail", status="completed", message="tail"),
        ]}
        update, pointer = svc.archive_trace_if_needed(state)
        assert pointer is not None
        assert update["execution_trace"]["__replace__"]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(archive, range(32)))
    payload = svc.storage.read_json("shared", "context/execution_trace_archive.json")
    assert {event["message"] for event in payload["trace"]} == {str(i) for i in range(32)}


def test_vector_search_survives_concurrent_ingestion(tmp_path, monkeypatch):
    store = vectorstore.InMemoryVectorStore(VectorDBSettings(provider="memory"))
    store.ensure_collection(2)
    store.upsert(ids=["old"], vectors=[[1.0, 0.0]], payloads=[{"text": "old"}])
    entered, release = Event(), Event()
    cosine = vectorstore._cosine_similarity
    def paused_cosine(left, right):
        entered.set()
        assert release.wait(5)
        return cosine(left, right)
    monkeypatch.setattr(vectorstore, "_cosine_similarity", paused_cosine)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(store.search, query_vector=[1.0, 0.0], top_k=5)
        assert entered.wait(5)
        try:
            store.upsert(ids=["new"], vectors=[[1.0, 0.0]], payloads=[{"text": "new"}])
        finally:
            release.set()
        assert future.result()[0]["id"] == "old"
    assert {item["id"] for item in store.search(query_vector=[1.0, 0.0], top_k=5)} == {"old", "new"}


def test_refreshed_memory_keeps_summary_only_open_issues(tmp_path):
    svc = _compaction(tmp_path)
    memory = svc.build_task_memory({"user_request": "goal", "task_memory": {
        "open_issues": ["Verify deployment owner"], "compression_level": "semantic",
    }})
    assert memory.open_issues == ["Verify deployment owner"]


def test_async_queue_burst_is_durable_and_does_not_block_health(tmp_path, monkeypatch):
    store = JobStore(str(tmp_path), max_pending=200)
    monkeypatch.setenv("TASK_QUEUE_MODE", "rabbitmq")
    monkeypatch.setattr(api, "_jobs", store)
    for field in ("_active_requests", "_active_writes", "_active_ingests"):
        monkeypatch.setattr(api, field, 0)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            submissions = [asyncio.create_task(client.post("/runs?wait=false", json={
                "task": f"Task {i}", "task_id": f"burst-{i}"})) for i in range(64)]
            health = await asyncio.wait_for(client.get("/health"), timeout=5)
            assert health.status_code == 200
            results = await asyncio.wait_for(asyncio.gather(*submissions), timeout=20)
            assert all(response.status_code == 202 for response in results)
            assert {r.json()["task_id"] for r in results} == {f"burst-{i}" for i in range(64)}
    asyncio.run(run())
    assert len(store.pending()) == 64
    assert api._active_requests == api._active_writes == api._active_ingests == 0


@pytest.mark.parametrize("checkpoint_mode", ["memory", "sqlite"])
def test_async_full_workflows_isolate_context_and_recover_checkpoints(tmp_path, monkeypatch, checkpoint_mode):
    from app.runner import WorkflowRunner
    from services.checkpoint import build_checkpointer
    from services.config import Settings
    from services.runtime import build_runtime
    import services.runtime as runtime_module

    class Provider:
        name = "local-test"
        def complete(self, *, system_prompt, user_prompt):
            sleep(0.005)
            payload = json.loads(user_prompt)
            schema = payload.get("response_schema", {})
            if "plan" in schema:
                task = payload["task"]
                return json.dumps({"plan": [
                    {"step_id": f"P{i}", "objective": task, "owner": "research_agent", "done_definition": "Evidence collected"}
                    for i in range(1, 5)], "acceptance_criteria": [task], "search_queries": [task]})
            if "findings" in schema:
                task = payload["task"]
                assert payload["task_context"]["current_goal"] == task
                return json.dumps({"findings": [{"finding_id": "F1", "theme": "Deployment", "insight": task,
                                                 "rationale": "Use the supplied evidence", "evidence_ids": [payload["ranked_evidence"][0]["evidence_id"]],
                                                 "risk_level": "low"}]})
            if "verdict" in schema:
                return json.dumps({"verdict": "approve", "summary": "Reviewed", "score": 0.95})
            return json.dumps(payload["memory"])
    provider = Provider()
    monkeypatch.setattr(runtime_module, "build_llm_provider", lambda _: provider)
    monkeypatch.setattr(runtime_module, "build_embedding_provider", lambda _: _Embeddings())
    monkeypatch.setattr(runtime_module, "build_vector_store", lambda settings: vectorstore.InMemoryVectorStore(settings))
    monkeypatch.setattr(runtime_module, "build_web_search_provider", lambda _: None)
    monkeypatch.setattr(runtime_module, "build_web_reader_provider", lambda _: None)
    settings = Settings(output_root=str(tmp_path), checkpoint_mode=checkpoint_mode, max_concurrent_runs=16,
                        context_compaction=ContextCompactionSettings(max_estimated_tokens=1000, recent_messages=1, trace_tail_size=2))
    settings.rag.enabled = False
    settings.rag.hybrid_enabled = False
    runtime = build_runtime(settings)
    def ground_query(*, query):
        return {"results": [{"source_id": "SRC-" + query, "provider": "memory", "source_type": "vector",
                             "title": query, "url": "https://example.com/" + query, "domain": "example.com",
                             "snippet": query + " deployment evidence", "content": (query + " tested deployment practices ") * 100,
                             "credibility": 0.95, "score": 0.95}]}
    runtime.grounding.ground_query = ground_query
    runner = WorkflowRunner(runtime)
    monkeypatch.setenv("TASK_QUEUE_MODE", "inline")
    monkeypatch.setattr(api, "_runner", runner)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    for field in ("_active_requests", "_active_writes", "_active_ingests"):
        monkeypatch.setattr(api, field, 0)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            tasks = [asyncio.create_task(client.post("/runs", json={
                "task": f"workflow-{i}", "task_id": f"workflow-{i}", "auto_approve": True})) for i in range(16)]
            await asyncio.sleep(0)
            assert (await asyncio.wait_for(client.get("/health"), 5)).status_code == 200
            responses = await asyncio.wait_for(asyncio.gather(*tasks), 45)
            for i, response in enumerate(responses):
                assert response.status_code == 200, response.text
                assert response.json()["task_id"] == f"workflow-{i}"
                assert response.json()["status"] == "completed", response.text
            assert (await client.post("/runs", json={"task": "duplicate", "task_id": "workflow-0"})).status_code == 409
            if checkpoint_mode == "memory":
                from langgraph.types import Command
                from schemas.state import initial_state
                async def native(index):
                    task_id = f"native-{index}"
                    config = {"configurable": {"thread_id": task_id}}
                    state = await runner.graph.ainvoke(initial_state(task_id, task_id=task_id), config=config)
                    if runner._extract_interrupt_payload(state) is not None:
                        state = await runner.graph.ainvoke(Command(resume={"approved": True}), config=config)
                    assert state["status"] == "completed"
                    assert state["task_memory"]["current_goal"] == task_id
                    assert len(state["model_call_history"]) == 3
                await asyncio.wait_for(asyncio.gather(*(native(i) for i in range(4))), 20)
    try:
        asyncio.run(run())
        for i in range(16):
            state, interrupt = runner.inspect(f"workflow-{i}")
            assert interrupt is None
            assert state["task_memory"]["current_goal"] == f"workflow-{i}"
            assert state["context_compaction_history"][0].stats.fallback_reason is None
            assert state["model_call_history"][0].node == "planner_agent"
            assert len(state["model_call_history"]) == 3
            assert len(state["context_compaction_history"]) == 2
            from langchain_core.messages import ToolMessage
            assert not any(isinstance(m, ToolMessage) for m in state["messages"])
            trace = runtime.storage.read_json(f"workflow-{i}", "workflow_trace.json")["trace"]
            assert any(event["node"] == "initialize_task" for event in trace)
            for pointer in state["context_artifacts"]:
                archive = runtime.storage.read_json(f"workflow-{i}", pointer.path)
                assert archive
        assert api._active_requests == api._active_writes == 0
        if checkpoint_mode == "sqlite":
            reopened = build_checkpointer(settings)
            try:
                snapshot = reopened.get_tuple({"configurable": {"thread_id": "workflow-15"}})
                assert snapshot.checkpoint["channel_values"]["task_memory"]["current_goal"] == "workflow-15"
            finally:
                reopened.conn.close()
    finally:
        runner.close()


def test_artifact_merge_is_safe_across_processes(tmp_path):
    import subprocess
    import sys
    script = '''
import sys
from services.storage import LocalArtifactStore
store = LocalArtifactStore(sys.argv[1])
worker = int(sys.argv[2])
for index in range(16):
    store.update_json("shared", "merged.json", lambda previous: (previous or []) + [worker * 16 + index])
'''
    def execute(index):
        subprocess.run([sys.executable, "-c", script, str(tmp_path), str(index)],
                       check=True, capture_output=True, text=True, timeout=20)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(execute, range(4)))
    assert sorted(LocalArtifactStore(tmp_path).read_json("shared", "merged.json")) == list(range(64))


def test_async_overload_and_failure_release_capacity(monkeypatch):
    from threading import Lock
    from schemas.models import RunResponse
    from services.config import Settings
    settings = Settings(max_concurrent_runs=8)
    entered, release = Event(), Event()
    lock = Lock()
    class Runner:
        count = 0
        def start(self, *, task, task_id, **kwargs):
            with lock:
                self.count += 1
                if self.count == 8:
                    entered.set()
            assert release.wait(5)
            if task == "fail":
                raise RuntimeError("failure")
            return {"task_id": task_id, "status": "completed"}, None
        def summarize(self, state, interrupt):
            return RunResponse(**state, approval_required=False)
    monkeypatch.setenv("TASK_QUEUE_MODE", "inline")
    monkeypatch.setattr(api, "_runner", Runner())
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    for field in ("_active_requests", "_active_writes", "_active_ingests"):
        monkeypatch.setattr(api, field, 0)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            tasks = [asyncio.create_task(client.post("/runs", json={"task": "fail" if i == 0 else "work", "task_id": f"overload-{i}"}))
                     for i in range(64)]
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert (await asyncio.wait_for(client.get("/health"), 5)).status_code == 200
            finally:
                release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), 10)
            codes = [r.status_code for r in responses]
            assert codes.count(200) == 7
            assert codes.count(500) == 1
            assert codes.count(503) == 56
            assert all(r.headers["Retry-After"] == "2" for r in responses if r.status_code == 503)
            assert api._active_requests == api._active_writes == 0
            assert (await client.post("/runs", json={"task": "work", "task_id": "after-overload"})).status_code == 200
    asyncio.run(run())


def test_summary_processes_long_message_tail_and_enforces_call_budget(tmp_path):
    from langchain_core.messages import AIMessage
    svc = _compaction(tmp_path)
    class Provider:
        seen = []
        def complete(self, *, system_prompt, user_prompt):
            payload = json.loads(user_prompt)
            self.seen.extend(item["content"] for item in payload["conversation"])
            return json.dumps(payload["memory"])
    svc.llm_provider = Provider()
    state = {"task_id": "long", "user_request": "goal", "messages": [
        AIMessage(content="x" * 12000 + "IMPORTANT_TAIL", id="long"), AIMessage(content="recent", id="recent")],
    }
    svc.settings.recent_messages = 1
    summary, pointer = svc.summarize_old_conversation_if_needed(state, "critical")
    assert summary and pointer
    assert "IMPORTANT_TAIL" in "".join(svc.llm_provider.seen)
    svc.settings.summary_max_calls = 1
    svc.llm_provider.seen = []
    state["messages"][0].content = "x" * 100000
    assert svc.summarize_old_conversation_if_needed(state, "critical") == (None, None)
    assert svc.llm_provider.seen == []


def test_cancelled_queue_wait_releases_request_but_keeps_durable_job(tmp_path, monkeypatch):
    store = JobStore(str(tmp_path))
    monkeypatch.setenv("TASK_QUEUE_MODE", "rabbitmq")
    monkeypatch.setattr(api, "_jobs", store)
    for field in ("_active_requests", "_active_writes", "_active_ingests"):
        monkeypatch.setattr(api, field, 0)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/runs", json={"task": "wait", "task_id": "cancelled-wait"}))
            try:
                async def wait_for_enqueue():
                    while await asyncio.to_thread(store.latest, "cancelled-wait") is None:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_for_enqueue(), 5)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert api._active_requests == api._active_writes == 0
    asyncio.run(run())
    assert store.latest("cancelled-wait")["status"] == "queued"
