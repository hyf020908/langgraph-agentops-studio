import json
import asyncio
from threading import Barrier
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.api as api
import services.runtime as runtime_module
from app.runner import WorkflowRunner
from schemas.collaboration import CollaborationConfig
from schemas.models import TraceEvent
from services.collaboration import run_specialist
from services.config import Settings
from services.runtime import build_runtime


class LocalProvider:
    name = "local-test"

    def __init__(self, mode="parallel", fail=False):
        self.mode, self.fail = mode, fail
        self.barrier = Barrier(2) if mode == "parallel" else None

    def complete(self, *, system_prompt, user_prompt):
        payload = json.loads(user_prompt)
        if "tools_remaining" in payload:
            if self.fail:
                raise RuntimeError("private-provider-secret")
            if self.barrier:
                self.barrier.wait(timeout=5)
                assert payload["peer_results"] == []
            elif payload["role"] == "second":
                assert payload["peer_results"][0]["summary"] == "first insight"
            return json.dumps(
                {"action": "final", "summary": payload["role"] + " insight"}
            )
        schema = payload.get("response_schema", {})
        if "plan" in schema:
            return json.dumps(
                {
                    "plan": [
                        {
                            "step_id": f"P{i}",
                            "objective": payload["task"],
                            "owner": "research_agent",
                            "done_definition": "Evidence collected",
                        }
                        for i in range(4)
                    ],
                    "acceptance_criteria": [payload["task"]],
                    "search_queries": [payload["task"]],
                }
            )
        if "findings" in schema:
            assert len(payload["task_context"]["specialist_results"]) == 2
            return json.dumps(
                {
                    "findings": [
                        {
                            "finding_id": "F1",
                            "theme": "Deployment",
                            "insight": "Use durable storage",
                            "rationale": "Based on source and specialist review",
                            "evidence_ids": [
                                payload["ranked_evidence"][0]["evidence_id"]
                            ],
                            "risk_level": "low",
                        }
                    ]
                }
            )
        if "verdict" in schema:
            return json.dumps(
                {"verdict": "approve", "summary": "Reviewed", "score": 0.95}
            )
        return json.dumps(payload["memory"])


def make_runner(tmp_path, monkeypatch, mode, checkpoint="memory", fail=False):
    monkeypatch.setattr(
        runtime_module, "build_llm_provider", lambda _: LocalProvider(mode, fail)
    )
    monkeypatch.setattr(
        runtime_module,
        "build_embedding_provider",
        lambda _: SimpleNamespace(name="test"),
    )
    settings = Settings(
        output_root=str(tmp_path),
        checkpoint_mode=checkpoint,
        vector_db={"provider": "memory"},
        rag={"enabled": False, "hybrid_enabled": False},
        web_grounding={"enable_web_search": False},
    )
    runtime = build_runtime(settings)
    runtime.grounding.ground_query = lambda **kwargs: {
        "results": [
            {
                "source_id": "SRC-1",
                "provider": "memory",
                "source_type": "vector",
                "title": "Deployment",
                "url": "https://example.com/deployment",
                "snippet": "Durable storage improves deployment reliability.",
                "content": "Durable storage improves deployment reliability. Verify recovery with regression tests.",
                "credibility": 0.95,
                "score": 0.95,
            }
        ]
    }
    return WorkflowRunner(runtime)


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
@pytest.mark.parametrize("checkpoint", ["memory", "sqlite"])
def test_collaboration_graph_handoffs_audit_and_checkpoint_recovery(
    tmp_path, monkeypatch, mode, checkpoint
):
    runner = make_runner(tmp_path, monkeypatch, mode, checkpoint)
    config = {
        "mode": mode,
        "agents": [{"name": "one", "role": "first"}, {"name": "two", "role": "second"}],
    }
    try:
        state, interrupt = runner.start(
            "Deployment reliability",
            task_id="collaborating",
            collaboration=config,
            auto_approve=True,
        )
        assert state["status"] == "completed" and interrupt is None
        results = runner.summarize(state).collaboration_results
        assert [item.agent for item in results] == ["one", "two"]
        assert [item.summary for item in results] == ["first insight", "second insight"]
        assert state["collaboration_complete"]
        specialist_calls = [
            item
            for item in state["model_call_history"]
            if item.node.startswith("specialist:")
        ]
        assert len(specialist_calls) == 2
        assert len({message.id for message in state["messages"]}) == len(
            state["messages"]
        )
        artifact = json.loads(
            (tmp_path / "collaborating" / "run_artifact.json").read_text()
        )
        assert len(artifact["collaboration_results"]) == 2
        assert artifact["collaboration"]["mode"] == mode
        recovered, _ = runner.inspect("collaborating")
        assert len(recovered["collaboration_results"]) == 2
    finally:
        runner.close()
    if checkpoint == "sqlite":
        reopened = make_runner(tmp_path, monkeypatch, mode, checkpoint)
        try:
            state, _ = reopened.inspect("collaborating")
            assert len(reopened.summarize(state).collaboration_results) == 2
        finally:
            reopened.close()


def test_failed_specialists_escalate_and_secrets_do_not_enter_state(
    tmp_path, monkeypatch
):
    runner = make_runner(tmp_path, monkeypatch, "sequential", fail=True)
    try:
        state, interrupt = runner.start(
            "Deployment reliability",
            collaboration={
                "agents": [
                    {"name": "one", "role": "first"},
                    {"name": "two", "role": "second"},
                ]
            },
        )
        assert interrupt is not None
        assert "specialist_execution_failed" in interrupt["triggered_policies"]
        assert all(item.status == "error" for item in state["collaboration_results"])
        assert "private-provider-secret" not in str(state)
    finally:
        runner.close()


@pytest.mark.parametrize(
    "response,budget",
    [
        ({"action": "tool", "server": "db", "tool": "drop", "arguments": {}}, 1),
        ({"action": "tool", "server": "db", "tool": "query", "arguments": {}}, 0),
        (
            {
                "action": "final",
                "summary": "Invented citation",
                "source_ids": ["missing"],
            },
            1,
        ),
        ({"action": "other"}, 1),
    ],
)
def test_specialist_rejects_unlisted_tools_budget_overruns_and_invalid_output(
    response, budget
):
    calls = []
    runtime = SimpleNamespace(
        llm_provider=SimpleNamespace(
            name="fake", complete=lambda **kwargs: json.dumps(response)
        ),
        settings=Settings(),
        mcp=SimpleNamespace(
            tools_sync=lambda name: [
                {"name": "query", "allowed": True, "inputSchema": {}}
            ],
            call_sync=lambda *args: calls.append(args),
        ),
        trace=lambda node, status, message, metadata: TraceEvent(
            timestamp="now", node=node, status=status, message=message
        ),
    )
    result = run_specialist(
        runtime,
        {
            "specialist": {"name": "test", "role": "Review", "mcp_servers": ["db"]},
            "user_request": "test",
            "tool_budget": budget,
        },
    )
    assert result["collaboration_results"][0].status == "error"
    assert not calls


def test_collaboration_validation_and_queue_payload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from services.jobs import JobStore

    with pytest.raises(ValidationError):
        CollaborationConfig.model_validate(
            {
                "agents": [
                    {"name": "same", "role": "one"},
                    {"name": "same", "role": "two"},
                ]
            }
        )
    monkeypatch.setenv("TASK_QUEUE_MODE", "rabbitmq")
    jobs = JobStore(str(tmp_path))
    monkeypatch.setattr(api, "_jobs", jobs)
    client = TestClient(api.app)
    config = {"mode": "sequential", "agents": [{"name": "expert", "role": "Review"}]}
    response = client.post(
        "/runs?wait=false",
        json={"task": "queue", "task_id": "collaboration", "collaboration": config},
    )
    assert response.status_code == 202
    payload = json.loads(jobs.latest("collaboration")["payload"])
    assert payload["collaboration"]["mode"] == "sequential"
    assert payload["collaboration"]["agents"][0]["name"] == "expert"


def test_queue_worker_executes_collaboration_payload(tmp_path, monkeypatch):
    from app.worker import execute_job
    from services.jobs import JobStore

    runner = make_runner(tmp_path, monkeypatch, "sequential", "sqlite")
    store = JobStore(str(tmp_path))
    job = store.enqueue(
        "worker-team",
        "start",
        {
            "task": "Deployment reliability",
            "task_id": "worker-team",
            "auto_approve": True,
            "collaboration": {
                "mode": "sequential",
                "agents": [
                    {"name": "one", "role": "first"},
                    {"name": "two", "role": "second"},
                ],
            },
        },
    )
    try:
        assert execute_job(store, runner, job["id"])
        result = json.loads(store.get(job["id"])["result"])
        assert (
            result["status"] == "completed"
            and len(result["collaboration_results"]) == 2
        )
    finally:
        runner.close()


def test_native_async_graph_executes_collaboration(tmp_path, monkeypatch):
    from schemas.state import initial_state

    runner = make_runner(tmp_path, monkeypatch, "parallel")
    state = initial_state("Deployment reliability", task_id="async-team")
    state["collaboration"] = CollaborationConfig(
        agents=[{"name": "one", "role": "first"}, {"name": "two", "role": "second"}]
    )
    try:
        result = asyncio.run(
            runner.graph.ainvoke(
                state, config={"configurable": {"thread_id": "async-team"}}
            )
        )
        assert len(result["collaboration_results"]) == 2
        assert all(item.status == "success" for item in result["collaboration_results"])
    finally:
        runner.close()
