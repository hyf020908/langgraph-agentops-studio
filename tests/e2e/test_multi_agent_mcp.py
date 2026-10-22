"""Socket-level API -> LLM HTTP -> MCP subprocess -> SQLite -> exported report."""

import json
import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def model_response(payload):
    if "tools_remaining" in payload:
        if payload["available_tools"] and not payload["observations"]:
            return {
                "action": "tool",
                "server": "sqlite",
                "tool": "query",
                "arguments": {"sql": "SELECT name, details FROM records"},
            }
        ids = [
            item["source_id"] for item in payload["observations"] if "source_id" in item
        ]
        if payload["observations"]:
            assert "durable-e2e" in json.dumps(payload["observations"])
        elif payload["role"] == "Peer reviewer":
            assert payload["peer_results"][0]["source_ids"]
        return {
            "action": "final",
            "summary": "Database evidence supports durable storage; verify recovery.",
            "source_ids": ids,
        }
    schema = payload.get("response_schema", {})
    if "plan" in schema:
        return {
            "plan": [
                {
                    "step_id": f"P{i}",
                    "objective": payload["task"],
                    "owner": "research_agent",
                    "done_definition": "Source verified",
                }
                for i in range(4)
            ],
            "acceptance_criteria": ["Verify durable storage"],
            "search_queries": [payload["task"]],
        }
    if "findings" in schema:
        evidence = payload["ranked_evidence"][0]
        assert "mcp://sqlite/" in json.dumps(evidence)
        assert len(payload["task_context"]["specialist_results"]) == 2
        return {
            "findings": [
                {
                    "finding_id": "F1",
                    "theme": "Database durability",
                    "insight": "Verify durable-e2e recovery",
                    "rationale": "SQLite query returned committed database records.",
                    "evidence_ids": [evidence["evidence_id"]],
                    "risk_level": "low",
                }
            ]
        }
    if "verdict" in schema:
        return {
            "verdict": "approve",
            "summary": "Database evidence reviewed",
            "score": 0.95,
        }
    return payload.get("memory", {})


class ProviderHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                output = {
                    "data": [
                        {"index": i, "embedding": [0.1] * 64}
                        for i, _ in enumerate(body["input"])
                    ]
                }
            else:
                payload = json.loads(body["messages"][-1]["content"])
                output = {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "local-e2e",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(model_response(payload)),
                            },
                        }
                    ],
                }
            raw = json.dumps(output).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception:
            self.send_error(500, "Local provider assertion failed")

    def log_message(self, *args):
        pass


@pytest.fixture
def live_api(tmp_path):
    database = tmp_path / "database.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("CREATE TABLE records (name TEXT, details TEXT)")
        conn.execute(
            "INSERT INTO records VALUES (?, ?)",
            (
                "durable-e2e",
                "Durable storage preserves committed records. Test recovery before deployment.",
            ),
        )
        conn.commit()
    provider = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    thread = Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "CHECKPOINT_MODE": "sqlite",
        "TASK_QUEUE_MODE": "inline",
        "LLM_PROVIDER": "openai_compatible",
        "LLM_MODEL": "local-e2e",
        "LLM_API_KEY": "local-fixture",
        "LLM_BASE_URL": f"http://127.0.0.1:{provider.server_port}/v1",
        "LLM_TIMEOUT": "5",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": "local-e2e",
        "EMBEDDING_API_KEY": "local-fixture",
        "EMBEDDING_BASE_URL": f"http://127.0.0.1:{provider.server_port}/v1",
        "EMBEDDING_DIMENSIONS": "64",
        "VECTOR_DB_PROVIDER": "memory",
        "RAG_ENABLED": "false",
        "RAG_HYBRID_ENABLED": "false",
        "ENABLE_WEB_SEARCH": "false",
        "ENABLE_VECTOR_RAG": "false",
        "MCP_ENABLED": "true",
        "MCP_ALLOW_STDIO": "true",
        "MCP_ALLOWED_COMMANDS": sys.executable,
        "MCP_ALLOWED_ENV_VARS": "E2E_DATABASE",
        "E2E_DATABASE": str(database),
        "LOG_LEVEL": "ERROR",
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process = None
    log = open(tmp_path / "api.log", "w+")
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
    )

    def restart():
        nonlocal process
        if process:
            process.terminate()
            process.wait(timeout=10)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        )
        deadline = monotonic() + 15
        while monotonic() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if process.poll() is not None:
                break
            sleep(0.05)
        log.seek(0)
        pytest.fail("API failed to start: " + log.read()[-3000:])

    try:
        restart()
        yield client, tmp_path, restart
    finally:
        client.close()
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        log.close()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_database_collaboration_complete_http_lifecycle(live_api, mode):
    client, root, restart = live_api
    server = {
        "name": "sqlite",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(ROOT / "examples" / "mcp_sqlite_server.py")],
        "env": {"MCP_SQLITE_PATH": "E2E_DATABASE"},
        "allowed_tools": ["query"],
    }
    response = client.post("/mcp/servers", json=server)
    assert response.status_code == 201, response.text
    assert client.post("/mcp/servers/sqlite/test").json()["status"] == "ok"
    assert (
        "CREATE TABLE records"
        in client.post(
            "/mcp/servers/sqlite/resources/read", json={"uri": "database://schema"}
        ).text
    )
    result = client.post(
        "/mcp/servers/sqlite/tools/query/call",
        json={"arguments": {"sql": "DROP TABLE records"}},
    )
    assert result.status_code == 200 and result.json()["isError"]
    agents = [
        {"name": "database", "role": "Database analyst", "mcp_servers": ["sqlite"]},
        {
            "name": "reviewer",
            "role": "Peer reviewer" if mode == "sequential" else "Independent reviewer",
            "mcp_servers": [] if mode == "sequential" else ["sqlite"],
        },
    ]
    response = client.post(
        "/runs",
        json={
            "task": "Assess durable storage reliability",
            "task_id": "e2e-run",
            "auto_approve": False,
            "collaboration": {"mode": mode, "agents": agents},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert all(
        item["status"] == "success" for item in result["collaboration_results"]
    ), result
    if result["approval_required"]:
        restart()
        paused = client.get("/runs/e2e-run").json()
        assert paused["approval_required"] and len(paused["collaboration_results"]) == 2
        response = client.post("/runs/e2e-run/continue", json={"approved": True})
        assert response.status_code == 200, response.text
        result = response.json()
    assert result["status"] == "completed", result
    assert "durable-e2e" in result["final_report"]
    assert any(
        call["tool_name"] == "mcp:sqlite:query" and call["status"] == "success"
        for call in result["tool_call_history"]
    )
    artifact = json.loads((root / "runs" / "e2e-run" / "run_artifact.json").read_text())
    assert len(artifact["collaboration_results"]) == 2
    state = json.loads((root / "runs" / "e2e-run" / "state_snapshot.json").read_text())
    assert any(item["provider"] == "mcp" for item in state["retrieved_sources"])
    assert any(
        "mcp://sqlite/" in json.dumps(item["citations"])
        for item in state["ranked_evidence"]
    )
    assert len(result["collaboration_results"]) == 2
    assert (
        client.post(
            "/runs", json={"task": "duplicate", "task_id": "e2e-run"}
        ).status_code
        == 409
    )
    restart()
    assert client.get("/mcp/servers/sqlite").json()["name"] == "sqlite"
    reopened = client.get("/runs/e2e-run").json()
    assert (
        reopened["status"] == "completed"
        and len(reopened["collaboration_results"]) == 2
    )
    assert client.delete("/mcp/servers/sqlite").status_code == 204
