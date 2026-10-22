import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.api as api
from schemas.mcp import MCPServerConfig
from services.config import MCPSettings, load_settings
from services.mcp import MCPError, MCPService

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


def stdio_config(**kwargs):
    return MCPServerConfig(
        name="local",
        transport="stdio",
        command=sys.executable,
        args=[FIXTURE],
        **kwargs,
    )


def service(root, **kwargs):
    return MCPService(
        MCPSettings(
            enabled=True, allow_stdio=True, allowed_commands=[sys.executable], **kwargs
        ),
        str(root),
    )


def test_registry_crud_persists_across_instances_and_enforces_policy(tmp_path):
    mcp = service(tmp_path)
    config = stdio_config(allowed_tools=["lookup"])
    mcp.save_server(config)
    reopened = service(tmp_path)
    assert reopened.list_servers() == [config]
    with pytest.raises(MCPError, match="already exists"):
        reopened.save_server(config)
    assert (
        reopened.save_server(
            config.model_copy(update={"enabled": False}), replace=True
        ).enabled
        is False
    )
    with pytest.raises(MCPError, match="disabled"):
        reopened.get_server("local", active=True)
    reopened.delete_server("local")
    with pytest.raises(MCPError, match="not found"):
        reopened.delete_server("local")
    with pytest.raises(MCPError, match="not found"):
        reopened.save_server(config, replace=True)
    assert reopened.list_servers() == []
    with pytest.raises(MCPError, match="disabled"):
        MCPService(MCPSettings(), str(tmp_path)).list_servers()
    with pytest.raises(MCPError, match="stdio requires"):
        MCPService(MCPSettings(enabled=True), str(tmp_path)).save_server(config)


@pytest.mark.parametrize(
    "config",
    [
        {"url": "file:///tmp/database"},
        {"url": "https://user:pass@example.com/mcp"},
        {"url": "https://example.com/mcp?token=secret"},
        {"url": "http://example.com", "command": "python"},
        {"transport": "stdio"},
        {
            "url": "https://example.com",
            "headers_env": {"Authorization": "Bearer secret"},
        },
        {"url": "https://example.com", "headers_env": {"bad\nheader": "TOKEN"}},
    ],
)
def test_invalid_configs_are_rejected(config):
    with pytest.raises(ValidationError):
        MCPServerConfig(name="test", **config)


def test_environment_references_are_operator_allowlisted(tmp_path, monkeypatch):
    config = stdio_config(
        env={"FIXTURE_SECRET": "TEST_MCP_SECRET"}, allowed_tools=["echo_secret"]
    )
    with pytest.raises(MCPError, match="MCP_ALLOWED_ENV_VARS"):
        service(tmp_path).save_server(config)
    mcp = service(tmp_path, allowed_env_vars=["TEST_MCP_SECRET"])
    mcp.save_server(config)
    with pytest.raises(MCPError, match="operation failed|missing"):
        mcp.call_sync("local", "echo_secret", {})
    monkeypatch.setenv("TEST_MCP_SECRET", 'token-"private-value')
    payload = mcp.call_sync("local", "echo_secret", {})
    assert "private-value" not in json.dumps(payload)
    assert "REDACTED" in json.dumps(payload)
    assert "private-value" not in mcp.path.read_bytes().decode(errors="ignore")


def test_stdio_tools_resources_limits_and_errors(tmp_path):
    mcp = service(tmp_path, max_output_chars=1000)
    mcp.save_server(stdio_config(allowed_tools=["lookup", "large", "fail"]))
    tools = mcp.tools_sync("local")
    assert any(tool["name"] == "lookup" and tool["allowed"] for tool in tools)
    assert not next(tool for tool in tools if tool["name"] == "slow")["allowed"]
    assert "SQLite" in json.dumps(
        mcp.call_sync("local", "lookup", {"query": "durability"})
    )
    assert mcp.call_sync("local", "large", {})["truncated"]
    assert mcp.call_sync("local", "fail", {})["isError"]
    assert asyncio.run(mcp.list_resources("local"))[0]["uri"] == "fixture://schema"
    assert "CREATE TABLE" in json.dumps(
        asyncio.run(mcp.read_resource("local", "fixture://schema"))
    )
    with pytest.raises(MCPError, match="allowed_tools"):
        mcp.call_sync("local", "slow", {})
    with pytest.raises(MCPError, match="size limit"):
        mcp.call_sync("local", "lookup", {"query": "x" * 2000})


def test_timeout_closes_session_and_next_operation_recovers(tmp_path):
    mcp = service(tmp_path, timeout=1.5)
    mcp.save_server(stdio_config(allowed_tools=["slow", "lookup"]))
    started = monotonic()
    with pytest.raises(MCPError, match="failed or timed out"):
        mcp.call_sync("local", "slow", {})
    assert monotonic() - started < 4
    assert not mcp.call_sync("local", "lookup", {"query": "recovered"}).get("isError")


@pytest.mark.parametrize(
    "transport,path", [("streamable-http", "/mcp"), ("sse", "/sse")]
)
def test_real_http_transports(tmp_path, monkeypatch, transport, path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, FIXTURE, "--transport", transport, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "FIXTURE_HTTP_AUTH": "Bearer local-test-auth"},
    )
    try:
        deadline = monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if process.poll() is not None or monotonic() > deadline:
                    pytest.fail("MCP test server failed to start")
                sleep(0.05)
        monkeypatch.setenv("MCP_TEST_AUTH", "Bearer local-test-auth")
        mcp = service(tmp_path, allowed_env_vars=["MCP_TEST_AUTH"])
        mcp.save_server(
            MCPServerConfig(
                name="remote",
                url=f"http://127.0.0.1:{port}{path}",
                transport=transport.replace("-", "_"),
                allowed_tools=["lookup"],
                headers_env={"Authorization": "MCP_TEST_AUTH"},
            )
        )
        assert asyncio.run(mcp.test_server("remote"))["status"] == "ok"
        assert "SQLite" in json.dumps(
            asyncio.run(mcp.call_tool("remote", "lookup", {"query": "network"}))
        )
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_api_crud_validation_and_no_llm_initialization(tmp_path, monkeypatch):
    mcp = service(tmp_path)
    monkeypatch.setenv("TASK_QUEUE_MODE", "inline")
    monkeypatch.setattr(api, "get_mcp_service", lambda: mcp)
    client = TestClient(api.app)
    config = stdio_config(allowed_tools=["lookup"]).model_dump()
    assert client.post("/mcp/servers", json=config).status_code == 201
    assert client.post("/mcp/servers", json=config).status_code == 409
    assert client.get("/mcp/servers").json()[0]["name"] == "local"
    assert client.get("/mcp/servers/local").status_code == 200
    assert client.put("/mcp/servers/other", json=config).status_code == 422
    assert client.post("/mcp/servers/local/test").json()["status"] == "ok"
    assert client.get("/mcp/servers/local/tools").status_code == 200
    assert (
        client.post(
            "/mcp/servers/local/tools/lookup/call", json={"arguments": {"query": "api"}}
        ).status_code
        == 200
    )
    assert client.post("/mcp/servers/local/tools/slow/call", json={}).status_code == 403
    assert client.get("/mcp/servers/local/resources").status_code == 200
    assert (
        client.post(
            "/mcp/servers/local/resources/read", json={"uri": "fixture://schema"}
        ).status_code
        == 200
    )
    config["enabled"] = False
    assert client.put("/mcp/servers/local", json=config).status_code == 200
    assert client.post("/mcp/servers/local/test").status_code == 409
    assert client.delete("/mcp/servers/local").status_code == 204
    assert client.get("/mcp/servers/local").status_code == 404


def test_mcp_settings_environment(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_ALLOWED_ENV_VARS", "DB_URL, MCP_TOKEN")
    monkeypatch.setenv("MCP_TIMEOUT", "5")
    load_settings.cache_clear()
    try:
        settings = load_settings().mcp
        assert (
            settings.enabled
            and settings.allowed_env_vars == ["DB_URL", "MCP_TOKEN"]
            and settings.timeout == 5
        )
    finally:
        load_settings.cache_clear()


def test_resource_truncation_and_short_secret_keep_valid_json(tmp_path, monkeypatch):
    from mcp.types import ReadResourceResult, TextResourceContents

    monkeypatch.setenv("SHORT_MCP_SECRET", "123")
    mcp = service(
        tmp_path, max_output_chars=1000, allowed_env_vars=["SHORT_MCP_SECRET"]
    )
    config = stdio_config(env={"TOKEN": "SHORT_MCP_SECRET"})
    result = ReadResourceResult(
        contents=[TextResourceContents(uri="fixture://large", text="secret123" * 1000)]
    )
    bounded = mcp._bound(result, config)
    assert bounded["truncated"] and bounded["contents"][0]["uri"] == "fixture://large"
    assert "123" not in json.dumps(bounded)
