"""Persistent MCP connections and short-lived, isolated SDK sessions."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from schemas.mcp import MCPServerConfig
from services.config import MCPSettings


class MCPError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class MCPService:
    def __init__(self, settings: MCPSettings, output_root: str):
        self.settings = settings
        self.path = Path(output_root) / "mcp.sqlite"

    def require_enabled(self) -> None:
        if not self.settings.enabled:
            raise MCPError("MCP is disabled. Set MCP_ENABLED=true to enable it.", 503)

    def _connect(self):
        self.require_enabled()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS servers (name TEXT PRIMARY KEY, config TEXT NOT NULL)"
            )
            conn.commit()
            self.path.chmod(0o600)
        except Exception:
            conn.close()
            raise
        return conn

    def list_servers(self) -> list[MCPServerConfig]:
        with closing(self._connect()) as conn:
            return [
                MCPServerConfig.model_validate_json(row[0])
                for row in conn.execute("SELECT config FROM servers ORDER BY name")
            ]

    def get_server(self, name: str, *, active: bool = False) -> MCPServerConfig:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT config FROM servers WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise MCPError("MCP server not found.", 404)
        server = MCPServerConfig.model_validate_json(row[0])
        if active:
            self._check_policy(server)
        return server

    def save_server(
        self, server: MCPServerConfig, *, replace: bool = False
    ) -> MCPServerConfig:
        self.require_enabled()
        if server.enabled:
            self._check_policy(server)
        with closing(self._connect()) as conn, conn:
            if replace:
                cursor = conn.execute(
                    "UPDATE servers SET config = ? WHERE name = ?",
                    (server.model_dump_json(), server.name),
                )
                if cursor.rowcount == 0:
                    raise MCPError("MCP server not found.", 404)
            else:
                try:
                    conn.execute(
                        "INSERT INTO servers VALUES (?, ?)",
                        (server.name, server.model_dump_json()),
                    )
                except sqlite3.IntegrityError:
                    raise MCPError("MCP server already exists.", 409) from None
        return server

    def delete_server(self, name: str) -> None:
        with closing(self._connect()) as conn, conn:
            if (
                conn.execute("DELETE FROM servers WHERE name = ?", (name,)).rowcount
                == 0
            ):
                raise MCPError("MCP server not found.", 404)

    def _check_policy(self, server: MCPServerConfig) -> None:
        self.require_enabled()
        if not server.enabled:
            raise MCPError("MCP server is disabled.", 409)
        references = set(server.env.values()) | set(server.headers_env.values())
        if not references.issubset(self.settings.allowed_env_vars):
            raise MCPError(
                "Credential references must be listed in MCP_ALLOWED_ENV_VARS.", 403
            )
        if server.transport == "stdio" and (
            not self.settings.allow_stdio
            or server.command not in self.settings.allowed_commands
        ):
            raise MCPError(
                "stdio requires MCP_ALLOW_STDIO and an executable in MCP_ALLOWED_COMMANDS.",
                403,
            )

    @staticmethod
    def _resolve(references: dict[str, str]) -> dict[str, str]:
        values = {
            key: os.environ.get(reference) for key, reference in references.items()
        }
        if any(value is None for value in values.values()):
            raise MCPError(
                "A required MCP credential environment variable is missing.", 422
            )
        return values

    @asynccontextmanager
    async def _session(self, server: MCPServerConfig):
        self._check_policy(server)
        async with asyncio.timeout(self.settings.timeout):
            if server.transport == "stdio":
                parameters = StdioServerParameters(
                    command=server.command,
                    args=server.args,
                    env=self._resolve(server.env),
                )
                # Server stderr can contain connection strings; do not copy it into API logs.
                with open(os.devnull, "w") as errlog:
                    async with stdio_client(parameters, errlog=errlog) as streams:
                        async with ClientSession(
                            *streams,
                            read_timeout_seconds=timedelta(
                                seconds=self.settings.timeout
                            ),
                        ) as session:
                            await session.initialize()
                            yield session
            elif server.transport == "sse":
                async with sse_client(
                    server.url,
                    headers=self._resolve(server.headers_env),
                    timeout=self.settings.timeout,
                    sse_read_timeout=self.settings.timeout,
                ) as streams:
                    async with ClientSession(
                        *streams,
                        read_timeout_seconds=timedelta(seconds=self.settings.timeout),
                    ) as session:
                        await session.initialize()
                        yield session
            else:
                async with httpx.AsyncClient(
                    headers=self._resolve(server.headers_env),
                    timeout=self.settings.timeout,
                    follow_redirects=False,
                ) as client:
                    async with streamable_http_client(
                        server.url, http_client=client
                    ) as (read, write, _):
                        async with ClientSession(
                            read,
                            write,
                            read_timeout_seconds=timedelta(
                                seconds=self.settings.timeout
                            ),
                        ) as session:
                            await session.initialize()
                            yield session

    async def _perform(self, server: MCPServerConfig, operation):
        try:
            async with self._session(server) as session:
                return await operation(session)
        except MCPError:
            raise
        except Exception:
            # SDK exception groups and remote diagnostics may contain secrets.
            raise MCPError(
                "MCP operation failed or timed out. Check the server and credentials."
            ) from None

    async def list_tools(self, name: str) -> list[dict[str, Any]]:
        server = self.get_server(name, active=True)

        async def discover(session):
            tools, cursor, seen = [], None, set()
            while True:
                page = await session.list_tools(cursor=cursor)
                for tool in page.tools:
                    record = tool.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                    record["allowed"] = (
                        tool.name in server.allowed_tools or "*" in server.allowed_tools
                    )
                    tools.append(record)
                    if len(tools) >= self.settings.max_tools:
                        return tools
                cursor = page.nextCursor
                if not cursor or cursor in seen:
                    return tools
                seen.add(cursor)

        return await self._perform(server, discover)

    async def test_server(self, name: str) -> dict:
        tools = await self.list_tools(name)
        return {"status": "ok", "server": name, "tool_count": len(tools)}

    def _bound(self, result, server: MCPServerConfig) -> dict:
        payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        secrets = [
            value
            for value in self._resolve({**server.env, **server.headers_env}).values()
            if value
        ]

        def redact(value):
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
                return value
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        payload = redact(payload)
        raw = json.dumps(payload, ensure_ascii=False)
        if len(raw) <= self.settings.max_output_chars:
            return payload
        if "contents" in payload:
            uri = (
                payload["contents"][0]["uri"]
                if payload["contents"]
                else "mcp://truncated"
            )
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": raw[: self.settings.max_output_chars],
                    }
                ],
                "truncated": True,
            }
        return {
            "content": [
                {"type": "text", "text": raw[: self.settings.max_output_chars]}
            ],
            "isError": bool(payload.get("isError", False)),
            "truncated": True,
        }

    async def call_tool(self, name: str, tool_name: str, arguments: dict) -> dict:
        server = self.get_server(name, active=True)
        if tool_name not in server.allowed_tools and "*" not in server.allowed_tools:
            raise MCPError("Tool is not in the server's allowed_tools.", 403)
        if len(json.dumps(arguments)) > self.settings.max_output_chars:
            raise MCPError("Tool arguments exceed the configured size limit.", 422)

        async def call(session):
            return self._bound(await session.call_tool(tool_name, arguments), server)

        return await self._perform(server, call)

    async def list_resources(self, name: str) -> list[dict]:
        server = self.get_server(name, active=True)

        async def discover(session):
            resources, cursor, seen = [], None, set()
            while True:
                page = await session.list_resources(cursor=cursor)
                resources.extend(
                    item.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for item in page.resources
                )
                if (
                    len(resources) >= self.settings.max_tools
                    or not page.nextCursor
                    or page.nextCursor in seen
                ):
                    return resources[: self.settings.max_tools]
                cursor = page.nextCursor
                seen.add(cursor)

        return await self._perform(server, discover)

    async def read_resource(self, name: str, uri: str) -> dict:
        server = self.get_server(name, active=True)

        async def read(session):
            return self._bound(await session.read_resource(uri), server)

        return await self._perform(server, read)

    def tools_sync(self, name: str) -> list[dict]:
        return asyncio.run(self.list_tools(name))

    def call_sync(self, name: str, tool_name: str, arguments: dict) -> dict:
        return asyncio.run(self.call_tool(name, tool_name, arguments))
