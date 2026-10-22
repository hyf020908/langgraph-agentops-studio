"""Real SDK server used by transport tests (no external credentials)."""

import argparse
import asyncio
import os

from mcp.server.fastmcp import FastMCP

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--transport", default="stdio")
args = parser.parse_args()
server = FastMCP(
    "test-server",
    port=args.port,
    log_level="ERROR",
    stateless_http=True,
    json_response=True,
)


@server.tool()
def lookup(query: str) -> dict:
    """Look up deployment evidence."""
    return {
        "query": query,
        "evidence": "SQLite durable transactions preserve committed records.",
    }


@server.tool()
def echo_secret() -> str:
    """Return a fixture secret to verify redaction."""
    return os.getenv("FIXTURE_SECRET", "fixture-secret")


@server.tool()
def large() -> str:
    """Return an oversized response."""
    return "x" * 10000


@server.tool()
async def slow() -> str:
    """Wait long enough to test timeout and cleanup."""
    await asyncio.sleep(5)
    return "done"


@server.tool()
def fail() -> str:
    """Report a tool error."""
    raise ValueError("fixture tool failed")


@server.resource("fixture://schema")
def schema() -> str:
    return "CREATE TABLE records (id INTEGER, name TEXT)"


if args.transport == "stdio":
    server.run(transport="stdio")
else:
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    async def authenticate(request, call_next):
        token = os.getenv("FIXTURE_HTTP_AUTH")
        if token and request.headers.get("Authorization") != token:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    app = (
        server.streamable_http_app()
        if args.transport == "streamable-http"
        else server.sse_app()
    )
    app.add_middleware(BaseHTTPMiddleware, dispatch=authenticate)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
