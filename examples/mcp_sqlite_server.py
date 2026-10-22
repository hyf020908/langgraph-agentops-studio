"""Read-only SQLite MCP server. Set MCP_SQLITE_PATH and run with Python."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("SQLite read-only")


def authorize(action, arg1, arg2, database, trigger):
    return (
        sqlite3.SQLITE_OK
        if action
        in {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }
        else sqlite3.SQLITE_DENY
    )


@server.tool()
def query(sql: str) -> list[dict]:
    """Run a read-only SELECT against the configured database; return at most 100 rows."""
    path = Path(os.environ["MCP_SQLITE_PATH"]).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(authorize)
        # Abort expensive queries as well as statements that modify/attach databases.
        remaining = 1000

        def budget():
            nonlocal remaining
            remaining -= 1
            return int(remaining <= 0)

        conn.set_progress_handler(budget, 1000)
        return [dict(row) for row in conn.execute(sql).fetchmany(100)]


@server.resource("database://schema")
def schema() -> str:
    """List tables and their CREATE statements."""
    import json

    return json.dumps(query("SELECT name, sql FROM sqlite_master WHERE type = 'table'"))


if __name__ == "__main__":
    server.run(transport="stdio")
