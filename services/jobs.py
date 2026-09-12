"""Durable job registry / transactional outbox for the single-host deployment.

SQLite files and artifacts must reside on one shared local volume. This is not
a distributed database; multi-host deployment requires a shared database backend.
"""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from services.concurrency import CapacityExceeded, RunConflictError, validate_task_id


class JobStore:
    def __init__(self, root: str, max_pending: int = 1000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "jobs.sqlite"
        self.max_pending = max_pending
        with closing(self.connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, operation TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    result TEXT, published_at REAL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs(task_id)
                    WHERE status IN ('queued', 'running');
                CREATE INDEX IF NOT EXISTS jobs_task ON jobs(task_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS jobs_outbox ON jobs(status, published_at);
            """)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def enqueue(self, task_id: str, operation: str, payload: dict) -> dict:
        validate_task_id(task_id)
        if operation not in {"start", "continue"}:
            raise ValueError("Unsupported job operation.")
        now = time.time()
        identifier = uuid4().hex
        with closing(self.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if operation == "start" and conn.execute("SELECT 1 FROM jobs WHERE task_id=?", (task_id,)).fetchone():
                raise RunConflictError("task_id already exists. Inspect it or use a new task_id.")
            if conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= self.max_pending:
                raise CapacityExceeded("Durable task backlog is full.")
            try:
                conn.execute("INSERT INTO jobs(id,task_id,operation,payload,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                             (identifier, task_id, operation, json.dumps(payload), now, now))
            except sqlite3.IntegrityError as exc:
                raise RunConflictError("This run already has an active job.") from exc
        return self.get(identifier)

    def get(self, identifier: str) -> dict | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        return dict(row) if row else None

    def latest(self, task_id: str) -> dict | None:
        validate_task_id(task_id)
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE task_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (task_id,)).fetchone()
        return dict(row) if row else None

    def pending(self) -> list[dict]:
        # Replay uncertain publication after a grace interval. Consumers dedupe.
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE status='queued' AND (published_at IS NULL OR published_at < ?) ORDER BY created_at LIMIT 100", (time.time() - 60,)).fetchall()
        return [dict(row) for row in rows]

    def published(self, identifier):
        with closing(self.connect()) as conn, conn:
            conn.execute("UPDATE jobs SET published_at=? WHERE id=?", (time.time(), identifier))

    def running(self, identifier):
        with closing(self.connect()) as conn, conn:
            conn.execute("UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE id=?", (time.time(), identifier))

    def finish(self, identifier: str, result: dict, failed=False):
        with closing(self.connect()) as conn, conn:
            conn.execute("UPDATE jobs SET status=?, result=?, updated_at=? WHERE id=?",
                         ("failed" if failed else "done", json.dumps(result), time.time(), identifier))
