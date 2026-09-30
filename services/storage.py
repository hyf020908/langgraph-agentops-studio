from __future__ import annotations

# Local artifact persistence.
# This store gives the executor a stable run directory per task ID and keeps the
# actual file-writing details out of graph nodes and export helpers.

import json
import os
import tempfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import portalocker

from services.serialization import dumps
from services.concurrency import validate_task_id


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, task_id: str) -> Path:
        # Exporters can assume the run directory exists after calling this helper.
        validate_task_id(task_id)
        directory = (self.root / task_id).resolve()
        if self.root.resolve() not in directory.parents:
            raise ValueError("Run directory escapes artifact root.")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _resolve_run_path(self, task_id: str, name: str) -> Path:
        run_directory = self.run_dir(task_id).resolve()
        path = (run_directory / name).resolve()
        if run_directory != path and run_directory not in path.parents:
            raise ValueError(f"Artifact path escapes run directory: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, task_id: str, name: str, content: str) -> Path:
        path = self._resolve_run_path(task_id, name)
        self._atomic_write(path, content)
        return path

    def write_json(self, task_id: str, name: str, payload: Any) -> Path:
        path = self._resolve_run_path(task_id, name)
        self._atomic_write(path, dumps(payload))
        return path

    def update_json(self, task_id: str, name: str, transform: Callable[[Any], Any]) -> Any:
        """Serialize read/merge/write across threads and processes on this volume."""
        path = self._resolve_run_path(task_id, name)
        lock_dir = self.run_dir(task_id) / ".artifact-locks"
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / (sha256(str(path).encode()).hexdigest() + ".lock")
        with portalocker.Lock(str(lock_path), mode="a", timeout=10, check_interval=0.01):
            try:
                previous = self.read_json(task_id, name)
            except FileNotFoundError:
                previous = None
            updated = transform(previous)
            self.write_json(task_id, name, updated)
            return updated

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        # Readers see either the old or complete new file, never partial JSON.
        fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read_text(self, task_id: str, name: str) -> str:
        path = self._resolve_run_path(task_id, name)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {name}")
        return path.read_text(encoding="utf-8")

    def read_json(self, task_id: str, name: str) -> Any:
        return json.loads(self.read_text(task_id, name))
