from __future__ import annotations

# Local artifact persistence.
# This store gives the executor a stable run directory per task ID and keeps the
# actual file-writing details out of graph nodes and export helpers.

import json
from pathlib import Path
from typing import Any

from services.serialization import dumps


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, task_id: str) -> Path:
        # Exporters can assume the run directory exists after calling this helper.
        directory = self.root / task_id
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
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, task_id: str, name: str, payload: Any) -> Path:
        path = self._resolve_run_path(task_id, name)
        path.write_text(dumps(payload), encoding="utf-8")
        return path

    def read_text(self, task_id: str, name: str) -> str:
        path = self._resolve_run_path(task_id, name)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {name}")
        return path.read_text(encoding="utf-8")

    def read_json(self, task_id: str, name: str) -> Any:
        return json.loads(self.read_text(task_id, name))
