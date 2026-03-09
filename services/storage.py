from __future__ import annotations

from pathlib import Path
from typing import Any

from services.serialization import dumps


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, task_id: str) -> Path:
        directory = self.root / task_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_text(self, task_id: str, name: str, content: str) -> Path:
        path = self.run_dir(task_id) / name
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, task_id: str, name: str, payload: Any) -> Path:
        path = self.run_dir(task_id) / name
        path.write_text(dumps(payload), encoding="utf-8")
        return path

