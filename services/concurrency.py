"""Bounded admission and run ownership; no waiting queue consumes worker threads."""
from contextlib import contextmanager
from threading import Lock
import re


class CapacityExceeded(RuntimeError):
    pass


class RunConflictError(RuntimeError):
    pass


def validate_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id):
        raise ValueError("task_id must contain 1-128 ASCII letters, digits, underscores or hyphens.")
    return task_id


class RunAdmission:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._active: set[str] = set()
        self._lock = Lock()

    @contextmanager
    def claim(self, task_id: str):
        validate_task_id(task_id)
        with self._lock:
            if task_id in self._active:
                raise RunConflictError("This run is already executing.")
            if len(self._active) >= self.capacity:
                raise CapacityExceeded("Workflow capacity exhausted. Retry later.")
            self._active.add(task_id)
        try:
            yield
        finally:
            with self._lock:
                self._active.remove(task_id)
