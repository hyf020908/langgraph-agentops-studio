"""Process-local provider bulkhead and circuit breaker; SDK owns bounded retries."""
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from time import monotonic

from services.concurrency import CapacityExceeded


class ProviderUnavailable(RuntimeError):
    pass


class ProviderGuard:
    def __init__(self, capacity=8, failure_threshold=5, cooldown=30.0):
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe = False
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown

    @contextmanager
    def call(self):
        if not self._slots.acquire(blocking=False):
            raise CapacityExceeded("Provider concurrency limit reached.")
        probe = False
        try:
            with self._lock:
                if self._opened_at is not None:
                    if monotonic() - self._opened_at < self.cooldown or self._probe:
                        raise ProviderUnavailable("Provider circuit is open. Retry later.")
                    self._probe = probe = True
            try:
                yield
            except Exception as exc:
                from openai import APIConnectionError, APIStatusError

                transient = isinstance(exc, APIConnectionError) or (
                    isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)
                )
                with self._lock:
                    if transient:
                        self._failures += 1
                        if probe or self._failures >= self.failure_threshold:
                            self._opened_at = monotonic()
                    elif probe:
                        self._opened_at = None
                        self._failures = 0
                raise
            else:
                with self._lock:
                    # An older in-flight success must not close a newly open circuit.
                    if probe or self._opened_at is None:
                        self._failures = 0
                        self._opened_at = None
            finally:
                if probe:
                    with self._lock:
                        self._probe = False
        finally:
            self._slots.release()
