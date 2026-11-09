"""Small process-local token bucket for the Notes completion endpoint."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class NoteCompletionRateLimiter:
    """Allow a short burst while enforcing a sustained request interval.

    This is intentionally process-local. It protects the current single-process
    deployment; a shared Redis-backed limiter is required before adding multiple
    API workers.
    """

    def __init__(self, *, interval_seconds: float = 3.0, capacity: int = 2) -> None:
        self.interval_seconds = interval_seconds
        self.capacity = capacity
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated_at=current)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, current - bucket.updated_at)
                bucket.tokens = min(
                    float(self.capacity),
                    bucket.tokens + elapsed / self.interval_seconds,
                )
                bucket.updated_at = current

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0

            retry_after = math.ceil((1.0 - bucket.tokens) * self.interval_seconds)
            return False, max(1, retry_after)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


note_completion_rate_limiter = NoteCompletionRateLimiter()
