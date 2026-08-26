"""Cooperative cancel and deadline control for one Chat Agent run."""

from __future__ import annotations

import threading
import time
from enum import Enum


class AbortKind(str, Enum):
    CANCELLED = "cancelled"
    QUEUE_WAIT_TIMEOUT = "queue_wait_timeout"
    AGENT_TIMEOUT = "agent_timeout"


class RunAborted(Exception):
    """Raised at a cooperative checkpoint when the run must stop."""

    def __init__(self, kind: AbortKind):
        self.kind = kind
        super().__init__(kind.value)

    @property
    def error_code(self) -> str:
        return self.kind.value

    @property
    def retryable(self) -> bool:
        return self.kind is not AbortKind.CANCELLED

    @property
    def public_message(self) -> str:
        if self.kind is AbortKind.QUEUE_WAIT_TIMEOUT:
            return "Chat waited too long to start. Please try again."
        if self.kind is AbortKind.AGENT_TIMEOUT:
            return "The Agent run timed out. Please try again."
        return "The Agent run was cancelled."


class RunControl:
    """Thread-safe cancel token + queue-wait / agent-total deadlines.

    Abort (``cancel_event``) tells the Agent to stop at the next checkpoint.
    Stream close (``stream_closed_event``) tells the bridge to stop publishing.
    Agent timeout must abort without closing the stream so a terminal
    ``RunFailed`` can still be queued.
    """

    __slots__ = (
        "cancel_event",
        "stream_closed_event",
        "queue_wait_timeout_s",
        "agent_timeout_s",
        "_pending_at",
        "_started_at",
        "_abort_kind",
        "_lock",
    )

    def __init__(
        self,
        *,
        queue_wait_timeout_s: float,
        agent_timeout_s: float,
        cancel_event: threading.Event | None = None,
        stream_closed_event: threading.Event | None = None,
    ):
        if queue_wait_timeout_s <= 0:
            raise ValueError("queue_wait_timeout_s must be > 0")
        if agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s must be > 0")
        self.cancel_event = cancel_event or threading.Event()
        self.stream_closed_event = stream_closed_event or threading.Event()
        self.queue_wait_timeout_s = queue_wait_timeout_s
        self.agent_timeout_s = agent_timeout_s
        self._pending_at: float | None = None
        self._started_at: float | None = None
        self._abort_kind: AbortKind | None = None
        self._lock = threading.Lock()

    def mark_queued(self) -> None:
        with self._lock:
            if self._pending_at is None:
                self._pending_at = time.monotonic()

    def mark_started(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()

    def cancel(self, kind: AbortKind = AbortKind.CANCELLED) -> None:
        with self._lock:
            if self._abort_kind is None:
                self._abort_kind = kind
        self.cancel_event.set()

    def close_stream(self) -> None:
        """Stop bridge publishes (disconnect / consumer gone). Does not imply abort alone."""
        self.stream_closed_event.set()

    def is_stream_closed(self) -> bool:
        return self.stream_closed_event.is_set()

    def is_aborted(self) -> bool:
        return self.peek_abort() is not None

    def peek_abort(self) -> AbortKind | None:
        """Return the abort kind if the run should stop, without raising."""
        with self._lock:
            if self._abort_kind is not None:
                return self._abort_kind
            now = time.monotonic()
            if self._started_at is None and self._pending_at is not None:
                if now - self._pending_at >= self.queue_wait_timeout_s:
                    return AbortKind.QUEUE_WAIT_TIMEOUT
            if self._started_at is not None:
                if now - self._started_at >= self.agent_timeout_s:
                    return AbortKind.AGENT_TIMEOUT
            if self.cancel_event.is_set():
                return AbortKind.CANCELLED
            return None

    def raise_if_aborted(self) -> None:
        """Cooperative checkpoint for Agent node boundaries."""
        kind = self.peek_abort()
        if kind is None:
            return
        self.cancel(kind)
        raise RunAborted(kind)
