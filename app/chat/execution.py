"""Dedicated full-profile executor for Chat Agent jobs with admission control."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

import config

T = TypeVar("T")

DEFAULT_AGENT_MAX_CONCURRENT = 2
DEFAULT_AGENT_MAX_PENDING = 8


class RunTerminal(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timedOut"


class _RunPhase(str, Enum):
    """Executor occupancy. Independent of semantic Chat outcome."""

    RESERVED = "reserved"
    PENDING = "pending"
    ACTIVE = "active"
    RELEASED = "released"


@dataclass(frozen=True)
class AgentRuntimeSnapshot:
    """Wire-safe runtime view for /health (no user/run identifiers)."""

    ready: bool
    accepting: bool
    capacity: int
    pending_capacity: int
    active: int
    pending: int
    reserved: int
    completed: int
    failed: int
    cancelled: int
    rejected: int
    timed_out: int
    queue_wait_ms_total: int
    execution_ms_total: int

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "accepting": self.accepting,
            "capacity": self.capacity,
            "pendingCapacity": self.pending_capacity,
            "active": self.active,
            "pending": self.pending,
            "reserved": self.reserved,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "rejected": self.rejected,
            "timedOut": self.timed_out,
            "queueWaitMsTotal": self.queue_wait_ms_total,
            "executionMsTotal": self.execution_ms_total,
        }


class AgentRunReservation:
    """One admitted Chat run.

    Occupancy (reserved/pending/active/released) tracks real executor capacity.
    Semantic outcome (completed/failed/cancelled/timedOut) may be recorded while
    the worker thread is still running after an SSE timeout.
    """

    __slots__ = (
        "_runtime",
        "_phase",
        "_reserved_at",
        "_pending_at",
        "_started_at",
        "_future",
        "_terminal",
    )

    def __init__(self, runtime: AgentExecutionRuntime):
        self._runtime = runtime
        self._phase = _RunPhase.RESERVED
        self._reserved_at = time.monotonic()
        self._pending_at: float | None = None
        self._started_at: float | None = None
        self._future: Future | None = None
        self._terminal: RunTerminal | None = None

    @property
    def future(self) -> Future | None:
        return self._future

    @property
    def is_terminal(self) -> bool:
        """True when a semantic outcome has been recorded (occupancy may remain)."""
        return self._terminal is not None

    @property
    def is_released(self) -> bool:
        return self._phase is _RunPhase.RELEASED

    def release(self) -> None:
        """Drop an unused reservation (validation/auth failure before submit)."""
        self._runtime._release_unused(self)

    def mark_cancelled(self) -> None:
        """Client disconnect / shutdown before or during the run."""
        self._runtime._record_outcome(self, RunTerminal.CANCELLED)

    def mark_completed(self) -> None:
        self._runtime._record_outcome(self, RunTerminal.COMPLETED)

    def mark_failed(self) -> None:
        self._runtime._record_outcome(self, RunTerminal.FAILED)

    def mark_timed_out(self) -> None:
        self._runtime._record_outcome(self, RunTerminal.TIMED_OUT)

    def ensure_closed(self, *, terminal: RunTerminal = RunTerminal.CANCELLED) -> None:
        """Idempotent semantic + occupancy cleanup when the stream ends."""
        if self._terminal is None:
            self._runtime._record_outcome(self, terminal)
        if self._phase is _RunPhase.RESERVED and self._future is None:
            self._runtime._release_occupancy(self)
            return
        if self._future is not None and self._future.done():
            self._runtime._release_occupancy(self)


class AgentExecutionRuntime:
    """Own the Agent thread pool and explicit admission (not the executor's unbounded queue)."""

    def __init__(
        self,
        max_workers: int,
        max_pending: int,
        *,
        sse_queue_size: int = config.AGENT_SSE_QUEUE_SIZE,
        sse_keepalive_seconds: float = config.AGENT_SSE_KEEPALIVE_SECONDS,
        queue_wait_timeout_seconds: float = config.AGENT_QUEUE_WAIT_TIMEOUT_SECONDS,
        agent_run_timeout_seconds: float = config.AGENT_RUN_TIMEOUT_SECONDS,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_pending < 0:
            raise ValueError("max_pending must be >= 0")
        if sse_queue_size < 1:
            raise ValueError("sse_queue_size must be at least 1")
        if sse_keepalive_seconds <= 0:
            raise ValueError("sse_keepalive_seconds must be > 0")
        if queue_wait_timeout_seconds <= 0:
            raise ValueError("queue_wait_timeout_seconds must be > 0")
        if agent_run_timeout_seconds <= 0:
            raise ValueError("agent_run_timeout_seconds must be > 0")

        self.max_workers = max_workers
        self.max_pending = max_pending
        self.max_admitted = max_workers + max_pending
        self.sse_queue_size = sse_queue_size
        self.sse_keepalive_seconds = sse_keepalive_seconds
        self.queue_wait_timeout_seconds = queue_wait_timeout_seconds
        self.agent_run_timeout_seconds = agent_run_timeout_seconds

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vitality-agent",
        )
        self._lock = threading.RLock()
        self._closed = False

        self._reserved = 0
        self._pending = 0
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._timed_out = 0
        self._queue_wait_ms_total = 0
        self._execution_ms_total = 0

    @property
    def admitted(self) -> int:
        with self._lock:
            return self._reserved + self._pending + self._active

    def try_reserve(self) -> AgentRunReservation | None:
        """Admit one run before SSE starts. None means overload → HTTP 503."""
        with self._lock:
            if self._closed:
                self._rejected += 1
                return None
            if self._reserved + self._pending + self._active >= self.max_admitted:
                self._rejected += 1
                return None
            self._reserved += 1
            return AgentRunReservation(self)

    def submit(
        self,
        reservation: AgentRunReservation,
        job: Callable[[], T],
    ) -> Future[T]:
        """Move reserved → pending and enqueue work on the dedicated executor."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent executor is shutting down")
            if reservation._runtime is not self:
                raise ValueError("Reservation belongs to a different runtime")
            if reservation._phase is not _RunPhase.RESERVED:
                raise RuntimeError(
                    f"Cannot submit reservation in phase {reservation._phase.value}"
                )
            self._reserved -= 1
            self._pending += 1
            reservation._phase = _RunPhase.PENDING
            reservation._pending_at = time.monotonic()

            def run_tracked() -> T:
                self._mark_active(reservation)
                return job()

            future = self._executor.submit(run_tracked)
            reservation._future = future

            def _on_done(_fut: Future[T]) -> None:
                self._release_occupancy(reservation)

            future.add_done_callback(_on_done)
            return future

    def snapshot(self) -> AgentRuntimeSnapshot:
        with self._lock:
            closed = self._closed
            admitted = self._reserved + self._pending + self._active
            return AgentRuntimeSnapshot(
                ready=not closed,
                accepting=not closed and admitted < self.max_admitted,
                capacity=self.max_workers,
                pending_capacity=self.max_pending,
                active=self._active,
                pending=self._pending,
                reserved=self._reserved,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
                rejected=self._rejected,
                timed_out=self._timed_out,
                queue_wait_ms_total=self._queue_wait_ms_total,
                execution_ms_total=self._execution_ms_total,
            )

    async def shutdown(self) -> None:
        import asyncio

        with self._lock:
            if self._closed:
                return
            self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )

    def _release_unused(self, reservation: AgentRunReservation) -> None:
        with self._lock:
            if reservation._phase is _RunPhase.RELEASED:
                return
            if reservation._phase is not _RunPhase.RESERVED:
                raise RuntimeError(
                    "Only unused reserved runs can be release()'d; "
                    f"phase={reservation._phase.value}"
                )
            self._reserved -= 1
            reservation._phase = _RunPhase.RELEASED
            # Validation failure is not a Chat outcome; do not bump cancelled.

    def _mark_active(self, reservation: AgentRunReservation) -> None:
        with self._lock:
            if reservation._phase is _RunPhase.RELEASED:
                return
            if reservation._phase is not _RunPhase.PENDING:
                return
            now = time.monotonic()
            if reservation._pending_at is not None:
                wait_ms = max(0, round((now - reservation._pending_at) * 1000))
                self._queue_wait_ms_total += wait_ms
            self._pending -= 1
            self._active += 1
            reservation._phase = _RunPhase.ACTIVE
            reservation._started_at = now

    def _record_outcome(
        self,
        reservation: AgentRunReservation,
        terminal: RunTerminal,
    ) -> None:
        """Record semantic Chat outcome without freeing executor occupancy."""
        with self._lock:
            if reservation._terminal is not None:
                return
            reservation._terminal = terminal
            if terminal is RunTerminal.COMPLETED:
                self._completed += 1
            elif terminal is RunTerminal.FAILED:
                self._failed += 1
            elif terminal is RunTerminal.CANCELLED:
                self._cancelled += 1
            elif terminal is RunTerminal.TIMED_OUT:
                self._timed_out += 1

            # Rejected-before-submit / unused reserved: no Future will release us.
            if (
                reservation._phase is _RunPhase.RESERVED
                and reservation._future is None
            ):
                self._reserved -= 1
                reservation._phase = _RunPhase.RELEASED

    def _release_occupancy(self, reservation: AgentRunReservation) -> None:
        """Free reserved/pending/active capacity once the Future is done or unused."""
        with self._lock:
            if reservation._phase is _RunPhase.RELEASED:
                return
            now = time.monotonic()
            if reservation._phase is _RunPhase.RESERVED:
                self._reserved -= 1
            elif reservation._phase is _RunPhase.PENDING:
                self._pending -= 1
                if reservation._pending_at is not None:
                    wait_ms = max(0, round((now - reservation._pending_at) * 1000))
                    self._queue_wait_ms_total += wait_ms
            elif reservation._phase is _RunPhase.ACTIVE:
                self._active -= 1
                if reservation._started_at is not None:
                    exec_ms = max(0, round((now - reservation._started_at) * 1000))
                    self._execution_ms_total += exec_ms
            reservation._phase = _RunPhase.RELEASED


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def create_agent_runtime() -> AgentExecutionRuntime:
    max_workers = _env_int("AGENT_MAX_CONCURRENT", DEFAULT_AGENT_MAX_CONCURRENT)
    max_pending = _env_int("AGENT_MAX_PENDING", DEFAULT_AGENT_MAX_PENDING)
    if max_workers < 1:
        raise ValueError("AGENT_MAX_CONCURRENT must be at least 1")
    if max_pending < 0:
        raise ValueError("AGENT_MAX_PENDING must be >= 0")
    return AgentExecutionRuntime(
        max_workers=max_workers,
        max_pending=max_pending,
        sse_queue_size=config.AGENT_SSE_QUEUE_SIZE,
        sse_keepalive_seconds=config.AGENT_SSE_KEEPALIVE_SECONDS,
        queue_wait_timeout_seconds=config.AGENT_QUEUE_WAIT_TIMEOUT_SECONDS,
        agent_run_timeout_seconds=config.AGENT_RUN_TIMEOUT_SECONDS,
    )
