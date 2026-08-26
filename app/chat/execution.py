"""Dedicated full-profile executor for Chat Agent jobs."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
DEFAULT_AGENT_MAX_CONCURRENT = 2


class AgentExecutionRuntime:
    """Own the thread pool used exclusively by Agent runs."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vitality-agent",
        )
        self._closed = False

    def submit(self, job: Callable[[], T]) -> Future[T]:
        if self._closed:
            raise RuntimeError("Agent executor is shutting down")
        return self._executor.submit(job)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


def create_agent_runtime() -> AgentExecutionRuntime:
    raw = os.environ.get("AGENT_MAX_CONCURRENT", str(DEFAULT_AGENT_MAX_CONCURRENT))
    try:
        max_workers = int(raw)
    except ValueError as error:
        raise ValueError("AGENT_MAX_CONCURRENT must be an integer") from error
    if max_workers < 1:
        raise ValueError("AGENT_MAX_CONCURRENT must be at least 1")
    return AgentExecutionRuntime(max_workers=max_workers)
