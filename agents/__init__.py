"""Agent entry points for chat / paper-search orchestration."""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def agent(agent_id: str) -> Callable[[F], F]:
    """Attach a stable agent id to an entrypoint (e.g. for future registry lookup)."""

    def decorator(fn: F) -> F:
        fn.agent_id = agent_id  # type: ignore[attr-defined]
        return fn

    return decorator
