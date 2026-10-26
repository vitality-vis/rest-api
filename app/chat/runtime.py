"""Select the Chat runner used by /chat/v2."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import config

ChatRunner = Callable[[Any], AsyncIterator[Any]]


def resolve_chat_runner() -> ChatRunner:
    """Return the configured Chat runner.

    ``CHAT_RUNTIME=v2`` keeps the existing deterministic runner. The default
    ``codex`` runtime uses the local Codex app-server.
    """
    if config.CHAT_RUNTIME == "v2":
        from agents.agent_v2.runner import run as run_agent_v2

        return run_agent_v2
    from app.codex.runner import run as run_codex

    return run_codex
