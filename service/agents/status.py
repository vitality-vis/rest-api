"""
Agent lifecycle status.

Aligned with Codex codex-rs/core/src/agent/status.rs.

Transitions
-----------
PENDING_INIT  ──► RUNNING ──► COMPLETED
                         └──► INTERRUPTED
                         └──► ERRORED
                         └──► SHUTDOWN
"""
from __future__ import annotations

from enum import Enum


class AgentStatus(str, Enum):
    PENDING_INIT = "pending_init"   # spawned, not yet started
    RUNNING      = "running"        # actively processing a turn
    COMPLETED    = "completed"      # finished successfully
    INTERRUPTED  = "interrupted"    # cancelled mid-turn (can be resumed)
    ERRORED      = "errored"        # unrecoverable error
    SHUTDOWN     = "shutdown"       # permanently stopped


def is_final(status: AgentStatus) -> bool:
    """True for terminal states — the agent will never run again."""
    return status not in (
        AgentStatus.PENDING_INIT,
        AgentStatus.RUNNING,
        AgentStatus.INTERRUPTED,
    )
