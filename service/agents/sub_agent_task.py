"""
SubAgentTask / SubAgentResult: typed task contracts for subagent dispatch.

Designed so that call_subagent() can evolve without changing its callers:

  Current:  synchronous execution (SubAgentBase.execute_task runs inline)

  Future upgrade path — zero changes to orchestrator callers:
    - task_id enables status lookup by ID after async dispatch
    - timeout is enforced by the async dispatcher (asyncio.wait_for)
    - SubAgentResult is the async completion payload returned by the watcher
    - Dispatcher upgrades: asyncio.create_task → distributed task queue
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .status import AgentStatus


def _new_task_id() -> str:
    return str(uuid.uuid4())


@dataclass
class SubAgentTask:
    """
    Task descriptor passed to call_subagent().

    Fields
    ------
    agent_path : Registry path of the target agent, e.g. "/root/assessor"
    payload    : Dict delivered to the agent's run() as input_data.
                 Must include "reply_to": AgentRegistry.ROOT so the agent
                 routes its result back to the orchestrator's inbox.
    task_id    : Stable UUID for async tracking (auto-generated if omitted)
    created_at : UTC timestamp for latency accounting
    timeout    : Seconds; future async dispatcher enforces this via wait_for
    """

    agent_path: str
    payload: dict
    task_id: str = field(default_factory=_new_task_id)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    timeout: Optional[float] = None


@dataclass
class SubAgentResult:
    """
    Result envelope returned by call_subagent().

    Fields
    ------
    task_id     : Matches the SubAgentTask.task_id that produced this result
    agent_path  : Path of the agent that executed the task
    output      : Result payload (type-tagged with "type" key by convention)
    status      : Terminal AgentStatus — COMPLETED or ERRORED
    duration_ms : Wall-clock execution time in milliseconds
    error       : Human-readable message when status == ERRORED
    """

    task_id: str
    agent_path: str
    output: dict
    status: AgentStatus
    duration_ms: float
    error: Optional[str] = None

    def is_ok(self) -> bool:
        """True when the task completed without error."""
        return self.status == AgentStatus.COMPLETED and self.error is None
