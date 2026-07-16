"""
SubAgentBase: task-oriented extension of BaseAgent.

Adds execute_task() as the canonical entry point that call_subagent() invokes.
The orchestrator always calls execute_task() — it never calls run_from_mailbox()
directly — so the dispatch strategy is fully encapsulated here.

Current dispatch (synchronous)
-------------------------------
  1. Call self.run(task.payload) directly with the task payload as input_data
  2. Worker agents call self.send(reply_to, result) internally, routing the
     result to /root's mailbox
  3. Drain /root inbox and match the message whose task_id equals the task's
     task_id (guards against stale messages from earlier tasks)
  4. Return wrapped in SubAgentResult

Future upgrade path (override execute_task — zero orchestrator changes)
------------------------------------------------------------------------
  - asyncio.create_task(self.run(task.payload)) with task_id
  - Await a completion event from a status-watch channel
  - Enforce task.timeout via asyncio.wait_for
  - Return SubAgentResult from a shared status store
  - Support remote dispatch (serialize task → queue → deserialize result)
"""
from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, AsyncGenerator

from .base_agent import BaseAgent
from .status import AgentStatus
from .sub_agent_task import SubAgentTask, SubAgentResult


class SubAgentBase(BaseAgent):
    """
    Task-oriented base for every subagent the orchestrator dispatches work to.

    Subclasses
    ----------
    Must implement: run(input_data) — the core agent logic (from BaseAgent)
    May override:   execute_task(task) — to change the dispatch strategy

    The task.payload passed to execute_task() must include:
      "reply_to": AgentRegistry.ROOT
    so the agent's run() can route its result back to the orchestrator inbox.
    """

    @abstractmethod
    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Core agent logic. Worker agents yield nothing; results go via self.send()."""
        ...
        yield  # satisfy AsyncGenerator typing

    async def execute_task(self, task: SubAgentTask) -> SubAgentResult:
        """
        Execute a SubAgentTask and return a SubAgentResult.

        Default synchronous implementation:
          1. Call self.run(task.payload) — worker agents yield no user-visible text
          2. Drain /root inbox, matching the message by task_id
          3. Return wrapped in SubAgentResult

        Override to change the dispatch strategy without touching the orchestrator.
        """
        from .registry import AgentRegistry

        start = time.monotonic()
        try:
            self.set_status(AgentStatus.RUNNING)

            # Run the agent directly with the task payload as input_data.
            # Worker agents (e.g. AssessorAgent) call self.send(reply_to, result)
            # internally, routing the result to the /root mailbox.
            async for _ in self.run(task.payload):
                pass  # worker agents produce no user-visible output

            # Drain /root inbox and match by task_id so stale messages from
            # prior tasks (or concurrent future tasks) are not accidentally consumed.
            result_msg: dict = {}
            for msg in self.registry.drain_inbox(AgentRegistry.ROOT):
                if msg.content.get("task_id") == task.task_id or not result_msg:
                    result_msg = msg.content
                    if msg.content.get("task_id") == task.task_id:
                        break  # exact match found; stop looking

            self.set_status(AgentStatus.COMPLETED)
            return SubAgentResult(
                task_id=task.task_id,
                agent_path=self.path,
                output=result_msg,
                status=AgentStatus.COMPLETED,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )

        except Exception as exc:
            self.set_status(AgentStatus.ERRORED)
            return SubAgentResult(
                task_id=task.task_id,
                agent_path=self.path,
                output={},
                status=AgentStatus.ERRORED,
                duration_ms=(time.monotonic() - start) * 1000.0,
                error=str(exc),
            )
