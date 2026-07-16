"""
TaskExecutor — the task-level analogue of ToolExecutor.

It does NOT run tools. It routes each ResearchTask to its specialized agent and
dispatches via the EXISTING call_subagent() substrate, threading earlier task
output forward as context for later tasks (sequential, dependency-aware).

Today only COLLECT_PAPERS is routed. Adding a task later = one more route entry
plus its registered agent. Returns structured results; the orchestrator owns
WorkflowState mutation (keeping state ownership in one place).
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List

from ..registry import AgentRegistry
from ..sub_agent_task import SubAgentTask, SubAgentResult
from .research_task import ResearchPlan, ResearchTaskType

logger = logging.getLogger(__name__)

# ResearchTaskType -> registered agent path. Only collect_papers is wired.
_ROUTES: Dict[ResearchTaskType, str] = {
    ResearchTaskType.COLLECT_PAPERS: AgentRegistry.TASK_COLLECT_PAPERS,
}


class TaskExecutor:
    """Dispatches a ResearchPlan to specialized agents via call_subagent()."""

    def __init__(
        self,
        call_subagent: Callable[[str, SubAgentTask], Awaitable[SubAgentResult]],
        session: dict,
    ) -> None:
        self._call_subagent = call_subagent
        self._session = session

    async def execute_plan(self, plan: ResearchPlan, state: Any) -> List[dict]:
        """
        Execute each task in order. Returns a list of normalized result dicts:
            {task, tool_name, output, docs, status, error}
        The caller applies these to WorkflowState.
        """
        results: List[dict] = []
        prior_context = ""

        for i, task in enumerate(plan.tasks, start=1):
            path = _ROUTES.get(task.type)
            if not path:
                logger.warning("[TaskExecutor] no agent for task %s", task.type)
                results.append({
                    "task": task.type.value, "tool_name": None, "output": "",
                    "docs": [], "status": "unrouted",
                    "error": f"no agent registered for task '{task.type.value}'",
                })
                continue

            query = (
                task.params.get("topic")
                or task.params.get("query")
                or getattr(state, "agent_input", "")
                or getattr(state, "clean_input", "")
            )
            sub = SubAgentTask(
                agent_path=path,
                payload={
                    "params":      task.params,
                    "context":     prior_context,
                    "query":       query,
                    "clean_input": getattr(state, "clean_input", ""),
                    "intent":      getattr(state, "intent", None),
                    "reply_to":    AgentRegistry.ROOT,
                },
            )
            # Echo task_id so the agent's reply can be matched on drain.
            sub.payload["task_id"] = sub.task_id

            logger.info("[TaskExecutor] step %d: %s -> %s", i, task.type.value, path)
            res = await self._call_subagent(path, sub)
            out = res.output if res.is_ok() else {}

            output = out.get("output", "") if isinstance(out, dict) else ""
            results.append({
                "task":      task.type.value,
                "tool_name": out.get("tool_name") if isinstance(out, dict) else None,
                "output":    output,
                "docs":      out.get("docs", []) if isinstance(out, dict) else [],
                "status":    out.get("status", "error") if isinstance(out, dict) else "error",
                "error":     res.error or (out.get("error") if isinstance(out, dict) else None),
            })
            if output:
                prior_context = output

        return results
