"""
CollectPapersAgent — the only specialized research agent wired today.

Role
----
Handles the COLLECT_PAPERS research task. It is the single agent that performs
retrieval, and it does so by delegating to the EXISTING ToolExecutor with the
EXISTING search tools — no retrieval logic is reimplemented or modified here.
This is what preserves search quality: collection runs the same tool the
deterministic pipeline would have run.

"Specialized agents internally decide which existing tools to use": this agent
picks the tool at runtime via the shared select_search_tool() helper (the same
heuristic the orchestrator uses), then calls ToolExecutor.execute().

Dispatch
--------
Dispatched by the orchestrator/TaskExecutor via call_subagent(); replies to
"/root" with a `collect_papers_result` message, echoing task_id for matching.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, TYPE_CHECKING

from ..status import AgentStatus
from ..sub_agent_base import SubAgentBase
from .search_tool_selection import select_search_tool

if TYPE_CHECKING:
    from ..registry import AgentRegistry

logger = logging.getLogger(__name__)


class CollectPapersAgent(SubAgentBase):
    """Retrieves papers for a COLLECT_PAPERS task using existing tools."""

    def __init__(self, chat_id: str, registry: "AgentRegistry", tool_executor: Any) -> None:
        super().__init__(chat_id, registry)
        self._tool_executor = tool_executor

    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """
        Expected input_data keys
        ------------------------
        query       : str  — search topic / question
        intent      : IntentResult or None — used to pick the right search tool
        clean_input : str  — original cleaned query (RAG_QA question fallback)
        reply_to    : str  — agent path to route the result to (typically "/root")
        task_id     : str  — echoed back so the dispatcher can match the result
        """
        self.set_status(AgentStatus.RUNNING)

        query       = str(input_data.get("query", ""))
        intent      = input_data.get("intent")
        clean_input = str(input_data.get("clean_input", "") or query)
        reply_to    = str(input_data.get("reply_to", ""))
        task_id     = input_data.get("task_id")

        tool_name, tool_args = select_search_tool(intent, query, clean_input)
        logger.info("[CollectPapersAgent] task_id=%s tool=%s", task_id, tool_name)

        content: dict
        if self._tool_executor is None:
            content = {
                "type": "collect_papers_result", "task_id": task_id,
                "tool_name": tool_name, "output": "", "docs": [],
                "status": "error", "error": "tool_executor unavailable",
            }
        else:
            result = self._tool_executor.execute(tool_name, tool_args)
            content = {
                "type": "collect_papers_result",
                "task_id": task_id,
                "tool_name": tool_name,
                "output": result.output if result.is_ok() else "",
                "docs": result.docs,
                "status": result.status,
                "error": result.error,
            }

        if reply_to:
            self.send(recipient=reply_to, content=content, trigger_turn=True)

        self.set_status(AgentStatus.COMPLETED)
        return
        yield  # satisfy AsyncGenerator typing
