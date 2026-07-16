"""
ResearchPlannerAgent — produces a ResearchTask plan (NOT a tool plan).

This is the task-level planner. It reads the live TaskCatalog (which currently
only contains `collect_papers`), lets the LLM choose an ordered list of tasks,
validates them against the catalog, and replies to the orchestrator.

Contrast with the existing PlannerAgent (left untouched): that one emits
{"tool": ...} steps validated against ToolRegistry. This one emits
{"task": ...} steps validated against TaskCatalog — the planner literally cannot
name a tool because tools are not in its prompt.

On failure it returns an empty plan + errors so the orchestrator falls back to
the deterministic intent→tool path (fail-open, never silent).

Dispatched via call_subagent(); replies to "/root" with a `research_plan`
message echoing task_id.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage

from ..status import AgentStatus
from ..sub_agent_base import SubAgentBase
from .task_catalog import get_task_catalog

if TYPE_CHECKING:
    from ..registry import AgentRegistry

logger = logging.getLogger(__name__)


_PLAN_PROMPT = """\
You are a planning agent for an academic research assistant.
Given a user request, produce the minimal ordered list of RESEARCH TASKS that
fulfills it. You plan TASKS, never tools — do NOT output tool names.

AVAILABLE TASKS (use only these task names):
{task_catalog}

USER REQUEST:
{user_request}

Planning rules:
1. Use only the task names listed above.
2. Produce only as many tasks as strictly needed (often just one).
3. Almost every information request begins with a "collect_papers" task.
4. Put task inputs in "params" (e.g. {{"topic": "..."}}). Never put tool names anywhere.

Respond ONLY with valid JSON (no markdown fences):
{{
  "reasoning": "<brief step-by-step thinking>",
  "tasks": [
    {{"task": "<task_name>", "params": {{...}}}}
  ]
}}\
"""


class ResearchPlannerAgent(SubAgentBase):
    """LLM planner that emits a validated ResearchTask plan."""

    def __init__(self, chat_id: str, registry: "AgentRegistry", llm: Any) -> None:
        super().__init__(chat_id, registry)
        self.llm = llm

    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:  # type: ignore[override]
        self.set_status(AgentStatus.RUNNING)

        user_request = str(input_data.get("user_request", ""))
        reply_to     = str(input_data.get("reply_to", ""))
        task_id      = input_data.get("task_id")

        tasks, errors = self._plan(user_request)

        if reply_to:
            self.send(
                recipient=reply_to,
                content={
                    "type":    "research_plan",
                    "task_id": task_id,
                    "tasks":   tasks,    # list of {"task","params"}
                    "errors":  errors,
                },
                trigger_turn=True,
            )

        self.set_status(AgentStatus.COMPLETED)
        return
        yield  # satisfy AsyncGenerator typing

    # ── Internal ────────────────────────────────────────────────────────────

    def _plan(self, user_request: str):
        catalog = get_task_catalog()
        catalog_text = catalog.get_catalog_text()
        if not catalog_text.strip():
            return [], ["TaskCatalog is empty — no tasks registered; cannot plan."]

        prompt = _PLAN_PROMPT.format(
            task_catalog=catalog_text,
            user_request=user_request,
        )

        raw = ""
        try:
            raw   = self.llm.invoke([HumanMessage(content=prompt)]).content
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.warning("[ResearchPlannerAgent] invalid JSON: %s — raw=%s", exc, raw[:300])
            return [], [f"Planner returned invalid JSON: {exc}"]
        except Exception as exc:
            logger.warning("[ResearchPlannerAgent] LLM call failed: %s", exc)
            return [], [f"Planner LLM call failed: {exc}"]

        reasoning = data.get("reasoning", "")
        if reasoning:
            logger.info("[ResearchPlannerAgent] reasoning: %s", str(reasoning)[:300])

        raw_tasks: List[dict] = [
            {"task": str(t.get("task", "")).strip().lower(), "params": t.get("params") or {}}
            for t in (data.get("tasks") or [])
            if t.get("task")
        ]

        errors = catalog.validate_plan(raw_tasks)
        if errors:
            logger.warning("[ResearchPlannerAgent] plan validation failed: %s", errors)
            return [], errors

        logger.info(
            "[ResearchPlannerAgent] chat=%s plan=[%s]",
            self.chat_id, ", ".join(t["task"] for t in raw_tasks),
        )
        return raw_tasks, []
