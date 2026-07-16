"""
Planner agent — produces a multi-step tool execution plan via LLM reasoning.

Role in the multi-agent pipeline
---------------------------------
Orchestrator  ──► [/root/planner mailbox]  {user_request, context, reply_to}
                       │
                  PlannerAgent.run(input_data)
                       │  LLM call: reads live tool catalog from registry
                       │           → reasons about tools → JSON plan
                       ▼
                  self.send("/root", {type="execution_plan", steps=[...], errors=[...]})

The model plans using tool knowledge it reads at runtime from ToolRegistry,
not from hardcoded prompt text.  Adding a new tool requires only registering its
ToolDescriptor — zero changes here.

Aligned with Codex's planner pattern: the model decides what tools to call and
in what order based on tool capabilities, not because it was told step-by-step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import HumanMessage

from ..base_agent import BaseAgent
from ..status import AgentStatus
from service.tools.tool_registry import get_registry

if TYPE_CHECKING:
    from ..registry import AgentRegistry

logger = logging.getLogger(__name__)


# ── Plan data structures ───────────────────────────────────────────────────────

@dataclass
class PlanStep:
    tool:    str
    args:    Dict[str, Any]
    purpose: str


@dataclass
class ExecutionPlan:
    steps:  List[PlanStep] = field(default_factory=list)
    errors: List[str]      = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def is_valid(self) -> bool:
        return not self.errors and not self.is_empty()


# ── Planner prompt ─────────────────────────────────────────────────────────────
# The tool catalog section is injected at runtime from get_registry().
# Each tool's Constraints are rendered in the catalog; the prompt instructs
# the model to respect them, so no constraint logic lives here.

_PLAN_PROMPT = """\
You are a planning agent for an academic research assistant.
Given a user request and available context, produce the minimal ordered sequence
of tool calls that fulfills the request.

AVAILABLE TOOLS (read carefully — each tool lists Constraints you MUST respect):
{tool_catalog}

CONTEXT (selected paper details or session state — may be "None"):
{context}

USER REQUEST:
{user_request}

Planning rules:
1. Use only the tools listed above.
2. Produce only as many steps as strictly needed.
3. A step's string args may reference output from a prior step:
     {{{{step_N.title}}}}    — title of the first paper returned by step N
     {{{{step_N.abstract}}}} — abstract of the first paper returned by step N
     {{{{step_N.text}}}}     — full formatted text output of step N
4. If the context already contains a paper's title and abstract, skip any
   fetch step and reference that content directly in the first step's args.
5. Do NOT include chat_id in any step's args — it is injected by the executor.
6. You MUST respect every Constraint listed under each tool.

Before producing the plan, reason step by step:
  Thought: What is the user actually asking for?
  Thought: Which tool handles this? Are there Constraints I must check?
  Thought: Does this require multiple steps? What depends on what?
  Thought: What is the minimal sequence that fulfills the request?

Respond ONLY with valid JSON (no markdown fences):
{{
  "reasoning": "<your step-by-step thinking above>",
  "steps": [
    {{"tool": "<name>", "args": {{...}}, "purpose": "<one sentence>"}},
    ...
  ]
}}\
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    LLM-based planner: reads the live tool catalog from ToolRegistry, lets the
    model reason over it to produce a JSON execution plan, validates the plan,
    and sends the result back to the orchestrator via the registry mailbox.

    On planning failure the agent sends back an execution_plan with an empty
    steps list and a non-empty errors list — the orchestrator can fall back to
    the standard ReAct agent instead of silently doing nothing.
    """

    def __init__(self, chat_id: str, registry: "AgentRegistry", llm: Any) -> None:
        super().__init__(chat_id, registry)
        self.llm = llm

    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:
        self.set_status(AgentStatus.RUNNING)

        user_request   = str(input_data.get("user_request", ""))
        context        = str(input_data.get("context", "None"))
        reply_to       = str(input_data.get("reply_to", ""))
        catalog_filter: Optional[List[str]] = input_data.get("catalog_filter")  # None = all tools

        plan = await self._plan(user_request, context, catalog_filter)

        if reply_to:
            self.send(
                recipient=reply_to,
                content={
                    "type":   "execution_plan",
                    "steps":  [
                        {"tool": s.tool, "args": s.args, "purpose": s.purpose}
                        for s in plan.steps
                    ],
                    "errors": plan.errors,
                },
                trigger_turn=True,
            )

        self.set_status(AgentStatus.COMPLETED)
        return
        yield  # satisfy AsyncGenerator type hint

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _plan(
        self,
        user_request: str,
        context: str,
        catalog_filter: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        tool_catalog = get_registry().get_catalog_text(tools=catalog_filter)

        if not tool_catalog.strip():
            err = "ToolRegistry is empty — no tools registered; cannot plan."
            logger.error("[PlannerAgent] %s", err)
            return ExecutionPlan(errors=[err])

        prompt = _PLAN_PROMPT.format(
            tool_catalog=tool_catalog,
            context=(context[:2000] if context and context != "None" else "None"),
            user_request=user_request,
        )

        try:
            raw   = self.llm.invoke([HumanMessage(content=prompt)]).content
            logger.info("[PlannerAgent] raw LLM response: %s", raw[:500])
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
        except json.JSONDecodeError as exc:
            err = f"Planner LLM returned invalid JSON: {exc}"
            logger.warning("[PlannerAgent] %s — raw was: %s", err, raw[:300])
            return ExecutionPlan(errors=[err])
        except Exception as exc:
            err = f"Planner LLM call failed: {exc}"
            logger.warning("[PlannerAgent] %s", err)
            return ExecutionPlan(errors=[err])

        reasoning = data.get("reasoning", "")
        if reasoning:
            logger.info("[PlannerAgent] reasoning: %s", reasoning[:300])

        raw_steps = data.get("steps") or []
        steps = [
            PlanStep(
                tool    = str(s.get("tool", "")),
                args    = s.get("args") or {},
                purpose = str(s.get("purpose", "")),
            )
            for s in raw_steps
            if s.get("tool")
        ]

        # Validate against the live registry
        validation_errors = get_registry().validate_plan(
            [{"tool": s.tool, "args": s.args} for s in steps]
        )
        if validation_errors:
            logger.warning(
                "[PlannerAgent] plan validation failed: %s", validation_errors
            )
            return ExecutionPlan(errors=validation_errors)

        logger.info(
            "[PlannerAgent] chat=%s plan=[%s]",
            self.chat_id,
            ", ".join(f"{s.tool}({list(s.args)})" for s in steps),
        )
        return ExecutionPlan(steps=steps)
