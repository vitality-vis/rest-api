"""
OrchestratorAction: typed action model for the explicit orchestrator loop.

The orchestrator's decide_next_action() returns one of these; execute_action()
dispatches to the matching handler; update_after_action() mutates WorkflowState.

Why a separate action model instead of if/elif chains?
  - Makes the control flow readable as a sequence of typed steps
  - Enables logging ("action=EXECUTE_TOOL tool=semantic_search")
  - Future-ready: actions can be serialized for replay / audit trails
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrchestratorActionType(Enum):
    FAST_REPLY            = "fast_reply"
    REWRITE_QUERY         = "rewrite_query"
    CLASSIFY_INTENT       = "classify_intent"
    QUERY_GATEWAY         = "query_gateway"
    PLAN_RESEARCH         = "plan_research"           # Planner → ResearchPlan
    EXECUTE_RESEARCH_PLAN = "execute_research_plan"   # TaskExecutor → agents
    EXECUTE_TOOL          = "execute_tool"
    ASSESS_RETRIEVAL      = "assess_retrieval"
    REFINE_QUERY          = "refine_query"
    GENERATE_FINAL_ANSWER = "generate_final_answer"
    FINALIZE              = "finalize"
    ERROR_FALLBACK        = "error_fallback"


@dataclass
class OrchestratorAction:
    type:      OrchestratorActionType
    tool_name: Optional[str] = None
    tool_args: dict           = field(default_factory=dict)
    reason:    str            = ""
