"""Plan node: propose a high-level LR plan, then pause for human approval.

Placeholder logic only — no LLM call. The node loops on ``interrupt()`` so the
human can iteratively *edit* the plan before approving:

    approve  -> proceed to retrieval
    edit     -> revise the plan with the feedback, then check in again
    (reject is handled by the session layer, which simply drops the thread)

The resume value supplied via ``Command(resume=...)`` is a dict shaped like
``{"action": <str>, "feedback": <str>}`` (see ``lr_session.py``).
"""
from typing import Any, Dict, List

from langgraph.types import interrupt

from service.agents.lr.lr_state import LRState

_BASE_PLAN: List[Dict[str, Any]] = [
    {
        "stage": 1,
        "title": "Retrieve relevant papers",
        "description": "Gather candidate papers relevant to the goal.",
    },
    {
        "stage": 2,
        "title": "Analyze methods and limitations",
        "description": "Extract methods, datasets, and limitations per paper.",
    },
    {
        "stage": 3,
        "title": "Synthesize themes and research gaps",
        "description": "Cluster findings into themes and identify open gaps.",
    },
    {
        "stage": 4,
        "title": "Draft literature review section",
        "description": "Write a coherent review section grounded in the analysis.",
    },
]


def _revise_plan(feedbacks: List[str]) -> List[Dict[str, Any]]:
    """Deterministically derive a plan from the base plan + ordered feedback.

    Determinism matters: LangGraph re-executes the node from the top on every
    resume, replaying each ``interrupt()`` in order, so the same feedback list
    must always yield the same plan.
    """
    plan = [dict(stage) for stage in _BASE_PLAN]
    for idx, fb in enumerate(feedbacks, start=1):
        plan.append(
            {
                "stage": len(_BASE_PLAN) + idx,
                "title": f"Revision {idx}: incorporate reviewer feedback",
                "description": fb,
            }
        )
    return plan


def plan_node(state: LRState) -> dict:
    """Generate a plan and request human approval/edits via interrupt()."""
    user_goal = state.get("user_goal", "")
    feedbacks: List[str] = []

    while True:
        plan = _revise_plan(feedbacks)
        decision = interrupt(
            {
                "checkpoint_type": "plan_review",
                "title": "Proposed literature-review plan",
                "message": "Review the proposed plan. Approve to continue, "
                "edit to revise it, or reject to cancel.",
                "user_goal": user_goal,
                "plan": plan,
            }
        )

        action, feedback = _read_decision(decision)

        if action in ("approve", "accept", "continue"):
            return {
                "plan": plan,
                "human_plan_feedback": "; ".join(feedbacks) or None,
            }

        # edit/revise: only loop if we actually got feedback to act on,
        # otherwise proceed to avoid getting stuck.
        if feedback:
            feedbacks.append(feedback)
        else:
            return {
                "plan": plan,
                "human_plan_feedback": "; ".join(feedbacks) or None,
            }


def _read_decision(decision: Any) -> tuple:
    """Normalize a resume value into ``(action, feedback)``."""
    if isinstance(decision, dict):
        action = str(decision.get("action") or "approve").strip().lower()
        feedback = str(decision.get("feedback") or "").strip()
        return action, feedback
    # Backward/defensive: a bare string resume is treated as approval.
    return "approve", str(decision or "").strip()
