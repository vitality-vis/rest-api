"""Shared state for the literature-review (LR) human-in-the-loop demo graph.

This is a self-contained demo state object. It is intentionally decoupled from
the existing chat/RAG agents so it can be replaced later with real
retrieval/analysis/writer agents without touching this module's consumers.
"""
from typing import Any, Dict, List, Optional, TypedDict


class LRState(TypedDict, total=False):
    """State threaded through the LR demo graph.

    ``total=False`` so the graph can start from a partial state (e.g. only
    ``user_goal``) and let each node fill in its own keys.

    NOTE: typing here uses Python 3.9-compatible forms (``Optional[...]``,
    ``List[...]``, ``Dict[...]``) rather than 3.10+ ``X | None`` / ``list[...]``
    syntax, because LangGraph resolves these hints via ``get_type_hints`` at
    runtime and ``str | None`` raises ``TypeError`` on Python 3.9.
    """

    # The literature-review goal provided by the user.
    user_goal: str
    # High-level plan: a list of {"stage": int, "title": str, "description": str}.
    plan: List[Dict[str, Any]]
    # Free-text human feedback captured at the plan-approval interrupt.
    human_plan_feedback: Optional[str]
    # Placeholder retrieved papers: list of {ID, Title, Authors, Year, Abstract}.
    retrieved_papers: List[Dict[str, Any]]
    # Free-text human feedback captured at the retrieval-review interrupt.
    human_retrieval_feedback: Optional[str]
    # Structured analysis derived from the retrieved papers.
    analysis_result: Dict[str, Any]
    # The final short literature-review draft.
    draft: str
