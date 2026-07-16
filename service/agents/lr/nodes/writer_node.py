"""Writer node: compose a short LR draft from the goal and analysis.

Placeholder logic only — deterministic string assembly. Replace with a real
writer agent later; keep the returned ``draft`` key.
"""
from __future__ import annotations

from service.agents.lr.lr_state import LRState


def writer_node(state: LRState) -> dict:
    """Produce a short literature-review draft paragraph."""
    user_goal = state.get("user_goal", "the requested topic")
    analysis = state.get("analysis_result", {}) or {}

    themes = analysis.get("key_themes", [])
    methods = analysis.get("common_methods", [])
    limitations = analysis.get("limitations", [])
    gap = analysis.get("possible_gap", "")

    themes_str = "; ".join(themes) if themes else "several recurring themes"
    methods_str = ", ".join(methods) if methods else "a range of methods"
    limitations_str = "; ".join(limitations) if limitations else "various limitations"

    draft = (
        f"This literature review addresses the goal: \"{user_goal}\" "
        f"The reviewed work converges on {themes_str}. "
        f"Methodologically, the literature commonly relies on {methods_str}. "
        f"Despite this progress, recurring limitations remain, including {limitations_str}. "
        f"Taken together, a notable research gap emerges: {gap} "
        f"Addressing this gap would strengthen agentic, human-in-the-loop approaches "
        f"to academic paper synthesis."
    )

    return {"draft": draft}
