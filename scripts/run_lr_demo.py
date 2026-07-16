"""Run the literature-review (LR) human-in-the-loop demo end to end.

Demonstrates the loop:
    plan -> human approval -> retrieval -> human review -> analysis -> writer

Usage (from the rest-api project root, with langgraph installed):
    python scripts/run_lr_demo.py
"""
from __future__ import annotations

import os
import sys

# Make the project root importable when run as `python scripts/run_lr_demo.py`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph.types import Command  # noqa: E402

from service.agents.lr.graph import build_lr_graph  # noqa: E402


def _print_interrupt(label: str, result: dict) -> None:
    """Pretty-print the interrupt payload returned by an invoke() call."""
    interrupts = result.get("__interrupt__")
    print(f"\n===== {label} =====")
    if not interrupts:
        print("(no interrupt — graph did not pause)")
        return
    for itr in interrupts:
        # langgraph Interrupt objects expose `.value`; fall back to repr.
        payload = getattr(itr, "value", itr)
        print(payload)


def main() -> None:
    graph = build_lr_graph()

    # A thread_id is required so the checkpointer can pause/resume this run.
    config = {"configurable": {"thread_id": "lr-demo-1"}}

    user_goal = (
        "Write a literature review section about agentic RAG for "
        "academic paper synthesis."
    )

    # 1) Start the graph — it runs `plan` and pauses at the plan-approval interrupt.
    result = graph.invoke({"user_goal": user_goal}, config=config)
    _print_interrupt("INTERRUPT 1 — Plan approval requested", result)

    # 2) Approve the plan. interrupt() in plan_node returns this resume value,
    #    then the graph proceeds to `retrieval` and pauses again.
    result = graph.invoke(
        Command(resume="Approved. Continue with retrieval."),
        config=config,
    )
    _print_interrupt("INTERRUPT 2 — Retrieval review requested", result)

    # 3) Accept the retrieved papers. The graph runs `analysis` then `writer`
    #    and reaches END, returning the final state.
    final_state = graph.invoke(
        Command(resume="Retrieved papers look good. Continue."),
        config=config,
    )

    print("\n===== FINAL DRAFT =====")
    print(final_state.get("draft", "(no draft produced)"))

    print("\n===== FINAL STATE KEYS =====")
    for key in (
        "user_goal",
        "plan",
        "human_plan_feedback",
        "retrieved_papers",
        "human_retrieval_feedback",
        "analysis_result",
        "draft",
    ):
        present = "yes" if key in final_state else "MISSING"
        print(f"  {key}: {present}")


if __name__ == "__main__":
    main()
