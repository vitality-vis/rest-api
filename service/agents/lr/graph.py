"""LangGraph wiring for the literature-review (LR) human-in-the-loop demo.

Graph shape:

    START -> plan -> retrieval -> analysis -> writer -> END

``plan`` and ``retrieval`` use ``interrupt()`` to pause for human approval /
review. The graph is compiled with an in-memory checkpointer so a thread can be
paused and resumed via ``Command(resume=...)``.

This module is self-contained and does not import or modify the existing
chat/RAG graph or agents.
"""
from __future__ import annotations

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from service.agents.lr.lr_state import LRState
from service.agents.lr.nodes import (
    plan_node,
    retrieval_node,
    analysis_node,
    writer_node,
)


def build_lr_graph():
    """Build and compile the LR demo graph with an in-memory checkpointer."""
    builder = StateGraph(LRState)

    builder.add_node("plan", plan_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieval")
    builder.add_edge("retrieval", "analysis")
    builder.add_edge("analysis", "writer")
    builder.add_edge("writer", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
