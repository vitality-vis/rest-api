"""Node functions for the literature-review (LR) human-in-the-loop demo graph.

Each node is a small, deterministic, placeholder step so the demo always runs
without a real LLM or search tool. Swap the placeholder bodies for real agent
calls later — the function signatures and returned keys are the contract.
"""
from service.agents.lr.nodes.plan_node import plan_node
from service.agents.lr.nodes.retrieval_node import retrieval_node
from service.agents.lr.nodes.analysis_node import analysis_node
from service.agents.lr.nodes.writer_node import writer_node

__all__ = [
    "plan_node",
    "retrieval_node",
    "analysis_node",
    "writer_node",
]
