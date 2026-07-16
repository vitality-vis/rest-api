"""Retrieval node: return placeholder papers, then pause for human review.

Placeholder logic only — no real search/vector tool. The node loops on
``interrupt()`` so the human can iteratively *refine* the retrieved set:

    accept         -> proceed to analysis
    refine_search  -> add results for the refinement note, then check in again
    (reject is handled by the session layer, which simply drops the thread)

The resume value is a dict ``{"action": <str>, "feedback": <str>}``.
"""
from typing import Any, Dict, List

from langgraph.types import interrupt

from service.agents.lr.lr_state import LRState

_BASE_PAPERS: List[Dict[str, Any]] = [
    {
        "ID": "P1",
        "Title": "Agentic RAG: Orchestrating Retrieval for Scholarly Synthesis",
        "Authors": ["A. Researcher", "B. Scholar"],
        "Year": 2024,
        "Abstract": "Introduces an agentic retrieval-augmented generation "
        "pipeline that plans, retrieves, and synthesizes academic papers.",
    },
    {
        "ID": "P2",
        "Title": "Human-in-the-Loop Workflows for Literature Reviews",
        "Authors": ["C. Author"],
        "Year": 2023,
        "Abstract": "Proposes interactive checkpoints that let researchers "
        "approve plans and curate retrieved evidence before synthesis.",
    },
    {
        "ID": "P3",
        "Title": "Evaluating Multi-Stage RAG for Scientific Summarization",
        "Authors": ["D. Writer", "E. Analyst"],
        "Year": 2025,
        "Abstract": "Benchmarks multi-stage retrieval and analysis strategies "
        "for producing grounded scientific literature summaries.",
    },
]


def _refine_papers(refinements: List[str]) -> List[Dict[str, Any]]:
    """Deterministically derive papers from base + ordered refinement notes."""
    papers = [dict(p) for p in _BASE_PAPERS]
    for idx, note in enumerate(refinements, start=1):
        papers.append(
            {
                "ID": f"R{idx}",
                "Title": f"Refined result for: {note}",
                "Authors": ["Refined Search"],
                "Year": 2025,
                "Abstract": "Additional paper surfaced after refining the "
                f"search with: {note}",
            }
        )
    return papers


def retrieval_node(state: LRState) -> dict:
    """Return placeholder papers and request human review via interrupt()."""
    refinements: List[str] = []

    while True:
        papers = _refine_papers(refinements)
        decision = interrupt(
            {
                "checkpoint_type": "retrieval_review",
                "title": "Retrieved papers",
                "message": "Review the retrieved papers. Accept to continue, "
                "refine the search, or reject to cancel.",
                "retrieved_papers": papers,
            }
        )

        action, feedback = _read_decision(decision)

        if action in ("accept", "approve", "continue"):
            return {
                "retrieved_papers": papers,
                "human_retrieval_feedback": "; ".join(refinements) or None,
            }

        if feedback:
            refinements.append(feedback)
        else:
            return {
                "retrieved_papers": papers,
                "human_retrieval_feedback": "; ".join(refinements) or None,
            }


def _read_decision(decision: Any) -> tuple:
    """Normalize a resume value into ``(action, feedback)``."""
    if isinstance(decision, dict):
        action = str(decision.get("action") or "accept").strip().lower()
        feedback = str(decision.get("feedback") or "").strip()
        return action, feedback
    return "accept", str(decision or "").strip()
