"""Analysis node: build a simple structured analysis from retrieved papers.

Placeholder logic only — deterministic derivation from the papers already in
state. Replace with a real analysis agent later; keep the returned key.
"""
from __future__ import annotations

from service.agents.lr.lr_state import LRState


def analysis_node(state: LRState) -> dict:
    """Produce a structured ``analysis_result`` from the retrieved papers."""
    papers = state.get("retrieved_papers", []) or []
    years = [p.get("Year") for p in papers if p.get("Year") is not None]
    year_span = f"{min(years)}-{max(years)}" if years else "n/a"

    analysis_result = {
        "key_themes": [
            "Agentic orchestration of retrieval and synthesis",
            "Human-in-the-loop oversight at plan and evidence stages",
            "Multi-stage RAG for scientific summarization",
        ],
        "common_methods": [
            "Retrieval-augmented generation (RAG)",
            "Staged planning with explicit checkpoints",
            "Embedding-based paper retrieval",
        ],
        "limitations": [
            "Placeholder evaluation; no large-scale benchmark",
            "Limited corpus coverage in the reviewed sample",
            "Human feedback not yet incorporated into re-planning",
        ],
        "possible_gap": "Closing the loop so human feedback at each checkpoint "
        "automatically refines the plan and retrieval, rather than only "
        "annotating a single linear pass.",
        "papers_considered": len(papers),
        "year_span": year_span,
    }

    return {"analysis_result": analysis_result}
