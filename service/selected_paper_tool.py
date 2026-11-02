"""Trusted Codex boundary for model-free access to selected-paper evidence.

Codex may choose the evidence detail and a subset, but the server injects the
current selection and authenticated user. The model can never access papers
outside that selection or supply storage credentials.
"""

from __future__ import annotations

from service.selected_paper_evidence import (
    EvidenceRequest,
    SelectedPaperEvidenceError,
    get_selected_paper_evidence,
    inspect_selected_papers,
)

SelectedPaperToolError = SelectedPaperEvidenceError

__all__ = [
    "EvidenceRequest",
    "SelectedPaperToolError",
    "get_selected_paper_evidence",
    "inspect_selected_papers",
]
