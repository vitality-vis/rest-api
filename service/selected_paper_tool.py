"""Trusted tool boundary for answering questions about selected papers.

The model may supply a question and paper IDs. The caller supplies the
authenticated user ID. This module never accepts an Azure file ID, vector
store ID, or credential from the model, and it is intentionally not registered
on the public MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from service.paper_qa import (
    SCOPE_WARNING_END,
    SCOPE_WARNING_START,
    PaperQAError,
    answer,
    build_evidence_plan,
)


class SelectedPaperToolError(RuntimeError):
    """User-safe failure from the selected-paper tool."""


MAX_SELECTED_PAPERS = 20
MAX_QUESTION_CHARS = 10_000
EvidenceMode = Literal["metadata", "full_text"]


@dataclass(frozen=True)
class SelectedPaperAnswer:
    """Grounded answer and the evidence actually used."""

    answer: str
    evidence: EvidenceMode
    paper_ids: list[str]
    full_text_paper_ids: list[str]


def answer_selected_papers(
    *,
    user_id: str | None,
    question: str,
    paper_ids: list[str],
    model: str | None = None,
) -> SelectedPaperAnswer:
    """Answer from selected papers, using indexed full text when it is available.

    Authenticated users use Azure File Search for papers whose uploads are
    indexed. Papers without full text, and all guest requests, use metadata.
    """
    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question:
        raise SelectedPaperToolError("A question is required.")
    if len(normalized_question) > MAX_QUESTION_CHARS:
        raise SelectedPaperToolError(
            f"The question must be at most {MAX_QUESTION_CHARS:,} characters."
        )

    normalized_ids = _normalize_paper_ids(paper_ids)
    use_file_search = bool(user_id)
    try:
        plan, _metadata = build_evidence_plan(
            user_id=user_id,
            paper_ids=normalized_ids,
            use_file_search=use_file_search,
        )
        grounded_answer = answer(
            user_id=user_id,
            paper_ids=normalized_ids,
            text=normalized_question,
            use_file_search=use_file_search,
            model=model,
        )
    except PaperQAError as error:
        raise SelectedPaperToolError(str(error)) from error

    return SelectedPaperAnswer(
        answer=_strip_scope_warning(grounded_answer),
        evidence="full_text" if plan.use_file_search else "metadata",
        paper_ids=list(plan.metadata_paper_ids),
        full_text_paper_ids=list(plan.file_search_paper_ids),
    )


def _normalize_paper_ids(paper_ids: object) -> list[str]:
    if not isinstance(paper_ids, list):
        raise SelectedPaperToolError("paper_ids must be an array of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise SelectedPaperToolError("paper_ids must be an array of strings.")
        value = paper_id.strip()
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise SelectedPaperToolError("Select at least one paper first.")
    if len(normalized) > MAX_SELECTED_PAPERS:
        raise SelectedPaperToolError(
            f"At most {MAX_SELECTED_PAPERS} papers can be used at once."
        )
    return normalized


def _strip_scope_warning(answer_text: str) -> str:
    """Remove the internal scope marker before the answer reaches the model."""
    start = answer_text.find(SCOPE_WARNING_START)
    if start < 0:
        return answer_text
    end = answer_text.find(SCOPE_WARNING_END, start)
    if end < 0:
        return answer_text[:start].rstrip()
    stripped = answer_text[:start] + answer_text[end + len(SCOPE_WARNING_END) :]
    return stripped.strip()
