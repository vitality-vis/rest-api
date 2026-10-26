"""Tests for the trusted selected-paper tool boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from service.paper_qa import SCOPE_WARNING_END, SCOPE_WARNING_START, PaperQAError
from service.selected_paper_tool import (
    SelectedPaperToolError,
    answer_selected_papers,
)


def _plan(*, use_file_search: bool) -> SimpleNamespace:
    return SimpleNamespace(
        use_file_search=use_file_search,
        metadata_paper_ids=["paper-1", "paper-2"],
        file_search_paper_ids=["paper-1"] if use_file_search else [],
    )


def test_authenticated_call_uses_existing_full_text_flow(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs):
        calls.append({"plan": kwargs})
        return _plan(use_file_search=True), ["metadata"]

    def fake_answer(**kwargs):
        calls.append({"answer": kwargs})
        warning = (
            f"{SCOPE_WARNING_START}"
            '{"unexpected_file_ids":["file-x"]}'
            f"{SCOPE_WARNING_END}"
        )
        return f"Grounded answer [[ID:paper-1]].\n\n{warning}"

    monkeypatch.setattr(
        "service.selected_paper_tool.build_evidence_plan", fake_plan
    )
    monkeypatch.setattr("service.selected_paper_tool.answer", fake_answer)

    result = answer_selected_papers(
        user_id="user-1",
        question="  What method does this paper use?  ",
        paper_ids=[" paper-1 ", "paper-2", "paper-1"],
        model="gpt-test",
    )

    assert result.answer == "Grounded answer [[ID:paper-1]]."
    assert result.evidence == "full_text"
    assert result.paper_ids == ["paper-1", "paper-2"]
    assert result.full_text_paper_ids == ["paper-1"]
    assert calls[0]["plan"]["user_id"] == "user-1"
    assert calls[0]["plan"]["paper_ids"] == ["paper-1", "paper-2"]
    assert calls[0]["plan"]["use_file_search"] is True
    assert calls[1]["answer"]["text"] == "What method does this paper use?"
    assert calls[1]["answer"]["use_file_search"] is True
    assert calls[1]["answer"]["model"] == "gpt-test"
    assert "file-1" not in result.answer


def test_guest_call_stays_on_metadata(monkeypatch):
    seen: dict[str, object] = {}

    def fake_plan(**kwargs):
        seen["plan"] = kwargs
        return _plan(use_file_search=False), ["metadata"]

    def fake_answer(**kwargs):
        seen["answer"] = kwargs
        return "Metadata answer [[ID:paper-1]]."

    monkeypatch.setattr(
        "service.selected_paper_tool.build_evidence_plan", fake_plan
    )
    monkeypatch.setattr("service.selected_paper_tool.answer", fake_answer)

    result = answer_selected_papers(
        user_id=None,
        question="Summarise this paper",
        paper_ids=["paper-1"],
    )

    assert result.evidence == "metadata"
    assert result.full_text_paper_ids == []
    assert seen["plan"]["use_file_search"] is False
    assert seen["answer"]["use_file_search"] is False


def test_provider_error_is_converted_to_tool_error(monkeypatch):
    monkeypatch.setattr(
        "service.selected_paper_tool.build_evidence_plan",
        lambda **_kwargs: (_plan(use_file_search=False), []),
    )

    def fail(**_kwargs):
        raise PaperQAError("Full-text search failed. Please try again.")

    monkeypatch.setattr("service.selected_paper_tool.answer", fail)

    with pytest.raises(SelectedPaperToolError, match="Full-text search failed"):
        answer_selected_papers(
            user_id="user-1",
            question="Explain the method",
            paper_ids=["paper-1"],
        )


@pytest.mark.parametrize(
    ("question", "paper_ids", "message"),
    [
        ("  ", ["paper-1"], "question is required"),
        ("Explain", [], "at least one paper"),
        ("Explain", [" "], "array of strings"),
        ("Explain", "paper-1", "array of strings"),
        ("Explain", [f"paper-{index}" for index in range(21)], "At most 20"),
    ],
)
def test_rejects_invalid_tool_input(question, paper_ids, message):
    with pytest.raises(SelectedPaperToolError, match=message):
        answer_selected_papers(
            user_id="user-1",
            question=question,
            paper_ids=paper_ids,
        )
