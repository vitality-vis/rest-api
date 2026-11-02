"""Tests for bounded selected-paper evidence exposed to Codex."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from service.selected_paper_tool import (
    EvidenceRequest,
    SelectedPaperToolError,
    get_selected_paper_evidence,
    inspect_selected_papers,
)


@dataclass(frozen=True)
class FakePaper:
    paper_id: str
    metadata: dict[str, object]
    library_paper: dict[str, object] | None = None


def _paper(paper_id: str, index: int) -> FakePaper:
    return FakePaper(
        paper_id=paper_id,
        metadata={
            "Title": f"Title {index}",
            "Abstract": f"Abstract {index} " + ("x" * 3_000),
            "Authors": [f"Author {index}"],
            "Keywords": ["topic"],
            "Source": "Venue",
            "Year": 2000 + index,
            "CitationCounts": index * 10,
            "doi": f"10.test/{index}",
        },
        library_paper=(
            {"vs_file_status": "completed", "azure_file_id": f"file-{index}"}
            if index % 2 == 0
            else None
        ),
    )


def _install_resolver(monkeypatch, seen):
    def resolve(**kwargs):
        seen.append(kwargs)
        return [_paper(paper_id, int(paper_id.split("-")[-1])) for paper_id in kwargs["paper_ids"]]

    monkeypatch.setattr(
        "service.selected_paper_evidence.resolve_selected_papers", resolve
    )


def test_inspection_resolves_only_a_representative_sample(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = inspect_selected_papers(
        user_id="user-1",
        selected_paper_ids=[f"paper-{index}" for index in range(30)],
    )

    assert result["selected_count"] == 30
    assert result["sampled_count"] == 12
    assert len(seen[0]["paper_ids"]) == 12
    assert seen[0]["paper_ids"][0] == "paper-0"
    assert seen[0]["paper_ids"][-1] == "paper-29"
    assert "abstract" not in result["sampled_papers"][0]
    assert "full_text_available" in result["sampled_papers"][0]


def test_auto_evidence_caps_records_and_abstracts(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = get_selected_paper_evidence(
        user_id="user-1",
        selected_paper_ids=[f"paper-{index}" for index in range(150)],
        request=EvidenceRequest(),
    )

    assert result["selected_count"] == 150
    assert result["returned_count"] == 100
    assert result["detailed_count"] == 20
    assert result["omitted_count"] == 50
    assert len(seen[0]["paper_ids"]) == 100
    assert sum("abstract" in paper for paper in result["papers"]) == 20
    assert all(
        len(paper["abstract"]) <= 1_000
        for paper in result["papers"]
        if "abstract" in paper
    )
    assert "Disclose" in result["instruction"]


def test_abstract_request_is_capped_at_twenty(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(40)],
        request=EvidenceRequest(detail_level="abstracts", limit=100),
    )

    assert result["returned_count"] == 20
    assert result["detailed_count"] == 20
    assert len(seen[0]["paper_ids"]) == 20


def test_small_selection_gets_larger_abstract_budget(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=["paper-1", "paper-2"],
        request=EvidenceRequest(detail_level="abstracts"),
    )

    assert result["policy"]["abstract_chars"] == 2_500
    assert len(result["papers"][0]["abstract"]) == 2_500


def test_requested_subset_must_be_inside_current_selection(monkeypatch):
    _install_resolver(monkeypatch, [])

    with pytest.raises(SelectedPaperToolError, match="current selection"):
        get_selected_paper_evidence(
            user_id="user-1",
            selected_paper_ids=["paper-1"],
            request=EvidenceRequest(paper_ids=["paper-2"]),
        )


def test_duplicate_ids_are_removed_before_sampling(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=["paper-1", "paper-1", "paper-2"],
        request=EvidenceRequest(detail_level="titles"),
    )

    assert result["selected_count"] == 2
    assert seen[0]["paper_ids"] == ["paper-1", "paper-2"]
