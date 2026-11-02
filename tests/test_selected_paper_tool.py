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


def test_named_group_limits_evidence_candidates(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=["paper-1", "paper-2", "paper-3"],
        selected_paper_groups={
            "library": ["paper-1"],
            "map": ["paper-2", "paper-3"],
        },
        request=EvidenceRequest(detail_level="titles", group="library"),
    )

    assert result["group"] == "library"
    assert result["groups"] == {"library": {"count": 1}, "map": {"count": 2}}
    assert seen[0]["paper_ids"] == ["paper-1"]


def test_named_group_rejects_ids_outside_group(monkeypatch):
    _install_resolver(monkeypatch, [])

    with pytest.raises(SelectedPaperToolError, match="current selection"):
        get_selected_paper_evidence(
            user_id=None,
            selected_paper_ids=["paper-1", "paper-2"],
            selected_paper_groups={"library": ["paper-1"], "map": ["paper-2"]},
            request=EvidenceRequest(
                detail_level="titles", group="library", paper_ids=["paper-2"]
            ),
        )


def test_map_group_representative_sampling_uses_map_space_coverage(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)
    points = [
        {
            "ID": f"paper-{index}",
            "specter_umap": {"x": float(index), "y": 0.0},
        }
        for index in range(10)
    ]
    monkeypatch.setattr(
        "service.selected_paper_evidence._umap_points", lambda: points
    )

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(10)],
        selected_paper_groups={
            "map": [f"paper-{index}" for index in range(10)]
        },
        request=EvidenceRequest(detail_level="titles", limit=3),
    )

    assert result["sampling_policy"] == "deterministic_map_coverage"
    assert set(seen[0]["paper_ids"]) == {"paper-0", "paper-4", "paper-9"}


def test_representative_sampling_falls_back_without_map_coordinates(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)
    monkeypatch.setattr(
        "service.selected_paper_evidence._umap_points", lambda: []
    )

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(10)],
        request=EvidenceRequest(detail_level="titles", limit=3),
    )

    assert result["sampling_policy"] == "deterministic_even_coverage"
    assert seen[0]["paper_ids"] == ["paper-0", "paper-4", "paper-9"]


def test_library_group_representative_sampling_uses_even_coverage(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)
    monkeypatch.setattr(
        "service.selected_paper_evidence._umap_points",
        lambda: [
            {
                "ID": f"paper-{index}",
                "specter_umap": {"x": float(index), "y": 0.0},
            }
            for index in range(10)
        ],
    )

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(10)],
        selected_paper_groups={
            "library": [f"paper-{index}" for index in range(10)]
        },
        request=EvidenceRequest(detail_level="titles", group="library", limit=3),
    )

    assert result["sampling_policy"] == "deterministic_even_coverage"
    assert seen[0]["paper_ids"] == ["paper-0", "paper-4", "paper-9"]


def test_all_groups_representative_sampling_is_stratified(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)
    monkeypatch.setattr(
        "service.selected_paper_evidence._umap_points",
        lambda: [
            {
                "ID": f"paper-{index}",
                "specter_umap": {"x": float(index), "y": 0.0},
            }
            for index in range(8)
        ],
    )

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(10)],
        selected_paper_groups={
            "map": [f"paper-{index}" for index in range(8)],
            "library": ["paper-8", "paper-9"],
        },
        request=EvidenceRequest(detail_level="titles", limit=3),
    )

    assert result["sampling_policy"] == "deterministic_stratified_coverage"
    assert len(seen[0]["paper_ids"]) == 3
    assert set(seen[0]["paper_ids"]) & {"paper-8", "paper-9"}
    assert len(set(seen[0]["paper_ids"]) & {f"paper-{index}" for index in range(8)}) == 2


def test_stratified_sampling_respects_limit_when_map_quota_is_zero(monkeypatch):
    seen: list[dict[str, object]] = []
    _install_resolver(monkeypatch, seen)
    monkeypatch.setattr(
        "service.selected_paper_evidence._umap_points",
        lambda: [
            {
                "ID": "paper-0",
                "specter_umap": {"x": 0.0, "y": 0.0},
            },
            {
                "ID": "paper-1",
                "specter_umap": {"x": 1.0, "y": 0.0},
            },
        ],
    )

    result = get_selected_paper_evidence(
        user_id=None,
        selected_paper_ids=[f"paper-{index}" for index in range(102)],
        selected_paper_groups={
            "map": ["paper-0", "paper-1"],
            "library": [f"paper-{index}" for index in range(2, 102)],
        },
        request=EvidenceRequest(detail_level="titles", limit=1),
    )

    assert result["returned_count"] == 1
    assert seen[0]["paper_ids"] == ["paper-2"]
