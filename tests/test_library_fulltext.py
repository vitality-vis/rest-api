"""Tests for Codex library full-text evidence tools."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from service.library_fulltext import (
    LibraryContentRequest,
    LibraryFulltextError,
    LibrarySearchRequest,
    get_selected_paper_full_text,
    search_library_full_text,
)


@dataclass(frozen=True)
class FakePaper:
    paper_id: str
    metadata: dict[str, object]
    library_paper: dict[str, object] | None = None


def _paper(paper_id: str, *, indexed: bool) -> FakePaper:
    index = paper_id.rsplit("-", 1)[-1]
    return FakePaper(
        paper_id=paper_id,
        metadata={"Title": f"Title {index}"},
        library_paper=(
            {
                "vs_file_status": "completed",
                "azure_file_id": f"file-{index}",
            }
            if indexed
            else {"vs_file_status": "failed", "azure_file_id": f"file-{index}"}
        ),
    )


def test_search_library_full_text_post_filters_by_allowlist(monkeypatch):
    def resolve(**kwargs):
        return [
            _paper(paper_id, indexed=True)
            for paper_id in kwargs["paper_ids"]
        ]

    monkeypatch.setattr(
        "service.library_fulltext.resolve_selected_papers", resolve
    )
    monkeypatch.setattr(
        "repositories.supabase.user_vector_stores_repository.get_user_vector_store",
        lambda **kwargs: {"azure_vector_store_id": "vs-1"},
    )

    def fake_search(**kwargs):
        assert kwargs["vector_store_id"] == "vs-1"
        assert "filters" not in kwargs
        return {
            "data": [
                {
                    "file_id": "file-1",
                    "score": 0.9,
                    "content": [{"type": "text", "text": "alpha chunk"}],
                },
                {
                    "file_id": "file-other",
                    "score": 0.8,
                    "content": [{"type": "text", "text": "outside chunk"}],
                },
                {
                    "file_id": "file-2",
                    "score": 0.7,
                    "content": [{"type": "text", "text": "beta chunk"}],
                },
            ]
        }

    monkeypatch.setattr(
        "repositories.azure_openai.vector_stores.search_vector_store", fake_search
    )

    result = search_library_full_text(
        user_id="user-1",
        project_id="project-1",
        selected_paper_ids=["paper-1", "paper-2", "paper-3"],
        request=LibrarySearchRequest(query="methods", paper_ids=["paper-1", "paper-2"]),
    )

    assert result["searched_paper_ids"] == ["paper-1", "paper-2"]
    assert result["chunk_count"] == 2
    assert result["omitted_outside_scope"] == 1
    assert [chunk["paper_id"] for chunk in result["chunks"]] == ["paper-1", "paper-2"]
    assert result["chunks"][0]["text"] == "alpha chunk"


def test_search_skips_papers_without_full_text(monkeypatch):
    def resolve(**kwargs):
        return [
            _paper("paper-1", indexed=True),
            _paper("paper-2", indexed=False),
        ]

    monkeypatch.setattr(
        "service.library_fulltext.resolve_selected_papers", resolve
    )
    monkeypatch.setattr(
        "repositories.supabase.user_vector_stores_repository.get_user_vector_store",
        lambda **kwargs: {"azure_vector_store_id": "vs-1"},
    )
    monkeypatch.setattr(
        "repositories.azure_openai.vector_stores.search_vector_store",
        lambda **kwargs: {"data": []},
    )

    result = search_library_full_text(
        user_id="user-1",
        selected_paper_ids=["paper-1", "paper-2"],
        request=LibrarySearchRequest(query="evaluation"),
    )

    assert result["searched_paper_ids"] == ["paper-1"]
    assert result["skipped_papers"] == [
        {"paper_id": "paper-2", "reason": "full_text_unavailable"}
    ]


def test_search_requires_sign_in():
    with pytest.raises(LibraryFulltextError, match="Sign in"):
        search_library_full_text(
            user_id=None,
            selected_paper_ids=["paper-1"],
            request=LibrarySearchRequest(query="x"),
        )


def test_get_selected_paper_full_text_paginates(monkeypatch):
    monkeypatch.setattr(
        "service.library_fulltext.resolve_selected_papers",
        lambda **kwargs: [_paper("paper-1", indexed=True)],
    )
    monkeypatch.setattr(
        "repositories.supabase.user_vector_stores_repository.get_user_vector_store",
        lambda **kwargs: {"azure_vector_store_id": "vs-1"},
    )
    monkeypatch.setattr(
        "repositories.azure_openai.vector_stores.get_vector_store_file_content",
        lambda **kwargs: {
            "data": [
                {"type": "text", "text": "seg-0"},
                {"type": "text", "text": "seg-1"},
                {"type": "text", "text": "seg-2"},
            ],
            "has_more": False,
        },
    )

    result = get_selected_paper_full_text(
        user_id="user-1",
        selected_paper_ids=["paper-1"],
        request=LibraryContentRequest(paper_id="paper-1", offset=1, limit=1),
    )

    assert result["full_text_available"] is True
    assert result["segments"] == [{"index": 1, "text": "seg-1"}]
    assert result["has_more"] is True
    assert result["next_offset"] == 2


def test_get_selected_paper_full_text_rejects_outside_selection(monkeypatch):
    with pytest.raises(LibraryFulltextError, match="current selection"):
        get_selected_paper_full_text(
            user_id="user-1",
            selected_paper_ids=["paper-1"],
            request=LibraryContentRequest(paper_id="paper-2"),
        )
