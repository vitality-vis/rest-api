"""Project authorization regressions for selected-paper resolution."""

from __future__ import annotations

import pytest

from repositories.supabase.project_papers_repository import ProjectPapersPersistenceError
from service.paper_registry import LibraryPaperResolutionError, resolve_selected_papers


PRIVATE_ID = "user:private-paper"
PRIVATE_ASSET = {
    "id": "asset-private",
    "paper_id": PRIVATE_ID,
    "origin": "user",
    "metadata_snapshot": {
        "ID": PRIVATE_ID,
        "Title": "Private imported paper",
        "Abstract": "Private project evidence.",
    },
}


def _install_library(monkeypatch):
    monkeypatch.setattr(
        "service.paper_registry.get_user_papers_by_ids",
        lambda **_kwargs: [PRIVATE_ASSET],
    )


def test_private_import_from_project_a_cannot_resolve_in_project_b(monkeypatch):
    _install_library(monkeypatch)
    monkeypatch.setattr(
        "service.paper_registry.has_project_paper",
        lambda **kwargs: kwargs["project_id"] == "project-a",
    )

    with pytest.raises(LibraryPaperResolutionError, match="imported papers are unavailable"):
        resolve_selected_papers(
            user_id="user-1", project_id="project-b", paper_ids=[PRIVATE_ID]
        )


def test_private_import_resolves_with_active_project_membership(monkeypatch):
    _install_library(monkeypatch)
    seen = {}

    def membership(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr("service.paper_registry.has_project_paper", membership)

    resolved = resolve_selected_papers(
        user_id="user-1", project_id="project-a", paper_ids=[PRIVATE_ID]
    )

    assert [paper.paper_id for paper in resolved] == [PRIVATE_ID]
    assert resolved[0].library_paper == PRIVATE_ASSET
    assert seen == {
        "user_id": "user-1",
        "project_id": "project-a",
        "user_paper_id": "asset-private",
    }


def test_public_corpus_only_paper_remains_selectable_across_projects(monkeypatch):
    monkeypatch.setattr(
        "service.paper_registry.get_user_papers_by_ids", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        "service.paper_registry.has_project_paper",
        lambda **_kwargs: pytest.fail("corpus-only papers need no membership lookup"),
    )
    monkeypatch.setattr(
        "service.paper_registry._catalog_metadata_by_id",
        lambda paper_ids: {
            paper_id: {"ID": paper_id, "Title": "Public", "Abstract": "Corpus"}
            for paper_id in paper_ids
        },
    )

    for project_id in ("project-a", "project-b"):
        resolved = resolve_selected_papers(
            user_id="user-1", project_id=project_id, paper_ids=["corpus-1"]
        )
        assert resolved[0].paper_id == "corpus-1"
        assert resolved[0].library_paper is None


def test_membership_persistence_error_fails_closed(monkeypatch):
    _install_library(monkeypatch)

    def unavailable(**_kwargs):
        raise ProjectPapersPersistenceError("database unavailable")

    monkeypatch.setattr("service.paper_registry.has_project_paper", unavailable)

    with pytest.raises(LibraryPaperResolutionError, match="temporarily unavailable"):
        resolve_selected_papers(
            user_id="user-1", project_id="project-a", paper_ids=[PRIVATE_ID]
        )


def test_paper_qa_evidence_plan_forwards_project_id(monkeypatch):
    from service.paper_qa import PaperQAError, build_evidence_plan

    seen: dict[str, object] = {}

    def resolve(**kwargs):
        seen.update(kwargs)
        raise LibraryPaperResolutionError("imported papers are unavailable")

    monkeypatch.setattr("service.paper_qa.resolve_selected_papers", resolve)

    with pytest.raises(PaperQAError, match="imported papers are unavailable"):
        build_evidence_plan(
            user_id="user-1",
            project_id="project-b",
            paper_ids=[PRIVATE_ID],
            use_file_search=False,
        )

    assert seen == {
        "user_id": "user-1",
        "project_id": "project-b",
        "paper_ids": [PRIVATE_ID],
    }
