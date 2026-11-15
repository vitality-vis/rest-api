"""OpenAlex search/hydration helpers used by Codex tools and paper resolve."""

from __future__ import annotations

from service.openalex_papers import (
    annotate_corpus_membership,
    resolve_openalex_api_papers,
    work_summary_to_api_paper,
)


def test_annotate_corpus_membership_prefers_corpus_ids(monkeypatch):
    monkeypatch.setattr(
        "service.openalex_papers.find_corpus_papers_by_dois",
        lambda dois: {"10.1/abc": "corpus-1"},
    )
    annotated = annotate_corpus_membership(
        [
            {"openalex_id": "W1", "doi": "10.1/abc", "title": "In corpus"},
            {"openalex_id": "W2", "doi": "10.2/xyz", "title": "External"},
            {"openalex_id": "W3", "doi": None, "title": "No DOI"},
        ]
    )
    assert annotated[0]["in_corpus"] is True
    assert annotated[0]["paper_id"] == "corpus-1"
    assert annotated[1]["in_corpus"] is False
    assert annotated[1]["paper_id"] == "openalex:W2"
    assert annotated[2]["paper_id"] == "openalex:W3"


def test_work_summary_to_api_paper_uses_stable_openalex_id():
    paper = work_summary_to_api_paper(
        {
            "title": "Example",
            "abstract": "Abs",
            "authors": ["Ada"],
            "keywords": ["hci"],
            "source": "UIST",
            "year": 2026,
            "citation_count": 3,
            "doi": "10.1/abc",
        },
        paper_id="openalex:W1",
    )
    assert paper["ID"] == "openalex:W1"
    assert paper["Title"] == "Example"
    assert paper["Source"] == "UIST"
    assert paper["umap"] is None


def test_resolve_openalex_api_papers(monkeypatch):
    monkeypatch.setattr(
        "service.openalex_papers.get_works_by_ids",
        lambda ids: {
            "W1": {
                "openalex_id": "W1",
                "title": "Resolved",
                "abstract": "Text",
                "authors": [],
                "keywords": [],
                "source": "UIST",
                "year": 2026,
                "citation_count": 0,
                "doi": None,
            }
        },
    )
    resolved = resolve_openalex_api_papers(["openalex:W1", "dblp:other"])
    assert set(resolved) == {"openalex:W1"}
    assert resolved["openalex:W1"]["Title"] == "Resolved"
