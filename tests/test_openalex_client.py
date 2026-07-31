"""Unit tests for the OpenAlex repository client."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from repositories.openalex import (
    OpenAlexConfigurationError,
    list_citing_works,
    list_referenced_works,
    normalize_doi,
    normalize_openalex_id,
    resolve_work_by_doi,
    summarize_work,
)


def test_normalize_doi_strips_prefixes():
    assert normalize_doi("https://doi.org/10.1145/123") == "10.1145/123"
    assert normalize_doi("doi:10.1145/123") == "10.1145/123"
    assert normalize_doi("  10.1145/123  ") == "10.1145/123"


def test_normalize_openalex_id_accepts_urls():
    assert normalize_openalex_id("https://openalex.org/W1") == "W1"
    assert normalize_openalex_id("W1") == "W1"


def test_summarize_work_projects_fields():
    summary = summarize_work(
        {
            "id": "https://openalex.org/W42",
            "doi": "https://doi.org/10.1/abc",
            "title": "Example",
            "publication_year": 2020,
            "cited_by_count": 7,
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {"display_name": " Alan Turing "}},
                {"author": {"display_name": ""}},
            ],
            "primary_location": {
                "source": {"display_name": "Nature"},
            },
            "abstract_inverted_index": {
                "Deep": [0],
                "learning": [1],
                "works": [2],
            },
            "keywords": [
                {"display_name": "machine learning"},
                {"display_name": "visualization"},
            ],
        }
    )
    assert summary["openalex_id"] == "W42"
    assert summary["title"] == "Example"
    assert summary["year"] == 2020
    assert summary["doi"] == "10.1/abc"
    assert summary["citation_count"] == 7
    assert summary["abstract"] == "Deep learning works"
    assert summary["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert summary["keywords"] == ["machine learning", "visualization"]
    assert summary["source"] == "Nature"
    assert isinstance(summary["raw"], dict)
    assert summary["raw"]["id"] == "https://openalex.org/W42"


def test_reconstruct_abstract_from_inverted_index():
    from repositories.openalex import reconstruct_abstract

    assert (
        reconstruct_abstract({"Hello": [0], "world": [1], "!": [2]})
        == "Hello world !"
    )
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_get_openalex_api_key_requires_configuration():
    with pytest.raises(OpenAlexConfigurationError):
        resolve_work_by_doi("10.1/abc", api_key="")


def test_resolve_work_by_doi_returns_none_when_empty(monkeypatch):
    session = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"results": []}
    session.get.return_value = response

    result = resolve_work_by_doi("10.1/abc", api_key="test-key", session=session)
    assert result is None
    assert session.get.called


def test_resolve_work_by_doi_includes_referenced_works():
    session = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/abc",
                "title": "Seed",
                "publication_year": 2019,
                "cited_by_count": 3,
                "referenced_works": [
                    "https://openalex.org/W2",
                    "https://openalex.org/W3",
                ],
            }
        ]
    }
    session.get.return_value = response

    result = resolve_work_by_doi("10.1/abc", api_key="test-key", session=session)
    assert result is not None
    assert result["openalex_id"] == "W1"
    assert result["referenced_works"] == ["W2", "W3"]


def test_list_referenced_works_batches_and_preserves_order():
    session = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W3",
                "title": "Third",
                "publication_year": 2018,
                "cited_by_count": 1,
                "doi": None,
            },
            {
                "id": "https://openalex.org/W2",
                "title": "Second",
                "publication_year": 2017,
                "cited_by_count": 2,
                "doi": "https://doi.org/10.2/x",
            },
        ]
    }
    session.get.return_value = response

    works = list_referenced_works(
        "W1",
        limit=10,
        referenced_work_ids=["W2", "W3"],
        api_key="test-key",
        session=session,
    )
    assert [work["openalex_id"] for work in works] == ["W2", "W3"]
    params = session.get.call_args.kwargs["params"]
    assert params["filter"] == "openalex:W2|W3"
    assert params["api_key"] == "test-key"
    assert "authorships" in params["select"]
    assert "abstract_inverted_index" in params["select"]


def test_list_citing_works_pages_until_limit(monkeypatch):
    monkeypatch.setattr("repositories.openalex.client.MAX_PER_PAGE", 1)
    session = Mock()
    first = Mock()
    first.status_code = 200
    first.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W10",
                "title": "Cite A",
                "publication_year": 2021,
                "cited_by_count": 0,
                "doi": None,
            }
        ]
    }
    second = Mock()
    second.status_code = 200
    second.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W11",
                "title": "Cite B",
                "publication_year": 2022,
                "cited_by_count": 0,
                "doi": None,
            }
        ]
    }
    session.get.side_effect = [first, second]

    works = list_citing_works("W1", limit=2, api_key="test-key", session=session)
    assert [work["openalex_id"] for work in works] == ["W10", "W11"]
    assert session.get.call_count == 2
    first_filter = session.get.call_args_list[0].kwargs["params"]["filter"]
    assert first_filter == "cites:W1"
