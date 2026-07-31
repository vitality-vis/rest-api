"""Live smoke tests for /getPaperCitations via OpenAlex.

Run against an already-running API server with OPENALEX_API_KEY configured:

    make test-live TESTS=tests/test_api_paper_citations.py

Override the seed DOI if needed:

    PAPER_CITATIONS_TEST_DOI=10.1109/TVCG.2024.3408255 \\
      make test-live TESTS=tests/test_api_paper_citations.py
"""

from __future__ import annotations

import os

import pytest
import requests


pytestmark = pytest.mark.live

DEFAULT_TEST_DOI = "10.1109/TVCG.2024.3408255"


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Return the explicitly selected running API server, or skip this live check."""
    base_url = os.getenv("API_BASE_URL")
    if not base_url:
        pytest.skip("API_BASE_URL is not set; no running API server was selected")
    return base_url.rstrip("/")


@pytest.fixture(scope="session")
def paper_citations_test_doi() -> str:
    return (os.getenv("PAPER_CITATIONS_TEST_DOI") or DEFAULT_TEST_DOI).strip()


def _post_json(api_base_url: str, path: str, payload: dict, *, timeout: int = 120):
    try:
        response = requests.post(
            f"{api_base_url}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at {path}: {error}")
    return response


def _assert_citation_item(item: dict, *, label: str) -> None:
    assert isinstance(item, dict), f"{label} item must be an object"
    assert isinstance(item.get("openalex_id"), str) and item["openalex_id"], (
        f"{label} item missing openalex_id: {item!r}"
    )
    assert isinstance(item.get("in_corpus"), bool), (
        f"{label} item missing in_corpus bool: {item!r}"
    )
    if item.get("in_corpus"):
        assert isinstance(item.get("ID"), str) and item["ID"], (
            f"{label} in_corpus item missing ID: {item!r}"
        )
    assert isinstance(item.get("Title"), str), f"{label} missing Title"
    assert isinstance(item.get("Abstract"), str), f"{label} missing Abstract"
    assert isinstance(item.get("Authors"), list), f"{label} missing Authors"
    assert isinstance(item.get("Keywords"), list), f"{label} missing Keywords"
    assert isinstance(item.get("Source"), str), f"{label} missing Source"
    if "Year" in item:
        assert item["Year"] is None or isinstance(item["Year"], int)
    if "doi" in item:
        assert item["doi"] is None or isinstance(item["doi"], str)
    if "CitationCounts" in item:
        assert item["CitationCounts"] is None or isinstance(
            item["CitationCounts"], int
        )
    if "raw" in item and item["raw"] is not None:
        assert isinstance(item["raw"], dict)


def test_get_paper_citations_for_known_doi(
    api_base_url, paper_citations_test_doi, capsys
):
    """POST /getPaperCitations should return OpenAlex neighbors for a known DOI."""
    response = _post_json(
        api_base_url,
        "/getPaperCitations",
        {"doi": paper_citations_test_doi, "limit": 20},
    )

    if response.status_code == 503:
        pytest.fail(
            "Paper citations unavailable (503). Configure OPENALEX_API_KEY on "
            f"the API server and restart it. Body: {response.text}"
        )
    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)
    source = data.get("source")
    assert isinstance(source, dict)
    assert source.get("doi") == paper_citations_test_doi
    assert isinstance(source.get("openalex_id"), str) and source["openalex_id"]

    references_group = data.get("references")
    cited_by_group = data.get("cited_by")
    assert isinstance(references_group, dict)
    assert isinstance(cited_by_group, dict)
    assert isinstance(references_group.get("total_hint"), int)
    assert isinstance(cited_by_group.get("total_hint"), int)

    references = references_group.get("papers")
    cited_by = cited_by_group.get("papers")
    assert isinstance(references, list)
    assert isinstance(cited_by, list)
    assert references_group["total_hint"] >= len(references)
    assert cited_by_group["total_hint"] >= len(cited_by)
    assert references or cited_by, (
        "Expected at least one reference or cited-by paper from OpenAlex"
    )

    for item in references:
        _assert_citation_item(item, label="references")
    for item in cited_by:
        _assert_citation_item(item, label="cited_by")

    print(
        f"\n[getPaperCitations] doi={paper_citations_test_doi!r} "
        f"openalex_id={source['openalex_id']!r} "
        f"references={len(references)}/{references_group['total_hint']} "
        f"cited_by={len(cited_by)}/{cited_by_group['total_hint']}"
    )
    for item in references[:3]:
        print(f"  reference: {item.get('Title')!r} ({item.get('Year')})")
    for item in cited_by[:3]:
        print(f"  cited_by: {item.get('Title')!r} ({item.get('Year')})")
