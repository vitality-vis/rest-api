"""Live smoke tests for /getPapers, including filtered count(*) totals.

Run against an already-running API server:
    make test-live TESTS=tests/test_api_getpapers.py

With pytest print output (enabled by default via make test-live):
    API_BASE_URL=http://127.0.0.1:3000 pytest tests/test_api_getpapers.py -m live -s
"""

from __future__ import annotations

import os

import pytest
import requests


pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Return the explicitly selected running API server, or skip this live check."""
    base_url = os.getenv("API_BASE_URL")
    if not base_url:
        pytest.skip("API_BASE_URL is not set; no running API server was selected")
    return base_url.rstrip("/")


def _post_json(api_base_url: str, path: str, payload: dict):
    try:
        response = requests.post(f"{api_base_url}{path}", json=payload, timeout=60)
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at {path}: {error}")

    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("application/json")
    return response.json()


def _assert_papers_payload(data: dict) -> dict:
    assert isinstance(data, dict)
    assert isinstance(data.get("papers"), list)
    assert isinstance(data.get("total"), int)
    assert data["total"] >= len(data["papers"])
    assert isinstance(data.get("has_more"), bool)
    return data


def test_get_papers_corpus_total(api_base_url, capsys):
    """Unfiltered /getPapers should return the full corpus size via num_entities."""
    data = _assert_papers_payload(
        _post_json(api_base_url, "/getPapers", {"offset": 0, "limit": 100})
    )

    print(
        f"\n[getPapers] corpus total={data['total']:,} "
        f"(returned {len(data['papers'])} papers, has_more={data['has_more']})"
    )

    assert len(data["papers"]) > 0
    if data["total"] > len(data["papers"]):
        assert data["has_more"] is True


def test_get_papers_chi_source_total(api_base_url, capsys):
    """Source filter CHI should return an exact filtered total via count(*)."""
    payload = {"offset": 0, "limit": 100, "source": ["CHI"]}
    data = _assert_papers_payload(
        _post_json(api_base_url, "/getPapers", payload)
    )

    print(
        f"\n[getPapers] CHI source filter total={data['total']:,} "
        f"(returned {len(data['papers'])} papers, has_more={data['has_more']})"
    )

    for paper in data["papers"][:5]:
        source = paper.get("Source", "")
        print(f"  sample source: {source!r}")

    assert data["total"] >= len(data["papers"])
    if data["total"] > 100:
        assert data["has_more"] is True
        assert len(data["papers"]) == 100
        # A corpus this size should have far more than 101 CHI papers. If total
        # is exactly page_size + 1, the API is still using the old lower-bound
        # estimate instead of Zilliz count(*).
        assert data["total"] != len(data["papers"]) + 1, (
            "total looks like offset + page + has_more; restart the API to pick "
            "up count(*) for filtered queries"
        )
    else:
        assert data["has_more"] is False
        assert len(data["papers"]) == data["total"]


def test_get_papers_cross_field_search_query(api_base_url):
    """Comma-separated search_query terms should match across paper metadata."""
    payload = {
        "offset": 0,
        "limit": 100,
        "search_query": "visualization, LLM",
    }
    data = _assert_papers_payload(_post_json(api_base_url, "/getPapers", payload))

    print(
        f"\n[getPapers] search_query={payload['search_query']!r} "
        f"total={data['total']:,} "
        f"(returned {len(data['papers'])} papers, has_more={data['has_more']})"
    )
    for paper in data["papers"][:5]:
        print(f"  sample title: {paper.get('Title', '')!r}")

    assert data["total"] > 0
