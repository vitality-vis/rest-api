"""Live smoke tests for the client bootstrap endpoints.

Run against an already-running API server:
    API_BASE_URL=http://127.0.0.1:3000 make test-live TESTS=tests/test_api_bootstrap.py
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


def _get_json(api_base_url: str, path: str):
    try:
        response = requests.get(f"{api_base_url}{path}", timeout=30)
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at {path}: {error}")

    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("application/json")
    return response.json()


def test_get_umap_points(api_base_url):
    """The running API serves the unchanged UMAP endpoint as a JSON array."""
    assert isinstance(_get_json(api_base_url, "/getUmapPoints"), list)


def test_get_metadata(api_base_url):
    """The running API serves all filter facets at the unchanged metadata endpoint."""
    data = _get_json(api_base_url, "/getMetaData")

    assert isinstance(data, dict)
    assert {
        "authors_summary",
        "sources_summary",
        "keywords_summary",
        "years_summary",
        "citation_counts",
    } <= data.keys()
