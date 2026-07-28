"""Live smoke test for the experimental ``/chat/v2`` endpoint.

Run against an already-running API server:
    API_BASE_URL=http://127.0.0.1:3000 pytest tests/test_api_chat_v2.py -m live -s
"""

from __future__ import annotations

import os
from uuid import uuid4

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


def test_chat_v2_returns_text_for_hello(api_base_url: str):
    """A non-search turn can fall back to the legacy chat path and stream text."""
    payload = {
        "chat_id": f"api-chat-v2-smoke-{uuid4()}",
        "text": "Hello",
    }

    try:
        response = requests.post(f"{api_base_url}/chat/v2", json=payload, timeout=180)
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at /chat/v2: {error}")

    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("text/plain")
    assert response.text.strip()


def test_chat_v2_finds_papers(api_base_url: str):
    """A paper-finding turn reaches v2 and emits the paper-panel payload."""
    payload = {
        "chat_id": f"api-chat-v2-paper-search-{uuid4()}",
        "text": (
            "Find papers about using large language models to support literature "
            "reviews. List the relevant papers."
        ),
        "effort": "low",
    }

    try:
        response = requests.post(f"{api_base_url}/chat/v2", json=payload, timeout=180)
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at /chat/v2: {error}")

    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("text/plain")
    print(f"\n[chat v2 paper search]\n{response.text.strip()}")
    assert "[[VITALITY_PAPERS_JSON]]" in response.text, response.text
