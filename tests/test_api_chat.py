"""Live smoke test for the streaming chat endpoint.

Run against an already-running API server:
    API_BASE_URL=http://127.0.0.1:3000 pytest tests/test_api_chat.py -m live -s
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


def test_chat_returns_text_for_hello(api_base_url: str):
    """A minimal Hello request returns a non-empty plain-text Agent response."""
    payload = {
        "chat_id": f"api-chat-smoke-{uuid4()}",
        "text": "Hello",
    }

    try:
        response = requests.post(f"{api_base_url}/chat", json=payload, timeout=120)
    except requests.RequestException as error:
        pytest.fail(f"Could not reach API_BASE_URL at /chat: {error}")

    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("text/plain")
    assert response.text.strip()


# TODO: Add focused live tests for Agent memory restoration, tool calls,
# structured paper results, and stream errors after those behaviours have
# stable contracts.
