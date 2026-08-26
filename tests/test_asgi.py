"""ASGI composition smoke tests."""

from __future__ import annotations

import logging

import pytest
from starlette.testclient import TestClient

from app.application import create_application
from app.asgi.lifespan import startup_bundle
from app.profiles import AppProfile


def _fake_logger():
    logger = logging.getLogger("vitality2-test-asgi")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@pytest.fixture
def suppress_lifecycle(monkeypatch):
    monkeypatch.setattr(
        "app.profiles.papers.initialize_runtime", lambda **_kwargs: _fake_logger()
    )
    monkeypatch.setattr(
        "app.profiles.full.initialize_runtime", lambda **_kwargs: _fake_logger()
    )
    monkeypatch.setattr(
        "app.asgi.lifespan.cached_data.init", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "app.asgi.lifespan.cached_data.zilliz_ready", True, raising=False
    )
    monkeypatch.setattr(
        "agents.agent_v1_legacy.runner.reset_all_sessions", lambda: None
    )


def test_papers_asgi_health_via_wsgi_middleware(suppress_lifecycle, monkeypatch):
    monkeypatch.setattr(
        "app.profiles.papers.cached_data.zilliz_ready", True, raising=False
    )
    bundle = create_application(AppProfile.PAPERS)
    assert bundle.socketio is None

    with TestClient(bundle.asgi_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "papers"
    assert payload["capabilities"]["socketIo"] is False
    assert payload["capabilities"]["chat"] is False


def test_papers_asgi_chat_is_404(suppress_lifecycle):
    bundle = create_application(AppProfile.PAPERS)
    with TestClient(bundle.asgi_app) as client:
        response = client.post("/chat", json={"text": "hi"})
    assert response.status_code == 404


def test_full_asgi_health_and_socket_server(suppress_lifecycle, monkeypatch):
    monkeypatch.setattr(
        "app.profiles.full.cached_data.zilliz_ready", True, raising=False
    )
    monkeypatch.setattr(
        "app.profiles.config.is_azure_chat_configured", lambda: True
    )
    monkeypatch.setattr(
        "app.profiles.config.is_supabase_configured", lambda: True
    )
    bundle = create_application(AppProfile.FULL)
    assert bundle.socketio is not None

    with TestClient(bundle.asgi_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "full"
    assert payload["capabilities"]["socketIo"] is True


def test_startup_bundle_refreshes_capabilities(suppress_lifecycle, monkeypatch):
    monkeypatch.setattr(
        "app.profiles.papers.cached_data.zilliz_ready", False, raising=False
    )
    bundle = create_application(AppProfile.PAPERS)
    assert bundle.capabilities.paper_search is False

    monkeypatch.setattr(
        "app.asgi.lifespan.cached_data.zilliz_ready", True, raising=False
    )
    startup_bundle(bundle)
    assert bundle.capabilities.paper_search is True
    assert bundle.flask_app.config["VITALITY_CAPABILITIES"]["paperSearch"] is True


def test_shutdown_bundle_awaits_socketio_shutdown(suppress_lifecycle):
    import asyncio
    from unittest.mock import AsyncMock

    from app.asgi.lifespan import shutdown_bundle

    bundle = create_application(AppProfile.FULL)
    bundle.socketio = AsyncMock()
    bundle.socketio.shutdown = AsyncMock()
    asyncio.run(shutdown_bundle(bundle))
    bundle.socketio.shutdown.assert_awaited_once()
