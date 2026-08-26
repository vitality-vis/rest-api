"""Profile factory and /health behavior."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import config
from app.application import create_application
from app.profiles import AppProfile, discover_capabilities, resolve_profile


REST_API_ROOT = Path(__file__).resolve().parents[1]


def _fake_logger():
    logger = logging.getLogger("vitality2-test-profile")
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
        "app.profiles.papers.cached_data.zilliz_ready", True, raising=False
    )
    monkeypatch.setattr(
        "app.asgi.lifespan.startup_bundle", lambda _bundle: None
    )
    monkeypatch.setattr(
        "app.asgi.lifespan.shutdown_bundle", lambda _bundle: None
    )
    monkeypatch.setattr(
        "agents.agent_v1_legacy.runner.reset_all_sessions", lambda: None
    )


def test_resolve_profile_defaults_to_full(monkeypatch):
    monkeypatch.delenv("VITALITY_APP_PROFILE", raising=False)
    assert resolve_profile() is AppProfile.FULL


def test_resolve_profile_reads_env(monkeypatch):
    monkeypatch.setenv("VITALITY_APP_PROFILE", "papers")
    assert resolve_profile() is AppProfile.PAPERS


def test_resolve_profile_rejects_unknown():
    with pytest.raises(ValueError, match="VITALITY_APP_PROFILE"):
        resolve_profile("nope")


def test_external_capability_configuration_requires_every_value(monkeypatch):
    chat_values = {
        "AZURE_OPENAI_ENDPOINT": "https://azure.example",
        "AZURE_OPENAI_API_KEY": "key",
        "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
        "AZURE_OPENAI_AVAILABLE_MODELS": {"gpt": "deployment"},
        "AZURE_OPENAI_DEFAULT_MODEL": "gpt",
    }
    for name, value in chat_values.items():
        monkeypatch.setattr(config, name, value)
    assert config.is_azure_chat_configured() is True
    for name, value in chat_values.items():
        monkeypatch.setattr(config, name, {} if isinstance(value, dict) else "")
        assert config.is_azure_chat_configured() is False
        monkeypatch.setattr(config, name, value)

    embedding_values = {
        "AZURE_OPENAI_ENDPOINT": "https://azure.example",
        "AZURE_OPENAI_API_KEY": "key",
        "AZURE_OPENAI_EMBED_API_VERSION": "2024-02-01",
        "AZURE_OPENAI_EMBED_DEPLOYMENT": "embed",
    }
    for name, value in embedding_values.items():
        monkeypatch.setattr(config, name, value)
    assert config.is_azure_embedding_configured() is True
    for name, value in embedding_values.items():
        monkeypatch.setattr(config, name, "")
        assert config.is_azure_embedding_configured() is False
        monkeypatch.setattr(config, name, value)

    monkeypatch.setattr(config, "SUPABASE_URL", "https://sb")
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", "key")
    assert config.is_supabase_configured() is True
    monkeypatch.setattr(config, "SUPABASE_URL", "")
    assert config.is_supabase_configured() is False


def test_discover_capabilities_requires_zilliz_probe(monkeypatch):
    monkeypatch.setattr(
        "app.profiles.config.is_azure_embedding_configured", lambda: True
    )
    monkeypatch.setattr("app.profiles.config.is_azure_chat_configured", lambda: False)
    monkeypatch.setattr("app.profiles.config.is_supabase_configured", lambda: False)

    papers = discover_capabilities(
        AppProfile.PAPERS,
        zilliz_ready=False,
        socket_io_enabled=False,
    )
    assert papers.as_dict() == {
        "paperSearch": False,
        "bm25Search": False,
        "vectorSearch": False,
        "chat": False,
        "userLibrary": False,
        "socketIo": False,
    }


def test_discover_capabilities_reflects_runtime_probes(monkeypatch):
    monkeypatch.setattr(
        "app.profiles.config.is_azure_embedding_configured", lambda: False
    )
    monkeypatch.setattr("app.profiles.config.is_azure_chat_configured", lambda: False)
    monkeypatch.setattr("app.profiles.config.is_supabase_configured", lambda: False)

    papers = discover_capabilities(
        AppProfile.PAPERS,
        zilliz_ready=True,
        socket_io_enabled=False,
    )
    assert papers.as_dict() == {
        "paperSearch": True,
        "bm25Search": True,
        "vectorSearch": False,
        "chat": False,
        "userLibrary": False,
        "socketIo": False,
    }

    monkeypatch.setattr(
        "app.profiles.config.is_azure_embedding_configured", lambda: True
    )
    papers_vector = discover_capabilities(
        AppProfile.PAPERS,
        zilliz_ready=True,
        socket_io_enabled=False,
    )
    assert papers_vector.vector_search is True

    monkeypatch.setattr("app.profiles.config.is_azure_chat_configured", lambda: True)
    monkeypatch.setattr("app.profiles.config.is_supabase_configured", lambda: True)
    full = discover_capabilities(
        AppProfile.FULL,
        zilliz_ready=True,
        socket_io_enabled=True,
    )
    assert full.chat is True
    assert full.user_library is True
    assert full.socket_io is True


def test_papers_profile_route_table(suppress_lifecycle):
    bundle = create_application(AppProfile.PAPERS)
    rules = {
        (rule.rule, tuple(sorted(rule.methods - {"HEAD", "OPTIONS"})))
        for rule in bundle.flask_app.url_map.iter_rules()
        if rule.endpoint != "static"
    }
    assert ("/health", ("GET",)) in rules
    assert ("/getPapers", ("GET", "POST")) in rules
    assert ("/getSimilarPapers", ("POST",)) in rules
    assert ("/getPaperCitations", ("POST",)) in rules
    assert ("/getPaperById", ("GET",)) in rules
    assert ("/getPaperByTitle", ("POST",)) in rules
    assert ("/getUmapPoints", ("GET",)) in rules
    assert ("/getMetaData", ("GET",)) in rules
    forbidden = {
        "/chat",
        "/chat/v2",
        "/library/papers",
        "/notes",
        "/papers/resolve",
        "/getPublicConfig",
        "/checkoutPapers",
        "/resetMemory",
        "/",
    }
    present = {rule for rule, _ in rules}
    assert present.isdisjoint(forbidden)
    assert bundle.socketio is None
    assert bundle.asgi_app is not None
    assert callable(bundle.asgi_app)


def test_papers_health_endpoint(suppress_lifecycle, monkeypatch):
    monkeypatch.setattr(
        "app.profiles.papers.cached_data.zilliz_ready", False, raising=False
    )
    bundle = create_application(AppProfile.PAPERS)
    client = bundle.flask_app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["profile"] == "papers"
    assert payload["capabilities"]["paperSearch"] is False
    assert payload["capabilities"]["chat"] is False
    assert payload["capabilities"]["socketIo"] is False
    assert payload.get("agentRuntime") is None

    monkeypatch.setattr(
        "app.profiles.papers.cached_data.zilliz_ready", True, raising=False
    )
    response = client.get("/health")
    payload = response.get_json()
    # Health returns the startup/provisional snapshot, not a live re-probe.
    assert payload["capabilities"]["paperSearch"] is False


def test_full_profile_includes_user_and_chat_routes(suppress_lifecycle):
    bundle = create_application(AppProfile.FULL)
    rules = {rule.rule for rule in bundle.flask_app.url_map.iter_rules()}
    assert "/health" in rules
    assert "/getPapers" in rules
    assert "/chat" not in rules
    assert "/chat/import" in rules
    assert "/chat/conversations" in rules
    assert "/chat_stream_simple" not in rules
    assert "/chat/v2" not in rules
    assert "/library/papers" in rules
    assert "/notes" in rules
    assert "/papers/resolve" in rules
    assert bundle.socketio is not None
    assert bundle.asgi_app is not None
    assert callable(bundle.asgi_app)

    # The native FastAPI route owns the typed SSE protocol before the catch-all
    # Flask mount. Socket.IO keeps that HTTP app as its ASGI fallback.
    http_app = bundle.asgi_app.other_asgi_app
    asgi_routes = {route.path for route in http_app.routes}
    assert "/chat/v2" in asgi_routes


def test_papers_factory_import_skips_agents_and_socketio():
    script = """
import sys
import logging

fake = logging.getLogger("t")
fake.handlers = []
fake.setLevel(logging.INFO)

import app.profiles.papers as papers_mod
papers_mod.initialize_runtime = lambda **k: fake

from app.application import create_application
from app.profiles import AppProfile
create_application(AppProfile.PAPERS)

forbidden = [
    name for name in sys.modules
    if name == "agents" or name.startswith("agents.")
    or name == "flask_socketio" or name.startswith("flask_socketio.")
    or name == "socketio" or name.startswith("socketio.")
    or name == "repositories.supabase" or name.startswith("repositories.supabase.")
]
if forbidden:
    raise SystemExit("forbidden imports: " + ", ".join(sorted(forbidden)))
"""
    env = {**os.environ, "VITALITY_APP_PROFILE": "papers"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REST_API_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(
            "papers factory import isolation failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
