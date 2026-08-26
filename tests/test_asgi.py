"""ASGI composition and native /chat/v2 smoke tests."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from app.application import create_application
from app.asgi.lifespan import shutdown_bundle, startup_bundle
from app.chat.event_bridge import bridge_agent_events
from app.chat.events import RunCompleted, TextDelta
from app.chat.execution import AgentExecutionRuntime
from app.chat.models import ChatTurnRequest, PreparedChatTurn
from app.chat.sse import SSEEncoder, encode_keepalive_comment
from app.profiles import AppProfile


def _fake_logger():
    logger = logging.getLogger("vitality2-test-asgi")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        data_lines = [
            line.removeprefix("data: ")
            for line in block.split("\n")
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        events.append(json.loads("\n".join(data_lines)))
    return events


@pytest.fixture
def suppress_lifecycle(monkeypatch):
    monkeypatch.setattr(
        "app.profiles.papers.initialize_runtime", lambda **_kwargs: _fake_logger()
    )
    monkeypatch.setattr(
        "app.profiles.full.initialize_runtime", lambda **_kwargs: _fake_logger()
    )
    monkeypatch.setattr("app.asgi.lifespan.cached_data.init", lambda *_a, **_k: None)
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
    assert payload["agentRuntime"] is None


def test_papers_asgi_chat_routes_are_404(suppress_lifecycle):
    bundle = create_application(AppProfile.PAPERS)
    with TestClient(bundle.asgi_app) as client:
        assert client.post("/chat", json={"text": "hi"}).status_code == 404
        assert (
            client.post(
                "/chat/v2",
                json={"client_request_id": str(uuid4()), "text": "hi"},
            ).status_code
            == 404
        )


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
    assert payload["agentRuntime"]["ready"] is True
    assert payload["agentRuntime"]["accepting"] is True
    assert payload["agentRuntime"]["capacity"] >= 1
    assert "pendingCapacity" in payload["agentRuntime"]


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


def test_startup_bundle_agent_runtime_is_idempotent(suppress_lifecycle):
    bundle = create_application(AppProfile.FULL)
    first = AgentExecutionRuntime(max_workers=1, max_pending=1)
    bundle.agent_runtime = first
    startup_bundle(bundle)
    assert bundle.agent_runtime is first
    assert callable(bundle.flask_app.config.get("VITALITY_AGENT_RUNTIME_SNAPSHOT"))
    asyncio.run(first.shutdown())


def test_shutdown_bundle_awaits_agent_and_socketio(suppress_lifecycle):
    bundle = create_application(AppProfile.FULL)
    runtime = AsyncMock()
    runtime.shutdown = AsyncMock()
    sio = AsyncMock()
    sio.shutdown = AsyncMock()
    bundle.agent_runtime = runtime
    bundle.socketio = sio
    asyncio.run(shutdown_bundle(bundle))
    runtime.shutdown.assert_awaited_once()
    sio.shutdown.assert_awaited_once()
    assert bundle.agent_runtime is None
    assert bundle.flask_app.config.get("VITALITY_AGENT_RUNTIME_SNAPSHOT") is None


def test_agent_runtime_admission_and_snapshot():
    runtime = AgentExecutionRuntime(max_workers=1, max_pending=1)
    first = runtime.try_reserve()
    second = runtime.try_reserve()
    assert first is not None and second is not None
    assert runtime.try_reserve() is None
    snap = runtime.snapshot()
    assert snap.ready is True
    assert snap.accepting is False
    assert snap.capacity == 1
    assert snap.pending_capacity == 1
    assert snap.reserved == 2
    assert snap.rejected == 1
    first.release()
    snap = runtime.snapshot()
    assert snap.accepting is True
    assert snap.reserved == 1
    third = runtime.try_reserve()
    assert third is not None
    second.release()
    third.release()
    asyncio.run(runtime.shutdown())


def test_chat_v2_returns_503_when_admission_full(suppress_lifecycle, monkeypatch):
    async def fake_run(_request: Any) -> AsyncIterator[Any]:
        yield TextDelta(text="ok")

    monkeypatch.setattr("agents.agent_v2.runner.run", fake_run)
    bundle = create_application(AppProfile.FULL)
    with TestClient(bundle.asgi_app) as client:
        runtime = bundle.agent_runtime
        assert runtime is not None
        held = []
        while True:
            reservation = runtime.try_reserve()
            if reservation is None:
                break
            held.append(reservation)
        assert held
        response = client.post(
            "/chat/v2",
            json={
                "client_request_id": "req-over-capacity",
                "chat_id": "c-over",
                "text": "Hello",
            },
        )
        assert response.status_code == 503
        assert "capacity" in response.json()["detail"].lower()
        health = client.get("/health").json()
        assert health["agentRuntime"]["ready"] is True
        assert health["agentRuntime"]["accepting"] is False
        assert health["agentRuntime"]["rejected"] >= 1
        for reservation in held:
            reservation.release()


def test_chat_v2_requires_client_request_id(suppress_lifecycle):
    bundle = create_application(AppProfile.FULL)
    with TestClient(bundle.asgi_app) as client:
        before = client.get("/health").json()["agentRuntime"]
        response = client.post("/chat/v2", json={"text": "Hello"})
        assert response.status_code == 400
        assert response.json() == {"detail": "client_request_id is required"}
        # Failed validation must release admission.
        after = client.get("/health").json()["agentRuntime"]
    assert after["reserved"] == before["reserved"]
    assert after["active"] == before["active"]
    assert after["pending"] == before["pending"]


def test_chat_v2_sse_order_with_fake_runner(suppress_lifecycle, monkeypatch):
    async def fake_run(_request: Any) -> AsyncIterator[Any]:
        yield TextDelta(text="Hello")
        yield TextDelta(text="!")

    monkeypatch.setattr("agents.agent_v2.runner.run", fake_run)
    bundle = create_application(AppProfile.FULL)
    with TestClient(bundle.asgi_app) as client:
        response = client.post(
            "/chat/v2",
            json={
                "client_request_id": "req-1",
                "chat_id": "c1",
                "text": "Hello",
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["type"] for event in events] == [
        "run.started",
        "text.delta",
        "text.delta",
        "run.completed",
    ]
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert events[0]["data"]["clientRequestId"] == "req-1"
    assert events[1]["data"]["text"] == "Hello"
    assert events[2]["data"]["text"] == "!"


def test_sse_encoder_assigns_monotonic_seq():
    encoder = SSEEncoder()
    first = encoder.encode(TextDelta(text="a"))
    second = encoder.encode(RunCompleted(duration_ms=1))
    assert "id: 1\n" in first
    assert "id: 2\n" in second
    assert '"seq":1' in first
    assert '"seq":2' in second
    assert encode_keepalive_comment() == ": keepalive\n\n"


def test_bridge_persists_failed_when_queued_job_cancelled(monkeypatch):
    saved: dict[str, Any] = {}
    started = threading.Event()
    release = threading.Event()

    def fake_save(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", fake_save)

    runtime = AgentExecutionRuntime(max_workers=1, max_pending=2)

    def blocking_job() -> None:
        started.set()
        release.wait(timeout=5)

    blocker_reservation = runtime.try_reserve()
    assert blocker_reservation is not None
    blocker = runtime.submit(blocker_reservation, blocking_job)
    assert started.wait(timeout=2)

    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello",
            chat_id="queued-1",
            assistant_message_id="asst-queued",
            pipeline="v2",
        ),
        user_id="user-1",
        history=[],
    )

    async def never_run(_request: Any) -> AsyncIterator[Any]:
        if False:  # pragma: no cover
            yield TextDelta(text="")
        raise AssertionError("queued job should be cancelled before start")

    async def exercise() -> None:
        disconnected = asyncio.Event()
        chat_reservation = runtime.try_reserve()
        assert chat_reservation is not None

        async def is_disconnected() -> bool:
            return disconnected.is_set()

        async def consume() -> None:
            async for _event in bridge_agent_events(
                prepared,
                run_agent=never_run,
                runtime=runtime,
                reservation=chat_reservation,
                logger=logging.getLogger("test-bridge"),
                is_disconnected=is_disconnected,
            ):
                pass

        task = asyncio.create_task(consume())
        # Allow submit() so the chat job sits queued behind the blocker.
        await asyncio.sleep(0.05)
        disconnected.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(exercise())
    release.set()
    blocker.result(timeout=2)
    blocker_reservation.mark_completed()
    asyncio.run(runtime.shutdown())

    assert saved["conversation_id"] == "queued-1"
    assert saved["status"] == "failed"
    assert saved["message_id"] == "asst-queued"
    assert "disconnected" in (saved["error_message"] or "").lower()
