"""Flask-free Chat application service tests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.chat.events import ChatEvent, RunCompleted, RunFailed, TextDelta
from app.chat.models import (
    ChatConflictError,
    ChatTurnRequest,
    ChatUnauthorizedError,
    ChatValidationError,
    PreparedChatTurn,
)
from app.chat.request_service import build_chat_turn_request, prepare_chat_turn
from app.chat.runner_adapter import build_agent_request
from app.chat.service import FALLBACK_TEXT, iter_chat_turn_events


async def _fake_runner(_request: Any) -> AsyncIterator[str]:
    yield "Hello"
    yield ", world"


async def _failing_runner(_request: Any) -> AsyncIterator[str]:
    yield "partial "
    raise RuntimeError("boom")
    if False:  # pragma: no cover
        yield ""


async def _collect_events(prepared: PreparedChatTurn, run_agent) -> list[ChatEvent]:
    return [
        event
        async for event in iter_chat_turn_events(
            prepared,
            run_agent=run_agent,
            logger=logging.getLogger("test-chat-service"),
        )
    ]


def test_build_chat_turn_request_rejects_empty_text():
    with pytest.raises(ChatValidationError, match="Please Input Your Text"):
        build_chat_turn_request({"text": "   "})


def test_build_chat_turn_request_rejects_malformed_client_request_id():
    with pytest.raises(ChatValidationError, match="client_request_id must be a UUID"):
        build_chat_turn_request({"text": "Hi", "client_request_id": "req-1"})


def test_build_chat_turn_request_normalises_client_request_uuid():
    request = build_chat_turn_request(
        {
            "text": "Hi",
            "client_request_id": "CCCCCCCC-CCCC-4CCC-8CCC-CCCCCCCCCCCC",
        }
    )
    assert request.client_request_id == "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def test_build_chat_turn_request_guest_history():
    request = build_chat_turn_request(
        {
            "text": "Hi",
            "chat_id": "c1",
            "history": [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "reply"},
            ],
        }
    )
    assert request.text == "Hi"
    assert request.chat_id == "c1"
    assert request.guest_history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
    ]


def test_prepare_chat_turn_requires_authentication():
    request = ChatTurnRequest(
        text="Hi",
        chat_id="guest-1",
        guest_history=[{"role": "user", "content": "before"}],
    )
    with pytest.raises(ChatUnauthorizedError, match="Unauthorized"):
        prepare_chat_turn(request)


def test_prepare_authenticated_turn_uses_one_atomic_repository_call(monkeypatch):
    request = ChatTurnRequest(
        text="Hi",
        chat_id="11111111-1111-1111-1111-111111111111",
        user_message_id="22222222-2222-2222-2222-222222222222",
        authorization_header="Bearer signed-token",
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "app.chat.request_service._jwt_subject_hint",
        lambda _token: "33333333-3333-3333-3333-333333333333",
    )
    monkeypatch.setattr(
        "app.chat.request_service.get_thread_id",
        lambda user_id, chat_id: "thread-1",
    )

    def fake_prepare(**kwargs):
        seen.update(kwargs)
        return "33333333-3333-3333-3333-333333333333", [], False

    monkeypatch.setattr(
        "app.chat.request_service.prepare_persisted_chat_turn", fake_prepare
    )

    prepared = prepare_chat_turn(request)

    assert prepared.user_id == "33333333-3333-3333-3333-333333333333"
    assert prepared.has_codex_thread is True
    assert prepared.history == []
    assert seen["access_token"] == "signed-token"
    assert seen["include_history"] is False


def test_prepare_v2_turn_includes_history_despite_codex_mapping(monkeypatch):
    request = ChatTurnRequest(
        text="Hi",
        chat_id="11111111-1111-1111-1111-111111111111",
        user_message_id="22222222-2222-2222-2222-222222222222",
        authorization_header="Bearer signed-token",
    )
    seen: dict[str, Any] = {}
    monkeypatch.setattr("app.chat.request_service.config.CHAT_RUNTIME", "v2")
    monkeypatch.setattr(
        "app.chat.request_service._jwt_subject_hint",
        lambda _token: "33333333-3333-3333-3333-333333333333",
    )
    monkeypatch.setattr(
        "app.chat.request_service.get_thread_id", lambda _user_id, _chat_id: "thread-1"
    )

    def fake_prepare(**kwargs):
        seen.update(kwargs)
        return (
            "33333333-3333-3333-3333-333333333333",
            [{"role": "user", "content": "Earlier"}],
            False,
        )

    monkeypatch.setattr(
        "app.chat.request_service.prepare_persisted_chat_turn", fake_prepare
    )

    prepared = prepare_chat_turn(request)

    assert seen["include_history"] is True
    assert prepared.has_codex_thread is False
    assert prepared.history == [{"role": "user", "content": "Earlier"}]


def test_build_agent_request_v2():
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Find papers",
            chat_id="c1",
            effort="medium",
            pipeline="v2",
            requested_mode="search",
            paper_ids=["p1"],
        ),
        user_id=None,
        history=[{"role": "user", "content": "prev"}],
    )
    agent_request = build_agent_request(prepared)
    assert agent_request.text == "Find papers"
    assert agent_request.chat_id == "c1"
    assert agent_request.history == [{"role": "user", "content": "prev"}]
    assert agent_request.effort == "medium"
    assert agent_request.selected_paper_ids == ["p1"]
    assert agent_request.requested_mode == "search"


def test_iter_chat_turn_events_without_flask_context():
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(text="Hello", chat_id="t1"),
        user_id=None,
        history=[],
    )
    events = asyncio.run(_collect_events(prepared, _fake_runner))
    assert [e for e in events if isinstance(e, TextDelta)] == [
        TextDelta(text="Hello"),
        TextDelta(text=", world"),
    ]
    assert isinstance(events[-1], RunCompleted)
    assert events[-1].duration_ms >= 0


def test_iter_chat_turn_events_fallback_on_runner_error():
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(text="Hello", chat_id="t2"),
        user_id=None,
        history=[],
    )
    events = asyncio.run(_collect_events(prepared, _failing_runner))
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["partial ", FALLBACK_TEXT]
    assert isinstance(events[-1], RunFailed)
    assert events[-1].message == FALLBACK_TEXT
    assert "boom" not in events[-1].message


def test_authenticated_turn_persists_assistant_on_completion(monkeypatch):
    saved: dict[str, Any] = {}

    def fake_save_assistant_result(**kwargs):
        saved.update(kwargs)
        return True

    monkeypatch.setattr(
        "app.chat.service.save_assistant_result", fake_save_assistant_result
    )
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello",
            chat_id="auth-1",
            assistant_message_id="asst-1",
        ),
        user_id="user-1",
        history=[],
    )
    events = asyncio.run(_collect_events(prepared, _fake_runner))
    assert isinstance(events[-1], RunCompleted)
    assert saved["conversation_id"] == "auth-1"
    assert saved["text"] == "Hello, world"
    assert saved["status"] == "completed"
    assert saved["message_id"] == "asst-1"
    assert saved["error_message"] is None


def test_authenticated_turn_persists_failed_on_client_disconnect(monkeypatch):
    """aclose mid-stream must still save assistant as failed (old finally behavior)."""
    saved: dict[str, Any] = {}

    def fake_save_assistant_result(**kwargs):
        saved.update(kwargs)
        return True

    monkeypatch.setattr(
        "app.chat.service.save_assistant_result", fake_save_assistant_result
    )
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello",
            chat_id="auth-disconnect",
            assistant_message_id="asst-disconnect",
        ),
        user_id="user-1",
        history=[],
    )

    async def _disconnect_after_first_delta() -> None:
        agen = iter_chat_turn_events(
            prepared,
            run_agent=_fake_runner,
            logger=logging.getLogger("test-chat-service"),
        ).__aiter__()
        first = await agen.__anext__()
        assert isinstance(first, TextDelta)
        assert first.text == "Hello"
        await agen.aclose()

    asyncio.run(_disconnect_after_first_delta())
    assert saved["conversation_id"] == "auth-disconnect"
    assert saved["text"] == "Hello"
    assert saved["status"] == "failed"
    assert saved["message_id"] == "asst-disconnect"
    assert saved["error_message"] is None


def test_prepare_duplicate_request_returns_conflict(monkeypatch):
    request = ChatTurnRequest(
        text="Hi",
        chat_id="11111111-1111-1111-1111-111111111111",
        user_message_id="22222222-2222-2222-2222-222222222222",
        client_request_id="44444444-4444-4444-4444-444444444444",
        authorization_header="Bearer signed-token",
    )
    monkeypatch.setattr(
        "app.chat.request_service.prepare_persisted_chat_turn",
        lambda **_kwargs: (
            "33333333-3333-3333-3333-333333333333", [], True
        ),
    )
    monkeypatch.setattr(
        "app.chat.request_service._jwt_subject_hint", lambda _token: None
    )

    with pytest.raises(ChatConflictError, match="already been submitted"):
        prepare_chat_turn(request)


def test_codex_persistence_failure_drops_thread_mapping(monkeypatch):
    dropped: list[tuple[str | None, str]] = []
    monkeypatch.setattr("app.chat.service.config.CHAT_RUNTIME", "codex")
    monkeypatch.setattr(
        "app.chat.service.save_assistant_result", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        "app.codex.threads.drop_thread_id",
        lambda user_id, chat_id: dropped.append((user_id, chat_id)),
    )
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello", chat_id="auth-codex", assistant_message_id="asst-codex"
        ),
        user_id="user-codex",
        history=[],
        has_codex_thread=True,
    )

    events = asyncio.run(_collect_events(prepared, _fake_runner))

    assert dropped == [("user-codex", "auth-codex")]
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_code == "persistence_failed"


def test_completion_waits_for_assistant_persistence(monkeypatch):
    monkeypatch.setattr(
        "app.chat.service.save_assistant_result", lambda **_kwargs: False
    )
    prepared = PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello", chat_id="auth-2", assistant_message_id="asst-2"
        ),
        user_id="user-1",
        history=[],
    )
    events = asyncio.run(_collect_events(prepared, _fake_runner))
    assert not any(isinstance(event, RunCompleted) for event in events)
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_code == "persistence_failed"
