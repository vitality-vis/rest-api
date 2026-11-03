"""Focused tests for idempotent chat persistence."""

from __future__ import annotations

from typing import Any

import pytest

from repositories.supabase.chat_repository import (
    ChatMessageConflictError,
    save_message,
)


class _Response:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _ids():
    return (
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    )

def test_save_message_accepts_exact_duplicate(monkeypatch):
    conversation_id, message_id = _ids()
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "POST":
            return _Response(200, [])
        return _Response(200, [{
            "conversation_id": conversation_id,
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "context": {},
            "status": "completed",
            "duration_ms": None,
            "error_message": None,
        }])

    monkeypatch.setattr("repositories.supabase.chat_repository._request", fake_request)
    assert save_message(
        conversation_id=conversation_id, role="user", text="Hello",
        message_id=message_id,
    ) is False
    assert [call[0] for call in calls] == ["POST", "GET"]


def test_save_message_rejects_mismatched_duplicate(monkeypatch):
    conversation_id, message_id = _ids()

    def fake_request(method, _path, **_kwargs):
        if method == "POST":
            return _Response(200, [])
        return _Response(200, [{
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": [{"type": "text", "text": "Different"}],
            "context": {},
            "status": "completed",
            "duration_ms": None,
            "error_message": None,
        }])

    monkeypatch.setattr("repositories.supabase.chat_repository._request", fake_request)
    with pytest.raises(ChatMessageConflictError):
        save_message(
            conversation_id=conversation_id, role="user", text="Hello",
            message_id=message_id,
        )


def test_save_message_allocates_id_before_retry(monkeypatch):
    conversation_id, _ = _ids()
    payload_ids = []

    def fake_request(_method, _path, **kwargs):
        payload_ids.append(kwargs["json"]["id"])
        return _Response(201, [kwargs["json"]])

    monkeypatch.setattr("repositories.supabase.chat_repository._request", fake_request)
    assert save_message(
        conversation_id=conversation_id, role="assistant", text="Hello"
    ) is True
    assert len(payload_ids) == 1
    assert payload_ids[0]


def test_prepare_chat_turn_propagates_and_validates_project(monkeypatch):
    from repositories.supabase.chat_repository import prepare_chat_turn

    conversation_id, message_id = _ids()
    project_id = "33333333-3333-3333-3333-333333333333"
    user_id = "44444444-4444-4444-4444-444444444444"
    seen = {}

    def fake_request(_method, _path, **kwargs):
        seen.update(kwargs["json"])
        return _Response(200, {"user_id": user_id, "project_id": project_id, "history": [], "duplicate": False})

    monkeypatch.setattr("repositories.supabase.chat_repository._request", fake_request)
    result = prepare_chat_turn(
        conversation_id=conversation_id, project_id=project_id,
        access_token="token", title="Chat", text="Hello",
        message_id=message_id, created_at=None, context=None,
        include_history=True,
    )
    assert result == (user_id, project_id, [], False)
    assert seen["p_project_id"] == project_id


def test_prepare_chat_turn_missing_canonical_rpc_fails_closed(monkeypatch):
    from repositories.supabase.chat_repository import ChatPersistenceError, prepare_chat_turn

    conversation_id, message_id = _ids()
    monkeypatch.setattr(
        "repositories.supabase.chat_repository._request",
        lambda *_args, **_kwargs: _Response(404, {"code": "PGRST202"}),
    )
    with pytest.raises(ChatPersistenceError, match="Project-aware"):
        prepare_chat_turn(
            conversation_id=conversation_id,
            project_id="33333333-3333-3333-3333-333333333333",
            access_token="token", title="Chat", text="Hello",
            message_id=message_id, created_at=None, context=None,
            include_history=True,
        )


def test_ensure_conversation_rejects_project_mismatch(monkeypatch):
    from repositories.supabase.chat_repository import ConversationOwnershipError, ensure_conversation

    conversation_id, _ = _ids()
    user_id = "44444444-4444-4444-4444-444444444444"
    monkeypatch.setattr(
        "repositories.supabase.chat_repository._request",
        lambda *_args, **_kwargs: _Response(200, [{
            "id": conversation_id, "user_id": user_id,
            "project_id": "55555555-5555-5555-5555-555555555555",
        }]),
    )
    with pytest.raises(ConversationOwnershipError, match="different project"):
        ensure_conversation(
            conversation_id=conversation_id, user_id=user_id,
            project_id="33333333-3333-3333-3333-333333333333",
        )
