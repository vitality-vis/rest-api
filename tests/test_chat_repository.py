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
