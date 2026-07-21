"""Persistence for authenticated chat conversations and messages."""

from __future__ import annotations

from uuid import UUID

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10


class ChatPersistenceError(RuntimeError):
    """Raised when the chat database cannot complete a required operation."""


class ConversationOwnershipError(ChatPersistenceError):
    """Raised when a conversation UUID belongs to a different user."""


def load_user_conversations(*, user_id: str) -> list[dict[str, object]]:
    """Load every conversation and message belonging to one verified user."""
    conversations_response = _request(
        "GET",
        "chat_conversations",
        params={
            "user_id": f"eq.{_normalise_uuid(user_id)}",
            "select": "id,title,created_at,updated_at",
            "order": "updated_at.desc",
        },
    )
    if conversations_response.status_code != 200:
        raise ChatPersistenceError("Could not load chat conversations")

    conversations = conversations_response.json()
    if not conversations:
        return []

    conversation_ids = [_normalise_uuid(conversation["id"]) for conversation in conversations]
    messages_response = _request(
        "GET",
        "chat_messages",
        params={
            "conversation_id": f"in.({','.join(conversation_ids)})",
            "select": "id,conversation_id,role,content,status,error_message,created_at",
            "order": "created_at.asc,id.asc",
        },
    )
    if messages_response.status_code != 200:
        raise ChatPersistenceError("Could not load chat messages")

    messages_by_conversation: dict[str, list[dict[str, object]]] = {
        conversation_id: [] for conversation_id in conversation_ids
    }
    for message in messages_response.json():
        conversation_id = message.get("conversation_id")
        if conversation_id in messages_by_conversation:
            messages_by_conversation[conversation_id].append(message)

    return [
        {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": messages_by_conversation[conversation["id"]],
        }
        for conversation in conversations
    ]


def load_completed_history(*, conversation_id: str, user_id: str) -> list[dict[str, str]]:
    """Return ordered, user-visible completed turns for one owned conversation."""
    conversation_id = _normalise_uuid(conversation_id)
    user_id = _normalise_uuid(user_id)
    ownership_response = _request(
        "GET",
        "chat_conversations",
        params={
            "id": f"eq.{conversation_id}",
            "user_id": f"eq.{user_id}",
            "select": "id",
        },
    )
    if ownership_response.status_code != 200:
        raise ChatPersistenceError("Could not verify chat conversation ownership")
    if not ownership_response.json():
        raise ConversationOwnershipError("Chat conversation does not belong to this user")

    messages_response = _request(
        "GET",
        "chat_messages",
        params={
            "conversation_id": f"eq.{conversation_id}",
            "status": "eq.completed",
            "select": "role,content,created_at,id",
            "order": "created_at.asc,id.asc",
        },
    )
    if messages_response.status_code != 200:
        raise ChatPersistenceError("Could not load chat history")

    turns: list[dict[str, str]] = []
    for message in messages_response.json():
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, list):
            continue
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
        if text:
            turns.append({"role": role, "content": text})
    return turns


def _request(method: str, path: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    headers = service_role_headers(settings)
    headers.update(kwargs.pop("headers", {}))
    try:
        return requests.request(
            method,
            f"{settings.url}/rest/v1/{path}",
            headers=headers,
            timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as error:
        raise ChatPersistenceError("Could not reach the chat database") from error


def _normalise_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ChatPersistenceError("chat_id must be a UUID") from error


def ensure_conversation(
    *,
    conversation_id: str,
    user_id: str,
    title: str = "New chat",
    created_at: str | None = None,
    updated_at: str | None = None,
) -> str:
    """Create a user-owned conversation, or prove the existing one is theirs."""
    conversation_id = _normalise_uuid(conversation_id)
    response = _request(
        "GET",
        "chat_conversations",
        params={"id": f"eq.{conversation_id}", "select": "id,user_id"},
    )
    if response.status_code != 200:
        raise ChatPersistenceError("Could not look up chat conversation")

    existing = response.json()
    if existing:
        if existing[0].get("user_id") != user_id:
            raise ConversationOwnershipError("Chat conversation does not belong to this user")
        return conversation_id

    create_response = _request(
        "POST",
        "chat_conversations",
        headers={"Prefer": "return=representation"},
        json={
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            **({"created_at": created_at} if created_at else {}),
            **({"updated_at": updated_at} if updated_at else {}),
        },
    )
    if create_response.status_code in {200, 201}:
        return conversation_id

    # A simultaneous request may have created the same UUID. Re-check ownership;
    # any other failure remains a persistence error rather than falling back to guest mode.
    if create_response.status_code == 409:
        return ensure_conversation(conversation_id=conversation_id, user_id=user_id)
    raise ChatPersistenceError("Could not create chat conversation")


def save_message(
    *,
    conversation_id: str,
    role: str,
    text: str,
    status: str = "completed",
    duration_ms: int | None = None,
    error_message: str | None = None,
    message_id: str | None = None,
    created_at: str | None = None,
) -> None:
    """Append one immutable user-visible text message to a conversation."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    if status not in {"streaming", "completed", "failed"}:
        raise ValueError("Unsupported message status")

    payload: dict[str, object] = {
        "conversation_id": _normalise_uuid(conversation_id),
        "role": role,
        "content": [{"type": "text", "text": text}],
        "status": status,
    }
    if message_id is not None:
        payload["id"] = _normalise_uuid(message_id)
    if created_at is not None:
        payload["created_at"] = created_at
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if error_message is not None:
        payload["error_message"] = error_message

    response = _request(
        "POST",
        "chat_messages",
        headers={"Prefer": "resolution=ignore-duplicates"},
        json=payload,
    )
    if response.status_code not in {200, 201}:
        raise ChatPersistenceError("Could not save chat message")


__all__ = [
    "ChatPersistenceError",
    "ConversationOwnershipError",
    "ensure_conversation",
    "load_completed_history",
    "load_user_conversations",
    "save_message",
]
