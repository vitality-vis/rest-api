"""Persistence for authenticated chat conversations and messages."""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10
DATABASE_REQUEST_ATTEMPTS = 2

logger = logging.getLogger(__name__)

# When created_at ties, prefer user before assistant ("user" > "assistant").
_MESSAGE_ORDER = "created_at.asc,role.desc,id.asc"


class ChatPersistenceError(RuntimeError):
    """Raised when the chat database cannot complete a required operation."""


class ConversationOwnershipError(ChatPersistenceError):
    """Raised when a conversation UUID belongs to a different user."""


class ChatAuthenticationError(ChatPersistenceError):
    """Raised when Supabase rejects the user's bearer token."""


class ChatMessageConflictError(ChatPersistenceError):
    """Raised when an idempotency key is reused for different content."""


def _sort_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Stable local sort matching _MESSAGE_ORDER (fixes equal-timestamp turns)."""
    role_rank = {"user": 0, "assistant": 1}

    def key(message: dict[str, object]) -> tuple[str, int, str]:
        created_at = str(message.get("created_at") or "")
        role = str(message.get("role") or "")
        message_id = str(message.get("id") or "")
        return (created_at, role_rank.get(role, 9), message_id)

    return sorted(messages, key=key)


def load_user_conversations(*, user_id: str) -> list[dict[str, object]]:
    """Load every conversation and message belonging to one verified user."""
    conversations_response = _request(
        "GET",
        "chat_conversations",
        params={
            "user_id": f"eq.{_normalise_uuid(user_id)}",
            "select": "id,title,is_closed,created_at,updated_at",
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
            "select": "id,conversation_id,role,content,context,status,error_message,created_at",
            "order": _MESSAGE_ORDER,
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
            "is_closed": conversation["is_closed"],
            "messages": _sort_messages(messages_by_conversation[conversation["id"]]),
        }
        for conversation in conversations
    ]


def _messages_to_history(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered_messages = _sort_messages(messages)
    turns: list[dict[str, object]] = []
    for message in ordered_messages:
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
            turn: dict[str, object] = {"role": str(role), "content": text}
            if isinstance(message.get("context"), dict) and message["context"]:
                turn["context"] = message["context"]
            turns.append(turn)
    return turns


def load_completed_history(
    *,
    conversation_id: str,
    user_id: str,
    exclude_message_id: str | None = None,
    ownership_verified: bool = False,
) -> list[dict[str, object]]:
    """Return completed turns, optionally reusing an earlier ownership check."""
    conversation_id = _normalise_uuid(conversation_id)
    user_id = _normalise_uuid(user_id)
    if not ownership_verified:
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
            raise ConversationOwnershipError(
                "Chat conversation does not belong to this user"
            )

    params = {
        "conversation_id": f"eq.{conversation_id}",
        "status": "eq.completed",
        "select": "role,content,context,created_at,id",
        "order": _MESSAGE_ORDER,
    }
    if exclude_message_id:
        params["id"] = f"neq.{_normalise_uuid(exclude_message_id)}"
    messages_response = _request("GET", "chat_messages", params=params)
    if messages_response.status_code != 200:
        raise ChatPersistenceError("Could not load chat history")
    return _messages_to_history(list(messages_response.json()))


def prepare_chat_turn(
    *,
    conversation_id: str,
    access_token: str,
    title: str,
    text: str,
    message_id: str | None,
    created_at: str | None,
    context: dict[str, object] | None,
    include_history: bool,
    client_request_id: str | None = None,
) -> tuple[str, list[dict[str, object]], bool]:
    """Authenticate and atomically prepare one persisted chat turn."""
    conversation_id = _normalise_uuid(conversation_id)
    message_uuid = _normalise_uuid(message_id) if message_id else str(uuid4())
    request_uuid = _normalise_uuid(client_request_id) if client_request_id else message_uuid
    response = _request(
        "POST",
        "rpc/prepare_chat_turn",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "p_conversation_id": conversation_id,
            "p_title": title,
            "p_message_id": message_uuid,
            "p_message_created_at": created_at,
            "p_text": text,
            "p_context": context or {},
            "p_include_history": include_history,
            "p_client_request_id": request_uuid,
        },
    )
    if response.status_code == 401:
        raise ChatAuthenticationError("Invalid or expired access token")
    if response.status_code == 409:
        raise ChatMessageConflictError(
            "This chat request was already submitted or its message ID conflicts"
        )
    if response.status_code == 403:
        raise ConversationOwnershipError(
            "Chat conversation does not belong to this user"
        )
    if response.status_code == 404 and _is_missing_prepare_rpc(response):
        logger.warning(
            "prepare_chat_turn RPC is not installed; using compatibility path"
        )
        from repositories.supabase.auth import (
            SupabaseAuthenticationError,
            verify_access_token,
        )

        try:
            user_id = verify_access_token(access_token)
        except SupabaseAuthenticationError as error:
            raise ChatAuthenticationError(str(error)) from error
        ensure_conversation(
            conversation_id=conversation_id, user_id=user_id, title=title
        )
        history = (
            load_completed_history(
                conversation_id=conversation_id,
                user_id=user_id,
                exclude_message_id=message_uuid,
                ownership_verified=True,
            )
            if include_history
            else []
        )
        inserted = save_message(
            conversation_id=conversation_id,
            role="user",
            text=text,
            message_id=message_uuid,
            created_at=created_at,
            context=context,
        )
        return user_id, history, not inserted
    if response.status_code not in {200, 201}:
        raise ChatPersistenceError("Could not prepare chat turn")
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict) or not isinstance(payload.get("user_id"), str):
        raise ChatPersistenceError("Chat turn preparation returned no user identity")
    user_id = _normalise_uuid(payload["user_id"])
    history = payload.get("history", [])
    duplicate = payload.get("duplicate", False) is True
    return (
        user_id,
        _messages_to_history(history if isinstance(history, list) else []),
        duplicate,
    )


def _is_missing_prepare_rpc(response: requests.Response) -> bool:
    try:
        payload = response.json()
    except requests.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("code") == "PGRST202"


def _request(method: str, path: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    headers = service_role_headers(settings)
    headers.update(kwargs.pop("headers", {}))
    url = f"{settings.url}/rest/v1/{path}"
    last_error: requests.RequestException | None = None
    for attempt in range(1, DATABASE_REQUEST_ATTEMPTS + 1):
        try:
            return requests.request(
                method,
                url,
                headers=headers,
                timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except requests.RequestException as error:
            last_error = error
            logger.warning(
                "Chat database %s %s failed (attempt %s/%s): %s",
                method,
                path,
                attempt,
                DATABASE_REQUEST_ATTEMPTS,
                error,
            )
    assert last_error is not None
    raise ChatPersistenceError("Could not reach the chat database") from last_error


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
    is_closed: bool = False,
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
            "is_closed": is_closed,
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


def set_conversation_closed(*, conversation_id: str, user_id: str, is_closed: bool) -> None:
    """Persist the tab-visibility preference for one owned conversation."""
    conversation_id = _normalise_uuid(conversation_id)
    user_id = _normalise_uuid(user_id)
    response = _request(
        "PATCH",
        "chat_conversations",
        params={"id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
        headers={"Prefer": "return=representation"},
        json={"is_closed": is_closed},
    )
    if response.status_code not in {200, 204}:
        raise ChatPersistenceError("Could not update chat tab state")
    if response.status_code == 200 and not response.json():
        raise ConversationOwnershipError("Chat conversation does not belong to this user")


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
    context: dict[str, object] | None = None,
) -> bool:
    """Insert a message; return False only for an exact idempotent duplicate."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    if status not in {"streaming", "completed", "failed"}:
        raise ValueError("Unsupported message status")

    stable_message_id = _normalise_uuid(message_id) if message_id else str(uuid4())
    payload: dict[str, object] = {
        "conversation_id": _normalise_uuid(conversation_id),
        "role": role,
        "content": [{"type": "text", "text": text}],
        "status": status,
        "id": stable_message_id,
    }
    if created_at is not None:
        payload["created_at"] = created_at
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if error_message is not None:
        payload["error_message"] = error_message
    if context:
        payload["context"] = context

    response = _request(
        "POST",
        "chat_messages",
        headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
        json=payload,
    )
    if response.status_code not in {200, 201}:
        raise ChatPersistenceError("Could not save chat message")
    representation = response.json()
    if representation:
        return True
    # An empty representation means ON CONFLICT ignored an existing row. Verify
    # immutable identity fields so idempotency never masks a different message.
    if not representation:
        existing = _request(
            "GET",
            "chat_messages",
            params={
                "id": f"eq.{payload['id']}",
                "select": "conversation_id,role,content,context,status,duration_ms,error_message",
            },
        )
        if existing.status_code != 200 or not existing.json():
            raise ChatPersistenceError("Could not verify duplicate chat message")
        row = existing.json()[0]
        expected = {
            "conversation_id": payload["conversation_id"],
            "role": payload["role"],
            "content": payload["content"],
            "context": payload.get("context") or {},
            "status": payload["status"],
            "duration_ms": payload.get("duration_ms"),
            "error_message": payload.get("error_message"),
        }
        actual = {key: row.get(key) for key in expected}
        if actual != expected:
            raise ChatMessageConflictError(
                "Chat message ID is already used by a different message"
            )
        return False
    return True


__all__ = [
    "ChatAuthenticationError",
    "ChatPersistenceError",
    "ChatMessageConflictError",
    "ConversationOwnershipError",
    "ensure_conversation",
    "load_completed_history",
    "load_user_conversations",
    "prepare_chat_turn",
    "save_message",
    "set_conversation_closed",
]
