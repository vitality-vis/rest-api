"""Persist the Codex thread bound to each Vitality conversation."""

from __future__ import annotations

# Bump when thread-scoped instructions or dynamic tool schemas change.
_THREAD_SCHEMA_VERSION = "research-v10"


def get_thread_id(user_id: str | None, chat_id: str) -> str | None:
    """Return a current-schema thread for this owned conversation, if any."""
    if not user_id or not chat_id:
        return None
    from repositories.supabase.chat_repository import (
        clear_codex_thread_binding,
        get_codex_thread_binding,
    )

    binding = get_codex_thread_binding(conversation_id=chat_id, user_id=user_id)
    if binding is None:
        return None
    thread_id, schema_version = binding
    if schema_version == _THREAD_SCHEMA_VERSION:
        return thread_id
    clear_codex_thread_binding(
        conversation_id=chat_id, user_id=user_id, expected_thread_id=thread_id
    )
    return None


def put_thread_id(user_id: str | None, chat_id: str, thread_id: str) -> bool:
    """Bind a completed thread when the conversation is still unbound."""
    if not user_id or not chat_id or not thread_id:
        return False
    from repositories.supabase.chat_repository import bind_codex_thread

    return bind_codex_thread(
        conversation_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
        schema_version=_THREAD_SCHEMA_VERSION,
    )


def drop_thread_id(
    user_id: str | None, chat_id: str, expected_thread_id: str | None = None
) -> bool:
    """Conditionally forget a failed thread without clearing a newer binding."""
    if not user_id or not chat_id:
        return False
    from repositories.supabase.chat_repository import (
        clear_codex_thread_binding,
        get_codex_thread_binding,
    )

    thread_id = expected_thread_id
    if thread_id is None:
        binding = get_codex_thread_binding(conversation_id=chat_id, user_id=user_id)
        thread_id = binding[0] if binding else None
    if not thread_id:
        return False
    return clear_codex_thread_binding(
        conversation_id=chat_id, user_id=user_id, expected_thread_id=thread_id
    )
