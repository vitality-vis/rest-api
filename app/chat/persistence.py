"""Assistant message terminal persistence for Chat turns."""

from __future__ import annotations

import logging

from repositories.supabase.chat_repository import ChatPersistenceError, save_message

logger = logging.getLogger(__name__)


def save_assistant_result(
    *,
    conversation_id: str,
    text: str,
    status: str,
    duration_ms: int,
    error_message: str | None,
    message_id: str | None,
    log: logging.Logger | None = None,
) -> None:
    """Persist assistant completed/failed. Log failures without raising."""
    active_logger = log or logger
    try:
        # created_at is when the reply finishes — leave unset so DB uses now().
        save_message(
            conversation_id=conversation_id,
            role="assistant",
            text=text,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
            message_id=message_id,
        )
    except ChatPersistenceError as error:
        active_logger.error("Could not save assistant chat message: %s", error)
