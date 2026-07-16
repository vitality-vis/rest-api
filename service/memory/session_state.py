"""
Centralized session state for the research assistant.
Holds the global SESSIONS dict and helpers so retrieval/application services
and agent orchestration modules can share session access without circular imports.
"""
from typing import Optional

# Single global store: chat_id -> session dict
SESSIONS: dict = {}


def get_session(chat_id: str) -> Optional[dict]:
    """Return the session dict for chat_id, or None if not found."""
    return SESSIONS.get(chat_id)


def save_session(chat_id: str, session: dict) -> None:
    """Store (or replace) the session for chat_id."""
    SESSIONS[chat_id] = session
