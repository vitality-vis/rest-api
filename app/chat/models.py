"""Transport-independent Chat turn request models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChatPipeline = Literal["v2"]
ChatEffort = Literal["low", "medium", "high"]
ChatMode = Literal["auto", "chat", "search", "synthesis"]


class ChatDomainError(Exception):
    """Domain failure mapped to an HTTP status by the transport layer."""

    status_code: int = 400

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.message = message


class ChatValidationError(ChatDomainError):
    status_code = 400


class ChatUnauthorizedError(ChatDomainError):
    status_code = 401


class ChatForbiddenError(ChatDomainError):
    status_code = 403


class ChatUnavailableError(ChatDomainError):
    status_code = 503


@dataclass(frozen=True)
class ChatTurnRequest:
    """Validated Chat turn input, independent of Flask/FastAPI."""

    text: str
    chat_id: str = "default"
    title: str = "New chat"
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    message_created_at: str | None = None
    effort: ChatEffort = "low"
    model: str | None = None
    message_context: dict[str, object] = field(default_factory=dict)
    guest_history: list[dict[str, str]] = field(default_factory=list)
    authorization_header: str | None = None
    pipeline: ChatPipeline = "v2"
    max_text_length: int | None = None
    trace_id: str | None = None
    client_request_id: str | None = None
    agent_run_id: str | None = None
    # v2-only fields
    requested_mode: ChatMode = "auto"
    paper_ids: list[str] = field(default_factory=list)
    advanced: Any | None = None


@dataclass(frozen=True)
class PreparedChatTurn:
    """Chat turn after auth, history load, and user-message persistence."""

    request: ChatTurnRequest
    user_id: str | None
    history: list[dict[str, str]]
