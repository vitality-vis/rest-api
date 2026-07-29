"""Minimal structured decision logging for search v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from logger_config import log_structured


@dataclass(frozen=True)
class SearchV2Trace:
    """Own the response trace ID and emit the sole search-v2 log event."""

    trace_id: str
    chat_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        trace_id: str | None = None,
        chat_id: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> "SearchV2Trace":
        return cls(
            trace_id=trace_id or uuid4().hex,
            chat_id=chat_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )

    def log_decision(
        self,
        *,
        decision: Literal["search", "other"],
        search_intent: dict[str, Any] | None,
        query: str,
        effort: Literal["low", "medium", "high"],
    ) -> None:
        data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "decision": decision,
            "search_intent": search_intent,
            "query": query,
            "effort": effort,
        }
        if self.chat_id:
            data["chat_id"] = self.chat_id
        if self.user_message_id:
            data["user_message_id"] = self.user_message_id
        if self.assistant_message_id:
            data["assistant_message_id"] = self.assistant_message_id
        log_structured(
            "search_v2.decision",
            data,
        )
