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

    def _with_ids(self, data: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"trace_id": self.trace_id, **data}
        if self.chat_id:
            payload["chat_id"] = self.chat_id
        if self.user_message_id:
            payload["user_message_id"] = self.user_message_id
        if self.assistant_message_id:
            payload["assistant_message_id"] = self.assistant_message_id
        return payload

    def log_decision(
        self,
        *,
        decision: Literal["talk", "search", "synthesis", "clarify"],
        search_intent: dict[str, Any] | None,
        query: str,
        effort: Literal["low", "medium", "high"],
        router_prompt: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "decision": decision,
            "search_intent": search_intent,
            "query": query,
            "effort": effort,
        }
        if router_prompt is not None:
            data["router_prompt"] = router_prompt
        log_structured("search_v2.decision", self._with_ids(data))

    def log_synthesis_evidence_plan(
        self,
        *,
        metadata_paper_ids: list[str],
        file_search_paper_ids: list[str],
        use_file_search: bool,
    ) -> None:
        log_structured(
            "agent_v2.synthesis_evidence_plan",
            self._with_ids(
                {
                    "metadata_paper_ids": metadata_paper_ids,
                    "file_search_paper_ids": file_search_paper_ids,
                    "use_file_search": use_file_search,
                }
            ),
        )

    def log_synthesis_payload(
        self,
        *,
        mode: Literal["metadata", "file_search"],
        input_text: str,
        vector_store_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Record the exact prompt sent to Azure Responses for synthesis."""
        data: dict[str, Any] = {
            "mode": mode,
            "input_text": input_text,
            "input_text_length": len(input_text),
        }
        if vector_store_id is not None:
            data["vector_store_id"] = vector_store_id
        if filters is not None:
            data["filters"] = filters
        log_structured("agent_v2.synthesis_payload", self._with_ids(data))

    def log_synthesis_scope_check(
        self,
        *,
        allowed_file_ids: list[str],
        cited_file_ids: list[str],
        unexpected_file_ids: list[str],
    ) -> None:
        """Record the best-effort citation scope check after File Search."""
        log_structured(
            "agent_v2.synthesis_scope_check",
            self._with_ids(
                {
                    "allowed_file_ids": allowed_file_ids,
                    "cited_file_ids": cited_file_ids,
                    "unexpected_file_ids": unexpected_file_ids,
                    "scope_warning": bool(unexpected_file_ids),
                }
            ),
        )
