"""Minimal structured decision logging for search v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from logger_config import log_structured

from .models import RetrievalPlan


@dataclass(frozen=True)
class SearchV2Trace:
    """Own the response trace ID used by search-v2 structured events."""

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
        response_mode: Literal["papers", "grounded_answer"] | None = None,
        decision_status: Literal[
            "explicit_mode",
            "model_decision",
            "json_parse_failed",
            "validation_failed",
            "router_error",
            "incomplete_search_decision",
        ] = "model_decision",
        router_prompt: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "decision": decision,
            "search_intent": search_intent,
            "query": query,
            "effort": effort,
            "response_mode": response_mode,
            "decision_status": decision_status,
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

    def log_retrieval_execution(
        self,
        *,
        plan: RetrievalPlan,
        retrieval_counts: dict[str, int],
        retrieval_failures: dict[str, str],
        rerank_status: Literal["skipped", "complete", "failed"],
        status: Literal["complete", "partial", "failed"],
    ) -> None:
        """Record the validated plan and each retrieval arm's outcome."""
        actions = [
            action.model_dump() if hasattr(action, "model_dump") else action.dict()
            for action in plan.actions
        ]
        log_structured(
            "search_v2.retrieval_execution",
            self._with_ids(
                {
                    "plan_source": plan.source,
                    "actions": actions,
                    "retrieval_counts": retrieval_counts,
                    "retrieval_failures": retrieval_failures,
                    "rerank_status": rerank_status,
                    "status": status,
                }
            ),
        )

    def log_medium_retrieval_plan(
        self,
        *,
        status: str,
        raw_tool_calls: list[dict[str, Any]],
        plan: RetrievalPlan | None,
        duplicate_calls_removed: int,
        calls_added_by_validator: int,
        calls_removed_by_validator: int,
        error_type: str | None,
        error_message: str | None,
        execution_mode: Literal["shadow", "active"],
    ) -> None:
        """Record a medium planner outcome and whether it may be executed."""
        actions = None
        if plan is not None:
            actions = [
                action.model_dump() if hasattr(action, "model_dump") else action.dict()
                for action in plan.actions
            ]
        log_structured(
            "search_v2.medium_retrieval_plan",
            self._with_ids(
                {
                    "status": status,
                    "raw_tool_calls": raw_tool_calls,
                    "validated_actions": actions,
                    "duplicate_calls_removed": duplicate_calls_removed,
                    "calls_added_by_validator": calls_added_by_validator,
                    "calls_removed_by_validator": calls_removed_by_validator,
                    "error_type": error_type,
                    "error_message": error_message,
                    "execution_mode": execution_mode,
                }
            ),
        )

    def log_retrieval_fallback(
        self,
        *,
        requested_plan_source: Literal["medium"],
        executed_plan_source: Literal["low"],
        reason: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record why a requested medium search executed the low plan."""
        log_structured(
            "search_v2.retrieval_fallback",
            self._with_ids(
                {
                    "requested_plan_source": requested_plan_source,
                    "executed_plan_source": executed_plan_source,
                    "reason": reason,
                    "error_type": error_type,
                    "error_message": error_message[:500] if error_message is not None else None,
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
