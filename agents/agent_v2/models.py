"""Public and internal DTOs for agent v2 and its search capability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


MAX_SEARCH_QUERY_LENGTH = 10_000


class SearchV2Request(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_SEARCH_QUERY_LENGTH)
    effort: Literal["low", "medium", "high"] = "low"
    result_limit: int = Field(default=10, ge=1, le=100)


class SearchIntent(BaseModel):
    topic: str | None = None
    title: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None
    min_citations: int | None = None


class SearchV2Paper(BaseModel):
    paper: dict
    retrieval_sources: list[str] = Field(default_factory=list)
    retrieval_ranks: dict[str, int] = Field(default_factory=dict)
    rrf_score: float | None = None
    rerank_score: float | None = None


class SearchV2Response(BaseModel):
    query: str
    effort: Literal["low", "medium", "high"]
    intent: SearchIntent
    policy: Literal["filter", "hybrid"]
    papers: list[SearchV2Paper] = Field(default_factory=list)
    status: Literal["complete", "partial"] = "complete"
    diagnostics: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class V2ChatRequest:
    """Request contract owned by the ``/chat/v2`` pipeline.

    This intentionally stays separate from the legacy agent request.  When v2
    temporarily falls back to v1, its runner performs an explicit conversion.
    """

    text: str
    chat_id: str = "default"
    history: list[dict[str, str]] | None = None
    selected_paper_ids: list[str] | None = None
    context: dict[str, object] | None = None
    effort: str = "low"
    trace_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    requested_mode: Literal["auto", "synthesis"] = "auto"
    user_id: str | None = None


RouteKind = Literal["talk", "search", "synthesis", "clarify"]
ResponseMode = Literal["papers", "grounded_answer"]
RouterDecisionStatus = Literal[
    "explicit_mode",
    "model_decision",
    "json_parse_failed",
    "validation_failed",
    "router_error",
    "incomplete_search_decision",
]


class ChatRequestContext(BaseModel):
    """Structured client context; IDs are validated again by the handler."""

    selected_paper_ids: list[str] = Field(default_factory=list)
    requested_mode: Literal["auto", "synthesis"] = "auto"


class RouteDecisionOutput(BaseModel):
    """Fields the router LLM is asked to return."""

    route: RouteKind
    response_mode: ResponseMode | None = None
    search_intent: SearchIntent | None = None
    clarification_question: str | None = None
    use_file_search: bool = False


class RouteDecision(RouteDecisionOutput):
    """Routing decision for a v2 chat turn, plus internal status."""

    decision_status: RouterDecisionStatus = "model_decision"


class SynthesisExecutionPlan(BaseModel):
    """Evidence plan resolved after checking the selected papers' file state.

    Metadata is always included for the supplied IDs. File Search is used only
    when the router judged full-text evidence necessary and at least one
    selected paper has a completed full-text index. Selected papers without an
    indexed PDF remain represented by their metadata.
    """

    use_file_search: bool
    metadata_paper_ids: list[str] = Field(default_factory=list)
    file_search_paper_ids: list[str] = Field(default_factory=list)
    file_search_file_ids: list[str] = Field(default_factory=list)
    file_search_file_to_paper_id: dict[str, str] = Field(default_factory=dict)
