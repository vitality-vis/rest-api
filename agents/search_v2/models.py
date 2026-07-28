"""Public and internal DTOs for search v2."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MAX_SEARCH_QUERY_LENGTH = 10_000


class SearchV2Request(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_SEARCH_QUERY_LENGTH)
    effort: Literal["low", "medium", "high"] = "low"
    result_limit: int = Field(default=10, ge=1, le=100)


class SearchIntent(BaseModel):
    retrieval_target: Literal["topic", "metadata_browse"]
    topic: str | None = None
    title: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None
    min_citations: int | None = None
    criteria: list[str] = Field(default_factory=list)


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


class ChatRoute(BaseModel):
    route: Literal["search", "other"]
    search_intent: SearchIntent | None = None
