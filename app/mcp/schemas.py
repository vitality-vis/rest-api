"""Compact, model-friendly schemas returned by public MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


MAX_ABSTRACT_CHARS = 2_000


class McpPaper(BaseModel):
    paper_id: str
    title: str = ""
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    citation_count: int | None = None
    doi: str | None = None
    score: float | None = None


class PaperSearchResult(BaseModel):
    papers: list[McpPaper] = Field(default_factory=list)
    total: int | None = None
    has_more: bool = False


class CitationGroup(BaseModel):
    total_hint: int = 0
    has_more: bool = False
    papers: list[McpPaper] = Field(default_factory=list)


class CitationResult(BaseModel):
    doi: str
    openalex_id: str
    references: CitationGroup
    cited_by: CitationGroup


def paper_from_api(payload: dict[str, Any]) -> McpPaper:
    """Convert the established REST paper shape into the MCP wire shape."""
    abstract = str(payload.get("Abstract") or "").strip()
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[: MAX_ABSTRACT_CHARS - 1].rstrip() + "…"
    paper_id = payload.get("ID") or payload.get("paper_id") or payload.get("openalex_id")
    return McpPaper(
        paper_id=str(paper_id or ""),
        title=str(payload.get("Title") or ""),
        abstract=abstract or None,
        authors=[str(value) for value in payload.get("Authors") or []],
        keywords=[str(value) for value in payload.get("Keywords") or []],
        venue=str(payload.get("Source") or "") or None,
        year=payload.get("Year"),
        citation_count=payload.get("CitationCounts"),
        doi=payload.get("doi"),
        score=payload.get("score"),
    )
