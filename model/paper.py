"""Paper models exposed by the REST API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PaperResponse(BaseModel):
    """Legacy-compatible paper payload returned by REST endpoints."""

    paper_id: Optional[str] = Field(default=None, alias="ID")
    title: str = Field(default="", alias="Title")
    abstract: str = Field(default="", alias="Abstract")
    authors: List[str] = Field(default_factory=list, alias="Authors")
    keywords: List[str] = Field(default_factory=list, alias="Keywords")
    source: str = Field(default="", alias="Source")
    year: Optional[int] = Field(default=None, alias="Year")
    citation_count: Optional[int] = Field(default=None, alias="CitationCounts")
    doi: Optional[str] = None
    dblp_key: Optional[str] = None
    dblp_source: Optional[str] = None
    full_paper: Optional[bool] = None
    umap: Optional[list] = None
    similarity: float = Field(default=0.0, alias="_Sim")
    sim: float = Field(default=0.0, alias="Sim")
    score: float = 0.0
    bm25_score: Optional[float] = None


class GetPapersRequest(BaseModel):
    """Payload accepted by the ``/getPapers`` endpoint."""

    search_query: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    author: Optional[List[str]] = None
    source: Optional[List[str]] = None
    keyword: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    id_list: Optional[List[str]] = None
    limit: int = 20
    offset: int = 0
    min_citation_counts: Optional[int] = None
    max_citation_counts: Optional[int] = None
    search_mode: Literal["exact", "bm25"] = "exact"


class GetPapersResponse(BaseModel):
    """Response returned by the ``/getPapers`` endpoint."""

    papers: List[PaperResponse] = Field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False
