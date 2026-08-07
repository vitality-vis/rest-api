"""Paper records and paper-search request/response models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PaperFilters(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    author: Optional[List[str]] = None
    source: Optional[List[str]] = None
    keyword: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    id_list: Optional[List[str]] = None
    min_citation_counts: Optional[int] = None
    max_citation_counts: Optional[int] = None


class SearchRequest(PaperFilters):
    search_query: Optional[str] = None
    limit: int = 20
    offset: int = 0
    search_mode: Literal["exact", "bm25", "vector"] = "exact"
    embedding_model: Optional[str] = None


class SimilarPapersRequest(PaperFilters):
    """Request for papers similar to one or more existing papers."""

    seed_ids: List[str] = Field(default_factory=list)
    limit: int = 25


class UmapCoordinates(BaseModel):
    """Current persisted 2D projection format."""

    x: float
    y: float
    embedding_model: Optional[str] = None


class PaperBase(BaseModel):
    """Canonical paper metadata shared by API paper payloads.

    Serialized with aliases (``Title``, ``Abstract``, …) to match the frontend
    ``Paper`` contract. Subclasses add endpoint-specific fields.
    """

    model_config = ConfigDict(populate_by_name=True)

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
    umap: Optional[UmapCoordinates] = None


class PaperResponse(PaperBase):
    """Paper payload returned by search / similar-paper endpoints."""

    score: Optional[float] = None


class PaperCitationsRequest(BaseModel):
    """Request for one paper's OpenAlex citation neighbors."""

    doi: str = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)
    direction: Optional[Literal["references", "cited_by"]] = None


class PaperCitationItem(PaperResponse):
    """Same paper payload as ``getPapers``, plus OpenAlex / corpus extras.

    Wire format matches frontend ``Paper`` (via ``PaperBase`` aliases).
    ``in_corpus`` is set by a Zilliz DOI gate before the response is returned.
    ``openalex_id`` / ``raw`` are citation-specific.
    """

    openalex_id: str
    in_corpus: bool = False
    raw: Optional[Dict[str, Any]] = None


class PaperCitationSource(BaseModel):
    """Resolved identity of the paper whose citations were requested."""

    doi: str
    openalex_id: str


class PaperCitationGroup(BaseModel):
    """One citation direction: OpenAlex total hint + papers."""

    total_hint: int = Field(ge=0)
    has_more: bool = False
    papers: List[PaperCitationItem] = Field(default_factory=list)


class PaperCitationsResponse(BaseModel):
    """Response returned by the ``/getPaperCitations`` endpoint."""

    source: PaperCitationSource
    references: PaperCitationGroup
    cited_by: PaperCitationGroup


@dataclass
class SearchResult:
    papers: List[Dict] = field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False


class GetPapersResponse(BaseModel):
    """Response returned by the ``/getPapers`` endpoint."""

    papers: List[PaperResponse] = Field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False
