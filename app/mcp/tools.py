"""Registration of public, read-only paper tools."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from model.paper import PaperCitationsResponse, SearchRequest, SimilarPapersRequest
from service.citations import (
    PaperCitationsNotFoundError,
    PaperCitationsProviderError,
    PaperCitationsUnavailableError,
    get_paper_citations as fetch_paper_citations,
)
from service.search import (
    ExactSearchExpressionError,
    SearchUnavailableError,
    find_paper_by_id,
    find_similar_by_papers,
    search,
)

from app.mcp.provenance import mcp_tool_logged
from app.mcp.schemas import (
    CitationGroup,
    CitationResult,
    McpPaper,
    PaperSearchResult,
    paper_from_api,
)


Limit = Annotated[int, Field(ge=1, le=50)]
Offset = Annotated[int, Field(ge=0, le=10_000)]
QueryOffset = Annotated[int, Field(ge=0, le=16_382)]
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
OPEN_WORLD_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


def register_public_tools(server: MCPServer) -> None:
    """Register the public corpus tools on ``server``."""

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("search_papers_bool")
    def search_papers_bool(
        expression: str,
        limit: Limit = 10,
        offset: QueryOffset = 0,
    ) -> PaperSearchResult:
        """Search exact words and quoted phrases with Boolean logic.

        The expression searches the combined paper search_text field. Plain
        multi-word input is treated as one exact phrase. Use AND, OR, NOT, and
        parentheses for Boolean logic; double quotes also mark an exact phrase.
        Matching is case-insensitive. For pagination, repeat the same expression
        and pass the returned next_offset. Examples: ``literature review``,
        ``privacy AND camera``, and
        ``("large language model" AND evaluation) OR hallucination``.
        """
        normalized_expression = expression.strip()
        if not normalized_expression:
            raise ToolError("expression is required.")
        _validate_query_page(offset, limit)
        try:
            return _execute_search(
                SearchRequest(
                    search_query=normalized_expression,
                    search_mode="bool",
                    limit=limit,
                    offset=offset,
                )
            )
        except ExactSearchExpressionError as error:
            raise ToolError(f"Invalid Boolean-search expression: {error}") from error
        except SearchUnavailableError as error:
            raise ToolError("Paper search is temporarily unavailable.") from error

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("search_papers_bm25")
    def search_papers_bm25(
        query: str,
        authors: list[str] | None = None,
        venues: list[str] | None = None,
        keywords: list[str] | None = None,
        title: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        limit: Limit = 10,
        offset: QueryOffset = 0,
    ) -> PaperSearchResult:
        """Search by lexical BM25 relevance.

        Use for exact terminology, named methods, system names, and title
        fragments. Filters are combined with the query using AND; values within
        authors, venues, or keywords are alternatives (OR). For another page,
        repeat the same arguments and pass the returned next_offset.
        """
        return _search(
            "bm25", query, authors, venues, keywords, title, year_from,
            year_to, min_citations, max_citations, limit, offset,
        )

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("search_papers_semantic")
    def search_papers_semantic(
        query: str,
        authors: list[str] | None = None,
        venues: list[str] | None = None,
        keywords: list[str] | None = None,
        title: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        limit: Limit = 10,
        offset: QueryOffset = 0,
    ) -> PaperSearchResult:
        """Search by conceptual similarity using dense embeddings.

        Use for research questions and concepts whose relevant papers may use
        different wording. Filters are combined with the query using AND;
        values within authors, venues, or keywords are alternatives (OR). For
        another page, repeat the same arguments and pass next_offset.
        """
        return _search(
            "vector", query, authors, venues, keywords, title, year_from,
            year_to, min_citations, max_citations, limit, offset,
        )

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("filter_papers")
    def filter_papers(
        title: str | None = None,
        abstract: str | None = None,
        authors: list[str] | None = None,
        venues: list[str] | None = None,
        keywords: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        paper_ids: list[str] | None = None,
        limit: Limit = 10,
        offset: QueryOffset = 0,
    ) -> PaperSearchResult:
        """Find papers using bibliographic metadata constraints, without ranking.

        Use when the request primarily names authors, venues, years, title or
        abstract fragments, keywords, citation ranges, or known paper IDs. For
        another page, repeat the same filters and pass next_offset.
        """
        _validate_ranges(year_from, year_to, min_citations, max_citations)
        _validate_query_page(offset, limit)
        if not any((title, abstract, authors, venues, keywords, paper_ids)) and all(
            value is None
            for value in (year_from, year_to, min_citations, max_citations)
        ):
            raise ToolError("Provide at least one metadata filter.")
        request = SearchRequest(
            search_mode="bool",
            title=title,
            abstract=abstract,
            author=authors,
            source=venues,
            keyword=keywords,
            min_year=year_from,
            max_year=year_to,
            min_citation_counts=min_citations,
            max_citation_counts=max_citations,
            id_list=paper_ids,
            limit=limit,
            offset=offset,
        )
        return _execute_search(request)

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("find_similar_papers")
    def find_similar_papers(
        paper_ids: list[str],
        authors: list[str] | None = None,
        venues: list[str] | None = None,
        keywords: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        limit: Limit = 10,
        offset: QueryOffset = 0,
    ) -> PaperSearchResult:
        """Find conceptually similar papers from one or more known paper IDs.

        Seed papers are excluded from results. Optional metadata filters narrow
        the candidate papers. For another page, repeat the same arguments and
        pass next_offset.
        """
        seeds = list(dict.fromkeys(value.strip() for value in paper_ids if value.strip()))
        if not seeds:
            raise ToolError("paper_ids must contain at least one paper ID.")
        if len(seeds) > 10:
            raise ToolError("At most 10 seed paper IDs may be supplied.")
        _validate_ranges(year_from, year_to, min_citations, max_citations)
        request = SimilarPapersRequest(
            seed_ids=seeds,
            author=authors,
            source=venues,
            keyword=keywords,
            min_year=year_from,
            max_year=year_to,
            min_citation_counts=min_citations,
            max_citation_counts=max_citations,
            limit=limit,
            offset=offset,
        )
        try:
            result = find_similar_by_papers(request)
        except SearchUnavailableError as error:
            raise ToolError("Similar-paper search is temporarily unavailable.") from error
        return _to_search_result(result, offset=offset, page_size=limit)

    @server.tool(annotations=READ_ONLY)
    @mcp_tool_logged("get_paper")
    def get_paper(paper_id: str) -> McpPaper:
        """Get one paper from the VitaLITy corpus by its stable paper ID."""
        normalized_id = paper_id.strip()
        if not normalized_id:
            raise ToolError("paper_id is required.")
        try:
            paper = find_paper_by_id(normalized_id)
        except SearchUnavailableError as error:
            raise ToolError("Paper lookup is temporarily unavailable.") from error
        if paper is None:
            raise ToolError("No paper was found for that paper ID.")
        return paper_from_api(paper)

    @server.tool(annotations=OPEN_WORLD_READ_ONLY)
    @mcp_tool_logged("get_paper_citations")
    def get_paper_citations(
        doi: str,
        direction: Literal["references", "cited_by"] | None = None,
        limit: Limit = 20,
        offset: Offset = 0,
    ) -> CitationResult:
        """Get references and/or citing works for a paper DOI from OpenAlex."""
        normalized_doi = doi.strip()
        if not normalized_doi:
            raise ToolError("doi is required.")
        try:
            raw = fetch_paper_citations(normalized_doi, limit, offset, direction)
            response = PaperCitationsResponse(**raw)
        except PaperCitationsNotFoundError as error:
            raise ToolError("No OpenAlex paper was found for that DOI.") from error
        except PaperCitationsUnavailableError as error:
            raise ToolError("Citation lookup is temporarily unavailable.") from error
        except PaperCitationsProviderError as error:
            raise ToolError("OpenAlex rejected the citation lookup.") from error

        return CitationResult(
            doi=response.source.doi,
            openalex_id=response.source.openalex_id,
            references=CitationGroup(
                total_hint=response.references.total_hint,
                has_more=response.references.has_more,
                papers=[
                    paper_from_api(paper.model_dump(by_alias=True, exclude_none=True))
                    for paper in response.references.papers
                ],
            ),
            cited_by=CitationGroup(
                total_hint=response.cited_by.total_hint,
                has_more=response.cited_by.has_more,
                papers=[
                    paper_from_api(paper.model_dump(by_alias=True, exclude_none=True))
                    for paper in response.cited_by.papers
                ],
            ),
        )


def _search(
    mode: Literal["bm25", "vector"],
    query: str,
    authors: list[str] | None,
    venues: list[str] | None,
    keywords: list[str] | None,
    title: str | None,
    year_from: int | None,
    year_to: int | None,
    min_citations: int | None,
    max_citations: int | None,
    limit: int,
    offset: int,
) -> PaperSearchResult:
    normalized_query = query.strip()
    if not normalized_query:
        raise ToolError("query is required.")
    if len(normalized_query) > 1_000:
        raise ToolError("query must be at most 1000 characters.")
    _validate_ranges(year_from, year_to, min_citations, max_citations)
    _validate_query_page(offset, limit)
    request = SearchRequest(
        search_query=normalized_query,
        search_mode=mode,
        author=authors,
        source=venues,
        keyword=keywords,
        title=title,
        min_year=year_from,
        max_year=year_to,
        min_citation_counts=min_citations,
        max_citation_counts=max_citations,
        limit=limit,
        offset=offset,
    )
    return _execute_search(request)


def _execute_search(request: SearchRequest) -> PaperSearchResult:
    try:
        return _to_search_result(
            search(request),
            offset=request.offset,
            page_size=request.limit,
        )
    except SearchUnavailableError as error:
        raise ToolError("Paper search is temporarily unavailable.") from error


def _to_search_result(
    result,
    *,
    offset: int = 0,
    page_size: int | None = None,
) -> PaperSearchResult:
    consumed = page_size if page_size is not None else len(result.papers)
    next_offset = offset + consumed if result.has_more else None
    if next_offset is not None and next_offset >= 16_382:
        next_offset = None
    return PaperSearchResult(
        papers=[paper_from_api(paper) for paper in result.papers],
        total=result.total,
        has_more=result.has_more,
        next_offset=next_offset,
    )


def _validate_query_page(offset: int, limit: int) -> None:
    # Repositories fetch one extra row to determine has_more. Milvus requires
    # offset + query limit to remain below 16,384.
    if offset + limit + 1 >= 16_384:
        raise ToolError("offset + limit must be at most 16382.")


def _validate_ranges(
    year_from: int | None,
    year_to: int | None,
    min_citations: int | None,
    max_citations: int | None,
) -> None:
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ToolError("year_from must not be greater than year_to.")
    if min_citations is not None and min_citations < 0:
        raise ToolError("min_citations must not be negative.")
    if max_citations is not None and max_citations < 0:
        raise ToolError("max_citations must not be negative.")
    if (
        min_citations is not None
        and max_citations is not None
        and min_citations > max_citations
    ):
        raise ToolError("min_citations must not be greater than max_citations.")
