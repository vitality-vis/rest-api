"""Application service for paper retrieval."""
from __future__ import annotations

from typing import Dict

import config
from logger_config import get_logger
from model.paper import SearchRequest, SearchResult
from repositories.zilliz import paper_repository
from repositories.zilliz.mappers import paper_to_api_response
from service.embed import embed_query


logging = get_logger()


class VectorSearchUnavailableError(RuntimeError):
    """Raised when dense retrieval cannot produce a valid ranked result."""


def _format_result(page) -> SearchResult:
    papers = []
    for repository_hit in page.hits:
        record = dict(repository_hit.paper)
        if repository_hit.score is not None:
            record["score"] = repository_hit.score
        papers.append(paper_to_api_response(record))
    return SearchResult(papers=papers, total=page.total, has_more=page.has_more)


def search(
    request: SearchRequest,
    *,
    default_embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
) -> SearchResult:
    """Execute one paper-search request using the selected retrieval strategy."""
    query_text = str(request.search_query or "").strip()
    is_bm25 = request.search_mode == "bm25" and bool(query_text)
    is_dense = request.search_mode == "vector" and bool(query_text)
    embedding_model = (
        request.embedding_model
        if is_dense and request.embedding_model
        else default_embedding_model
    )
    if not config.is_supported_embedding_model(embedding_model):
        logging.error("Unsupported embedding model: %s", embedding_model)
        return SearchResult(total=0)

    try:
        if is_bm25:
            page = paper_repository.search_bm25(
                query_text,
                request,
                limit=request.limit,
                offset=request.offset,
            )
            return _format_result(page)

        if is_dense:
            query_vector = embed_query(query_text)
            if not query_vector:
                raise VectorSearchUnavailableError(
                    "The query embedding service did not return a usable embedding."
                )
            page = paper_repository.search_by_vector(
                query_vector,
                request,
                limit=request.limit,
                offset=request.offset,
            )
            return _format_result(page)

        page = paper_repository.search_filtered(
            request,
            query_text=query_text or None,
            limit=request.limit,
            offset=request.offset,
        )
        return _format_result(page)
    except paper_repository.InvalidRetrievalScoreError as error:
        raise VectorSearchUnavailableError(str(error)) from error
    except VectorSearchUnavailableError:
        raise
    except Exception as error:
        logging.error("Paper search failed: %s", error, exc_info=True)
        return SearchResult(total=0)


def to_legacy_payload(result: SearchResult) -> Dict:
    """Return the dictionary shape consumed by callers not yet migrated."""
    payload = {
        "papers": result.papers,
        "has_more": result.has_more,
    }
    if result.total is not None:
        payload["total"] = result.total
    return payload
