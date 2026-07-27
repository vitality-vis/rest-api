"""Application service for paper retrieval."""
from __future__ import annotations

from typing import Dict

import config
from logger_config import get_logger
from model.paper import SearchRequest, SearchResult, SimilarPapersRequest
from repositories.zilliz import paper_repository
from repositories.zilliz.mappers import paper_to_api_response
from service.embed import embed_query


logging = get_logger()
RRF_K = 60
SIMILAR_PAPERS_CANDIDATES_PER_SEED = 100


class SearchUnavailableError(RuntimeError):
    """Raised when a required retrieval dependency cannot complete a search."""


class VectorSearchUnavailableError(SearchUnavailableError):
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
    except paper_repository.RepositoryUnavailableError as error:
        raise SearchUnavailableError(str(error)) from error
    except SearchUnavailableError:
        raise
    except Exception as error:
        logging.error("Paper search failed: %s", error, exc_info=True)
        raise SearchUnavailableError("Paper search is temporarily unavailable.") from error


def find_similar_by_papers(request: SimilarPapersRequest) -> SearchResult:
    """Find papers related to multiple seed papers with rank-based fusion."""
    seed_ids = list(dict.fromkeys(str(seed_id).strip() for seed_id in request.seed_ids if str(seed_id).strip()))
    if not seed_ids:
        return SearchResult()

    try:
        seed_records = paper_repository.get_embeddings_by_ids(seed_ids)
        vectors = [
            record.get(config.PAPER_VECTOR_FIELD)
            for record in seed_records
            if record.get(config.PAPER_VECTOR_FIELD)
        ]
        if not vectors:
            return SearchResult()

        ranked_lists = paper_repository.search_by_vectors(
            vectors,
            request,
            candidate_limit=SIMILAR_PAPERS_CANDIDATES_PER_SEED,
        )
        seed_id_set = set(seed_ids)
        rrf_scores: Dict[str, float] = {}
        for ranked_hits in ranked_lists:
            rank = 0
            for hit in ranked_hits:
                if hit.paper_id in seed_id_set:
                    continue
                rank += 1
                rrf_scores[hit.paper_id] = rrf_scores.get(hit.paper_id, 0.0) + 1.0 / (RRF_K + rank)

        requested_limit = min(max(int(request.limit or 25), 1), 100)
        ordered_ids = sorted(rrf_scores, key=lambda paper_id: (-rrf_scores[paper_id], paper_id))
        selected_ids = ordered_ids[:requested_limit]
        hits = paper_repository.hydrate_ranked_papers(selected_ids, rrf_scores)
        return _format_result(
            paper_repository.RepositoryPage(
                hits=hits,
                has_more=len(ordered_ids) > requested_limit,
            )
        )
    except paper_repository.InvalidRetrievalScoreError as error:
        raise VectorSearchUnavailableError(str(error)) from error
    except paper_repository.RepositoryUnavailableError as error:
        raise SearchUnavailableError(str(error)) from error
    except Exception as error:
        logging.error("Similar-paper search failed: %s", error, exc_info=True)
        raise SearchUnavailableError("Similar-paper search is temporarily unavailable.") from error


def to_legacy_payload(result: SearchResult) -> Dict:
    """Return the dictionary shape consumed by callers not yet migrated."""
    payload = {
        "papers": result.papers,
        "has_more": result.has_more,
    }
    if result.total is not None:
        payload["total"] = result.total
    return payload
