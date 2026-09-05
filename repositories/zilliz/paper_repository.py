"""Data operations for papers stored in Zilliz."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from logger_config import get_logger
from model.paper import PaperFilters
from repositories.zilliz.connection import ensure_collection_loaded, get_client
from repositories.zilliz.mappers import SCALAR_FIELDS, search_hit_to_id_and_distance
from repositories.zilliz.query_expressions import (
    build_paper_query_expr,
    compile_boolean_search_expr,
    dois_to_expr,
    ids_to_expr,
)


logging = get_logger()

# Zilliz caps the number of query vectors (nq) in a single search request at 10.
MAX_QUERY_VECTORS_PER_SEARCH = 10


class InvalidRetrievalScoreError(RuntimeError):
    """Raised when Zilliz returns an unusable relevance score."""


class RepositoryUnavailableError(RuntimeError):
    """Raised when a required Zilliz operation cannot be completed."""


@dataclass
class RepositoryHit:
    paper: Dict[str, Any]
    score: Optional[float] = None


@dataclass(frozen=True)
class RepositoryVectorHit:
    """A paper ID and its score from one vector-search result list."""

    paper_id: str
    score: float


@dataclass
class RepositoryPage:
    hits: List[RepositoryHit] = field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False


def _client():
    if not ensure_collection_loaded(config.PAPER_COLLECTION):
        raise RepositoryUnavailableError("Zilliz paper collection is unavailable.")
    client = get_client()
    if not client:
        raise RepositoryUnavailableError("Zilliz client is unavailable.")
    return client


def _safe_limit_offset(limit: int, offset: int) -> tuple[int, int]:
    # The HTTP route caps pages at 100. Agent semantic search intentionally
    # retrieves 120 candidates before CrossEncoder reranking.
    return min(max(int(limit or 100), 1), 120), max(int(offset or 0), 0)


def _count_matching(client, expression: str) -> int:
    try:
        rows = client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=expression,
            output_fields=["count(*)"],
        ) or []
        if not rows:
            return 0
        row = rows[0]
        for key in ("count(*)", "count()"):
            if key in row:
                return int(row[key])
        return int(next(iter(row.values())))
    except Exception as error:
        logging.error(
            "Zilliz count(*) failed for filter=%r: %s",
            expression,
            error,
        )
        raise RepositoryUnavailableError("Zilliz count query failed.") from error


def get_paper_by_id(
    paper_id: str,
) -> Optional[Dict[str, Any]]:
    """Return one raw paper record, or ``None`` when it is absent."""
    papers = get_papers_by_ids([paper_id])
    return papers[0] if papers else None


def get_papers_by_ids(
    paper_ids: List[str],
) -> List[Dict[str, Any]]:
    """Return raw paper records for the requested IDs."""
    if not paper_ids:
        return []
    client = _client()
    try:
        return client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=ids_to_expr([str(paper_id) for paper_id in paper_ids]),
            output_fields=SCALAR_FIELDS,
            limit=len(paper_ids) + 100,
        ) or []
    except Exception as error:
        logging.error("Error fetching papers by ID: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz paper lookup failed.") from error


_DOI_LOOKUP_BATCH_SIZE = 100


def find_corpus_papers_by_dois(dois: List[str]) -> Dict[str, str]:
    """Map casefolded DOIs to corpus ``paper_uid`` values found in Zilliz.

    Vitality ingestion builds ``paper_uid`` as ``doi:{doi.casefold()}``
    (see ``vitality2-dataset/script/upload_papers_to_zilliz.py``). Lookup
    therefore always queries the casefolded bare DOI / ``doi:`` uid form.
    """
    unique: List[str] = []
    seen = set()
    for raw in dois:
        doi = str(raw or "").strip().casefold()
        if not doi or doi in seen:
            continue
        seen.add(doi)
        unique.append(doi)
    if not unique:
        return {}

    client = _client()
    found: Dict[str, str] = {}
    try:
        for start in range(0, len(unique), _DOI_LOOKUP_BATCH_SIZE):
            chunk = unique[start : start + _DOI_LOOKUP_BATCH_SIZE]
            rows = client.query(
                collection_name=config.PAPER_COLLECTION,
                filter=dois_to_expr(chunk),
                output_fields=["doi", "paper_uid"],
                limit=len(chunk) + 100,
            ) or []
            for row in rows:
                paper_uid = str(row.get("paper_uid") or "").strip()
                if not paper_uid:
                    continue
                doi = str(row.get("doi") or "").strip().casefold()
                if not doi:
                    lowered = paper_uid.casefold()
                    if lowered.startswith("doi:"):
                        doi = paper_uid[4:].strip().casefold()
                if doi and doi not in found:
                    found[doi] = paper_uid
    except Exception as error:
        logging.error("Error looking up corpus DOIs: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz DOI lookup failed.") from error
    return found


def find_corpus_dois(dois: List[str]) -> set[str]:
    """Return the subset of ``dois`` that exist in the Zilliz paper corpus."""
    return set(find_corpus_papers_by_dois(dois))


def get_embeddings_by_ids(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch stored embeddings for seed papers, without applying search filters."""
    if not paper_ids:
        return []
    client = _client()
    try:
        return client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=ids_to_expr([str(paper_id) for paper_id in paper_ids]),
            output_fields=["paper_uid", config.PAPER_VECTOR_FIELD],
            limit=len(paper_ids) + 100,
        ) or []
    except Exception as error:
        logging.error("Error fetching paper embeddings: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz embedding lookup failed.") from error


def search_filtered(
    filters: PaperFilters,
    *,
    query_text: Optional[str],
    limit: int,
    offset: int,
) -> RepositoryPage:
    """Run scalar filtering with optional analyzed text matching."""
    safe_limit, safe_offset = _safe_limit_offset(limit, offset)
    client = _client()

    expression = build_paper_query_expr(filters, query_text=query_text)
    try:
        rows = client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=expression,
            output_fields=SCALAR_FIELDS,
            limit=safe_limit + 1,
            offset=safe_offset,
            order_by=["paper_uid:asc"],
        ) or []
        has_more = len(rows) > safe_limit
        hits = [RepositoryHit(paper=row) for row in rows[:safe_limit] if row]

        if expression == 'paper_uid != ""':
            try:
                stats = client.get_collection_stats(config.PAPER_COLLECTION) or {}
                total = int(stats.get("row_count", 0))
            except Exception as error:
                raise RepositoryUnavailableError(
                    "Zilliz collection statistics query failed."
                ) from error
        else:
            total = _count_matching(client, expression)
        return RepositoryPage(hits=hits, total=total, has_more=has_more)
    except Exception as error:
        logging.error("Zilliz filtered search failed: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz filtered search failed.") from error


def search_exact_boolean(
    expression: str,
    filters: PaperFilters,
    *,
    limit: int,
    offset: int = 0,
) -> RepositoryPage:
    """Run an unranked exact token/phrase Boolean search on ``search_text``."""
    safe_limit, safe_offset = _safe_limit_offset(limit, offset)
    text_expression = compile_boolean_search_expr(expression)
    metadata_expression = build_paper_query_expr(filters, include_query_text=False)
    compiled = (
        text_expression
        if metadata_expression == 'paper_uid != ""'
        else f"({text_expression}) and ({metadata_expression})"
    )
    client = _client()
    try:
        rows = client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=compiled,
            output_fields=SCALAR_FIELDS,
            limit=safe_limit + 1,
            offset=safe_offset,
            order_by=["paper_uid:asc"],
        ) or []
        return RepositoryPage(
            hits=[RepositoryHit(paper=row) for row in rows[:safe_limit] if row],
            total=_count_matching(client, compiled),
            has_more=len(rows) > safe_limit,
        )
    except RepositoryUnavailableError:
        raise
    except Exception as error:
        logging.error("Zilliz exact Boolean search failed: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz exact Boolean search failed.") from error


def hydrate_ranked_papers(
    ordered_ids: List[str],
    scores: Dict[str, Optional[float]],
) -> List[RepositoryHit]:
    records = get_papers_by_ids(ordered_ids)
    records_by_id = {
        str(record.get("paper_uid")): record
        for record in records
        if record.get("paper_uid")
    }
    return [
        RepositoryHit(paper=records_by_id[paper_id], score=scores.get(paper_id))
        for paper_id in ordered_ids
        if paper_id in records_by_id
    ]


def search_bm25(
    query_text: str,
    filters: PaperFilters,
    *,
    limit: int,
    offset: int,
) -> RepositoryPage:
    """Run native sparse BM25 search and hydrate results in rank order."""
    safe_limit, safe_offset = _safe_limit_offset(limit, offset)
    client = _client()

    expression = build_paper_query_expr(filters, include_query_text=False)
    kwargs = {
        "collection_name": config.PAPER_COLLECTION,
        "data": [query_text.strip()],
        "anns_field": "search_sparse",
        "search_params": {"metric_type": "BM25", "params": {}},
        "filter": expression,
        "limit": safe_limit + 1,
        "output_fields": ["paper_uid"],
    }
    if safe_offset:
        kwargs["offset"] = safe_offset
    try:
        results = client.search(**kwargs) or []
    except Exception as error:
        logging.error("Zilliz BM25 search failed: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz BM25 search failed.") from error

    raw_hits = results[0] if results else []
    has_more = len(raw_hits) > safe_limit
    ordered_ids: List[str] = []
    scores: Dict[str, Optional[float]] = {}
    for hit in raw_hits[:safe_limit]:
        paper_id, score = search_hit_to_id_and_distance(hit)
        if not paper_id or paper_id in scores:
            continue
        ordered_ids.append(paper_id)
        scores[paper_id] = score
    return RepositoryPage(
        hits=hydrate_ranked_papers(ordered_ids, scores),
        has_more=has_more,
    )


def search_by_vector(
    vector: List[float],
    filters: PaperFilters,
    *,
    limit: int,
    offset: int,
) -> RepositoryPage:
    """Run a filtered dense search and hydrate results in rank order."""
    safe_limit, safe_offset = _safe_limit_offset(limit, offset)
    client = _client()

    metadata_expression = build_paper_query_expr(filters, include_query_text=False)
    expression = (
        f"({metadata_expression}) and has_embedding == true "
        f'and embedding_model == "{config.PAPER_EMBEDDING_MODEL}"'
    )
    kwargs = {
        "collection_name": config.PAPER_COLLECTION,
        "data": [vector],
        "anns_field": config.PAPER_VECTOR_FIELD,
        "search_params": {"metric_type": config.PAPER_VECTOR_METRIC, "params": {}},
        "filter": expression,
        "limit": safe_limit + 1,
        "output_fields": ["paper_uid"],
    }
    if safe_offset:
        kwargs["offset"] = safe_offset
    try:
        results = client.search(**kwargs) or []
    except Exception as error:
        logging.error("Zilliz vector search failed: %s", error, exc_info=True)
        raise RepositoryUnavailableError("Zilliz vector search failed.") from error

    raw_hits = results[0] if results else []
    has_more = len(raw_hits) > safe_limit
    ordered_ids: List[str] = []
    scores: Dict[str, float] = {}
    for hit in raw_hits[:safe_limit]:
        paper_id, score = search_hit_to_id_and_distance(hit)
        if not paper_id or paper_id in scores:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as error:
            raise InvalidRetrievalScoreError(
                "Zilliz vector search returned a result without a numeric score."
            ) from error
        if not math.isfinite(numeric_score):
            raise InvalidRetrievalScoreError(
                "Zilliz vector search returned a non-finite score."
            )
        ordered_ids.append(paper_id)
        scores[paper_id] = numeric_score
    return RepositoryPage(
        hits=hydrate_ranked_papers(ordered_ids, scores),
        has_more=has_more,
    )


def search_by_vectors(
    vectors: List[List[float]],
    filters: PaperFilters,
    *,
    candidate_limit: int,
) -> List[List[RepositoryVectorHit]]:
    """Search a common filtered corpus once for every supplied query vector."""
    if not vectors:
        return []
    client = _client()

    metadata_expression = build_paper_query_expr(filters, include_query_text=False)
    expression = (
        f"({metadata_expression}) and has_embedding == true "
        f'and embedding_model == "{config.PAPER_EMBEDDING_MODEL}"'
    )
    results: List[List[Any]] = []
    for start in range(0, len(vectors), MAX_QUERY_VECTORS_PER_SEARCH):
        vector_batch = vectors[start:start + MAX_QUERY_VECTORS_PER_SEARCH]
        try:
            batch_results = client.search(
                collection_name=config.PAPER_COLLECTION,
                data=vector_batch,
                anns_field=config.PAPER_VECTOR_FIELD,
                search_params={"metric_type": config.PAPER_VECTOR_METRIC, "params": {}},
                filter=expression,
                limit=min(max(int(candidate_limit), 1), 120),
                output_fields=["paper_uid"],
            ) or []
        except Exception as error:
            logging.error(
                "Zilliz bulk vector search failed for vectors %d-%d: %s",
                start + 1,
                start + len(vector_batch),
                error,
                exc_info=True,
            )
            raise RepositoryUnavailableError("Zilliz bulk vector search failed.") from error
        results.extend(batch_results)

    ranked_results: List[List[RepositoryVectorHit]] = []
    for raw_hits in results:
        ranked_hits: List[RepositoryVectorHit] = []
        seen_ids = set()
        for hit in raw_hits or []:
            paper_id, score = search_hit_to_id_and_distance(hit)
            if not paper_id or paper_id in seen_ids:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError) as error:
                raise InvalidRetrievalScoreError(
                    "Zilliz vector search returned a result without a numeric score."
                ) from error
            if not math.isfinite(numeric_score):
                raise InvalidRetrievalScoreError(
                    "Zilliz vector search returned a non-finite score."
                )
            seen_ids.add(paper_id)
            ranked_hits.append(RepositoryVectorHit(paper_id, numeric_score))
        ranked_results.append(ranked_hits)
    ranked_results.extend([[] for _ in range(len(vectors) - len(ranked_results))])
    return ranked_results
