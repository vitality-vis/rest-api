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
from repositories.zilliz.query_expressions import build_paper_query_expr, ids_to_expr


logging = get_logger()


class InvalidRetrievalScoreError(RuntimeError):
    """Raised when Zilliz returns an unusable relevance score."""


@dataclass
class RepositoryHit:
    paper: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class RepositoryPage:
    hits: List[RepositoryHit] = field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False


def _client():
    if not ensure_collection_loaded(config.PAPER_COLLECTION):
        return None
    return get_client()


def _safe_limit_offset(limit: int, offset: int) -> tuple[int, int]:
    return min(max(int(limit or 100), 1), 100), max(int(offset or 0), 0)


def _count_matching(client, expression: str) -> Optional[int]:
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
        logging.warning(
            "Zilliz count(*) failed for filter=%r: %s",
            expression,
            error,
        )
        return None


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
    if not client:
        return []
    try:
        return client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=ids_to_expr([str(paper_id) for paper_id in paper_ids]),
            output_fields=SCALAR_FIELDS,
            limit=len(paper_ids) + 100,
        ) or []
    except Exception as error:
        logging.error("Error fetching papers by ID: %s", error, exc_info=True)
        return []


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
    if not client:
        return RepositoryPage(total=0)

    expression = build_paper_query_expr(filters, query_text=query_text)
    try:
        rows = client.query(
            collection_name=config.PAPER_COLLECTION,
            filter=expression,
            output_fields=SCALAR_FIELDS,
            limit=safe_limit + 1,
            offset=safe_offset,
        ) or []
        has_more = len(rows) > safe_limit
        hits = [RepositoryHit(paper=row) for row in rows[:safe_limit] if row]

        if expression == 'paper_uid != ""':
            try:
                stats = client.get_collection_stats(config.PAPER_COLLECTION) or {}
                total = int(stats.get("row_count", 0))
            except Exception:
                total = safe_offset + len(hits) + int(has_more)
        else:
            total = _count_matching(client, expression)
            if total is None:
                total = safe_offset + len(hits) + int(has_more)
        return RepositoryPage(hits=hits, total=total, has_more=has_more)
    except Exception as error:
        logging.error("Zilliz filtered search failed: %s", error, exc_info=True)
        return RepositoryPage(total=0)


def _hydrate_ranked_hits(
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
    if not client:
        return RepositoryPage()

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
        return RepositoryPage()

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
        hits=_hydrate_ranked_hits(ordered_ids, scores),
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
    if not client:
        return RepositoryPage()

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
        return RepositoryPage()

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
        hits=_hydrate_ranked_hits(ordered_ids, scores),
        has_more=has_more,
    )


def search_papers_by_vector(
    vector: List[float],
    *,
    embedding_type: str,
    limit: int,
    exclude_ids: Optional[List[str]] = None,
) -> List[RepositoryHit]:
    """Legacy-compatible unfiltered vector lookup used outside SearchService."""
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        vector = tolist()
    is_supported = config.is_supported_embedding_model(embedding_type)
    if not is_supported or len(vector) != config.PAPER_VECTOR_DIMENSION:
        if is_supported:
            logging.error(
                "Query vector has %s dimensions for model %s; expected %s",
                len(vector),
                config.PAPER_EMBEDDING_MODEL,
                config.PAPER_VECTOR_DIMENSION,
            )
        return []
    client = _client()
    if not client:
        return []

    excluded = {str(paper_id) for paper_id in (exclude_ids or [])}
    multiplier = getattr(config, "ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER", 1.5)
    top_k = max(limit + len(excluded) + 5, int(limit * multiplier))
    index_type = getattr(config, "ZILLIZ_INDEX_TYPE", "IVF_FLAT")
    if index_type.upper() == "HNSW":
        search_params = {
            "metric_type": config.PAPER_VECTOR_METRIC,
            "params": {"ef": getattr(config, "ZILLIZ_SEARCH_EF", 64)},
        }
    else:
        search_params = {
            "metric_type": config.PAPER_VECTOR_METRIC,
            "params": {"nprobe": getattr(config, "ZILLIZ_SEARCH_NPROBE", 128)},
        }

    try:
        results = client.search(
            collection_name=config.PAPER_COLLECTION,
            data=[vector],
            anns_field=config.PAPER_VECTOR_FIELD,
            search_params=search_params,
            limit=min(top_k, 16384),
            output_fields=["paper_uid"],
        ) or []
    except Exception as error:
        logging.error("Zilliz search failed: %s", error, exc_info=True)
        return []

    ordered_ids: List[str] = []
    scores: Dict[str, Optional[float]] = {}
    for hit in (results[0] if results else []):
        paper_id, score = search_hit_to_id_and_distance(hit)
        if not paper_id or paper_id in excluded or paper_id in scores:
            continue
        ordered_ids.append(paper_id)
        scores[paper_id] = score
        if len(ordered_ids) >= limit:
            break
    return _hydrate_ranked_hits(ordered_ids, scores)
