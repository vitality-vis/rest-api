"""
Zilliz Cloud (Milvus-compatible) vector database service.
"""
import json
import math
import sys
import numpy as np
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from model.const import EMBED
from config import (
    DEFAULT_EMBEDDING_MODEL,
    PAPER_COLLECTION,
    PAPER_EMBEDDING_MODEL,
    PAPER_UMAP_FIELD,
    PAPER_VECTOR_FIELD,
    is_supported_embedding_model,
)
from logger_config import get_logger
from repositories.zilliz.connection import (
    ensure_collection_loaded,
    get_client as _get_milvus_client,
)
from repositories.zilliz.query_expressions import (
    ids_to_expr as _ids_to_expr,
    where_to_expr as _zilliz_where_to_expr,
)
from repositories.zilliz import paper_repository
from repositories.zilliz.mappers import (
    SCALAR_FIELDS as _SCALAR_FIELDS,
    paper_to_api_response as format_doc_for_frontend,
    row_to_metadata as _row_to_meta,
    rows_to_umap_points as format_umap_points,
)
from service.metadata_normalizer import parse_string_list
from service.search import (
    VectorSearchUnavailableError,
)

logging = get_logger()


# Collection name mapping; include string keys for agent_tools
COLLECTION_MAPPING = {
    PAPER_EMBEDDING_MODEL: PAPER_COLLECTION,
}


def _embedding_model_supported_or_log(
    embedding_type: str = DEFAULT_EMBEDDING_MODEL,
):
    supported = is_supported_embedding_model(embedding_type)
    if not supported:
        logging.error("Unsupported embedding model: %s", embedding_type)
    return supported


class _MilvusCollectionCompat:
    """Temporary adapter for legacy code while repository methods migrate."""

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def query(self, *, expr, output_fields, limit=None, offset=None):
        client = _get_milvus_client()
        if not client:
            return []
        pagination = {}
        if limit is not None:
            pagination["limit"] = limit
        if offset is not None:
            pagination["offset"] = offset
        return client.query(
            collection_name=self._collection_name,
            filter=expr,
            output_fields=output_fields,
            **pagination,
        )

    def search(
        self, *, data, anns_field, param, limit, output_fields, filter=None, offset=None
    ):
        client = _get_milvus_client()
        if not client:
            return []
        kwargs = {
            "collection_name": self._collection_name,
            "data": data,
            "anns_field": anns_field,
            "search_params": param,
            "limit": limit,
            "output_fields": output_fields,
        }
        if filter:
            kwargs["filter"] = filter
        if offset:
            kwargs["offset"] = offset
        return client.search(**kwargs)

    @property
    def num_entities(self):
        client = _get_milvus_client()
        if not client:
            return 0
        stats = client.get_collection_stats(self._collection_name) or {}
        return int(stats.get("row_count", 0))


_collection_adapters = {}


def _get_collection(collection_name: str):
    """Temporary legacy name backed by MilvusClient, not ORM Collection."""
    if not ensure_collection_loaded(collection_name):
        return None
    if collection_name not in _collection_adapters:
        _collection_adapters[collection_name] = _MilvusCollectionCompat(collection_name)
    return _collection_adapters[collection_name]

_QUERY_BATCH_SIZE = 2000
_ID_BATCH_SIZE = 5000

def _query_all_batched(coll, output_fields: List[str], *, desc: Optional[str] = None):
    """
    Fetch all rows in batches to avoid gRPC 'message larger than max' (4MB).
    Phase 1: collect all IDs (ID only, small response). Phase 2: for each chunk
    of IDs query full rows with 'ID in [chunk]' (small request/response).
    """
    return _query_by_expr_batched(coll, 'paper_uid != ""', output_fields, desc=desc)

def _query_by_expr_batched(
    coll,
    base_expr: str,
    output_fields: List[str],
    *,
    desc: Optional[str] = None,
):
    """
    Fetch all rows matching base_expr in batches to stay under gRPC 4MB.
    Phase 1: collect IDs matching base_expr (ID only, batched with ID not in seen).
    Phase 2: fetch full rows by ID chunks.
    """
    label = desc or "Zilliz"
    all_ids = []
    seen_ids = []
    # disable=False: Cursor / piped terminals often report not-a-TTY and hide bars.
    with tqdm(
        desc=f"{label}: IDs",
        unit="id",
        leave=True,
        disable=False,
        file=sys.stderr,
        mininterval=0.3,
    ) as pbar:
        while True:
            if seen_ids:
                expr = f"({base_expr}) and {_zilliz_where_to_expr({'ID': {'$nin': seen_ids}})}"
            else:
                expr = base_expr
            try:
                res = coll.query(expr=expr, output_fields=["paper_uid"], limit=_ID_BATCH_SIZE)
            except Exception as e:
                logging.warning(f"Batch ID fetch failed: {e}. Collected {len(all_ids)} IDs.")
                break
            if not res:
                break
            ids_batch = [r["paper_uid"] for r in res if r.get("paper_uid")]
            all_ids.extend(ids_batch)
            seen_ids.extend(ids_batch)
            pbar.update(len(ids_batch))
            if len(res) < _ID_BATCH_SIZE:
                break
    if not all_ids:
        return []
    all_rows = []
    with tqdm(
        total=len(all_ids),
        desc=f"{label}: rows",
        unit="row",
        leave=True,
        disable=False,
        file=sys.stderr,
        mininterval=0.3,
    ) as pbar:
        for i in range(0, len(all_ids), _QUERY_BATCH_SIZE):
            chunk = all_ids[i : i + _QUERY_BATCH_SIZE]
            expr = _ids_to_expr(chunk)
            try:
                res = coll.query(expr=expr, output_fields=output_fields, limit=len(chunk) + 10)
                rows = res or []
                all_rows.extend(rows)
                pbar.update(len(rows))
            except Exception as e:
                logging.warning(f"Batch row fetch failed: {e}")
                pbar.update(len(chunk))
    return all_rows

# --- Cache ---
_all_papers_cache = {}

def load_all_papers_to_cache(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    global _all_papers_cache
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        return
    coll = _get_collection(collection_name)
    if not coll:
        return
    try:
        logging.info(f"Loading all papers from {collection_name} into memory cache (batched)...")
        res = _query_all_batched(coll, _SCALAR_FIELDS, desc=f"cache {collection_name}")
        cached_papers = []
        for r in (res or []):
            cached_papers.append(format_doc_for_frontend(_row_to_meta(r)))
        _all_papers_cache[collection_name] = cached_papers
        logging.info(f"Cached {len(cached_papers)} papers for {collection_name}")
    except Exception as e:
        logging.error(f"Failed to load papers to cache: {e}", exc_info=True)
        _all_papers_cache[collection_name] = []

def get_cached_papers(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        return []
    if collection_name not in _all_papers_cache:
        load_all_papers_to_cache(embedding_type)
    return _all_papers_cache.get(collection_name, [])

# --- Query helpers ---
def _parse_string_list(value) -> List[str]:
    return parse_string_list(value)


def normalize_results(results, mode="nD"):
    normalized = []
    for doc in results:
        sim = doc.get("score", 0.0)
        try:
            sim = float(sim)
        except Exception:
            sim = 0.0
        if not math.isfinite(sim):
            sim = 0.0
        doc["score"] = float(sim)
        normalized.append(doc)
    return normalized

def query_doc_by_id(_id: str, embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    if not _embedding_model_supported_or_log(embedding_type):
        return None
    record = paper_repository.get_paper_by_id(str(_id))
    return format_doc_for_frontend(record) if record else None

def query_doc_by_title(title: str, embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> list:
    all_papers = get_cached_papers(embedding_type)
    if not all_papers:
        return []
    normalized = title.strip().lower().rstrip(".")
    matches = []
    for doc in all_papers:
        doc_title = str(doc.get("Title") or "").strip().lower().rstrip(".")
        if normalized == doc_title or normalized in doc_title:
            matches.append(doc)
    if not matches:
        try:
            from rapidfuzz import process
            all_titles = [str(d.get("Title") or "") for d in all_papers]
            best_match, score, idx = process.extractOne(normalized, all_titles)
            if score > 80:
                matches.append(all_papers[idx])
        except ImportError:
            pass
    return matches

def query_doc_by_ids(ids: List[str], embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[dict]:
    if not _embedding_model_supported_or_log(embedding_type):
        return []
    records = paper_repository.get_papers_by_ids(ids)
    return [format_doc_for_frontend(record) for record in records if record]

def query_doc_by_embedding(
    paper_ids: Optional[List[str]],
    embedding: List[float],
    embedding_type: str,
    limit: int,
    lang_filter: Dict = None,
) -> List[Dict]:
    hits = paper_repository.search_papers_by_vector(
        embedding,
        embedding_type=embedding_type,
        limit=limit,
        exclude_ids=paper_ids or [],
    )
    documents = []
    for hit in hits:
        record = dict(hit.paper)
        if hit.score is not None:
            record["score"] = hit.score
        documents.append(format_doc_for_frontend(record))
    return documents

def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    paper_ids_to_exclude = [str(p.get("ID")) for p in papers if p.get("ID")]
    if not _embedding_model_supported_or_log(embedding_type):
        return []
    coll = _get_collection(PAPER_COLLECTION)
    if not coll:
        return []
    try:
        expr = _ids_to_expr(paper_ids_to_exclude)
        res = coll.query(expr=expr, output_fields=[PAPER_VECTOR_FIELD], limit=len(paper_ids_to_exclude) + 100)
    except Exception as e:
        logging.error(f"Failed to fetch embeddings from Zilliz: {e}", exc_info=True)
        return []
    vectors_for_mean = []
    for r in (res or []):
        emb = r.get(PAPER_VECTOR_FIELD)
        if isinstance(emb, (list, np.ndarray)) and (np.any(emb) if hasattr(emb, "__len__") else emb):
            vectors_for_mean.append(emb if isinstance(emb, list) else emb.tolist())
    if not vectors_for_mean:
        return []
    mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
    return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)

def query_similar_doc_by_embedding_2d(
    papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None
):
    if not _embedding_model_supported_or_log(embedding_type):
        return []
    umap_field = PAPER_UMAP_FIELD
    query_points = []
    for p in papers:
        coords = p.get(umap_field)
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = None
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                xy = np.asarray(coords, dtype=float)
                if np.all(np.isfinite(xy)):
                    query_points.append(xy)
            except Exception:
                pass
    if not query_points:
        return []
    mean_vector = np.mean(np.vstack(query_points), axis=0)
    all_points_data = get_all_umap_points(embedding_type)
    results = []
    for doc in all_points_data:
        coords = doc.get(umap_field)
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                continue
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                xy = np.asarray(coords, dtype=float)
                if not np.all(np.isfinite(xy)):
                    continue
                dist = float(np.linalg.norm(xy - mean_vector))
                if not math.isfinite(dist):
                    continue
                score = 1.0 / (1.0 + dist)
                results.append({
                    "ID": str(doc.get("ID")) if doc.get("ID") else None,
                    "Title": doc.get("Title", ""),
                    "Abstract": doc.get("Abstract", ""),
                    "Authors": doc.get("Authors", []),
                    "Keywords": doc.get("Keywords", []),
                    "Source": doc.get("Source", ""),
                    "Year": doc.get("Year"),
                    "umap": doc.get("umap"),
                    "ada_umap": doc.get("ada_umap"),
                    "specter_umap": doc.get("specter_umap"),
                    "distance": dist,
                    "score": score,
                })
            except Exception:
                continue
    results.sort(key=lambda x: x["distance"])
    return results[:limit]

def query_similar_doc_by_paper(paper: dict, embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    return query_similar_doc_by_embedding_full([paper], embedding_type, limit, lang_filter)

def get_all_umap_points(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_prod"))
    if not coll:
        return []
    try:
        rows = _query_all_batched(coll, _SCALAR_FIELDS, desc="UMAP")
        return format_umap_points(rows or [])
    except Exception as e:
        logging.error(f"Failed to load UMAP points from Zilliz: {e}", exc_info=True)
        return []


def get_all_static_cache_rows(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[dict]:
    """Fetch all fields needed for metadata and UMAP cache snapshots in one pass."""
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_prod"))
    if not coll:
        return []
    try:
        rows = _query_all_batched(coll, _SCALAR_FIELDS, desc="static cache") or []
        return [format_doc_for_frontend(row) for row in rows if row]
    except Exception as e:
        logging.error(f"Failed to fetch static cache rows from Zilliz: {e}")
        return []


def get_all_metadatas(
    embedding_type: str = DEFAULT_EMBEDDING_MODEL,
    limit: Optional[int] = None,
) -> List[dict]:
    """Return metadata rows from Zilliz (batched). Optional limit samples for cheap calls."""
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_prod"))
    if not coll:
        return []
    try:
        if limit is not None:
            safe_limit = max(1, int(limit))
            res = coll.query(
                expr='paper_uid != ""',
                output_fields=_SCALAR_FIELDS,
                limit=safe_limit,
            )
        else:
            res = _query_all_batched(coll, _SCALAR_FIELDS, desc="metadata")
        return [format_doc_for_frontend(row) for row in (res or []) if row]
    except Exception as e:
        logging.error(f"Failed to fetch metadatas from Zilliz: {e}")
        return []

def _aggregate_count(field: str, embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    docs = get_all_metadatas(embedding_type)
    counter = {}
    for doc in docs:
        values = doc.get(field)
        if values is None:
            continue
        if field in ("Authors", "Keywords"):
            values = _parse_string_list(values)
        elif not isinstance(values, list):
            values = [values]
        for v in values:
            if v:
                key_str = str(v).strip()
                if key_str:
                    counter[key_str] = counter.get(key_str, 0) + 1
    return sorted([{"_id": k, "count": v} for k, v in counter.items()], key=lambda x: -x["count"])

def get_distinct_authors_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return _aggregate_count("Authors", embedding_type)
def get_distinct_sources_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return _aggregate_count("Source", embedding_type)
def get_distinct_keywords_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return _aggregate_count("Keywords", embedding_type)
def get_distinct_years_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return sorted(_aggregate_count("Year", embedding_type), key=lambda x: x["_id"])
def get_distinct_titles_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return _aggregate_count("Title", embedding_type)
def get_distinct_citation_counts_with_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    return _aggregate_count("CitationCounts", embedding_type)

def get_distinct_authors(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    docs = get_all_metadatas(embedding_type)
    authors_set = set()
    for doc in docs:
        authors = doc.get("Authors", "")
        if isinstance(authors, str):
            for a in authors.split(","):
                if a.strip():
                    authors_set.add(a.strip())
        elif isinstance(authors, list):
            for a in authors:
                if a and isinstance(a, str):
                    authors_set.add(a.strip())
    return list(authors_set)

def get_distinct_sources(embedding_type: str = DEFAULT_EMBEDDING_MODEL):
    docs = get_all_metadatas(embedding_type)
    formatted = [format_doc_for_frontend(d) for d in docs]
    return list(set(d.get("Source") for d in formatted if d.get("Source")))

def get_distinct_keywords(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    docs = get_all_metadatas(embedding_type)
    keywords_set = set()
    for doc in docs:
        keywords = doc.get("Keywords", "")
        if isinstance(keywords, str):
            for k in keywords.split(","):
                if k.strip():
                    keywords_set.add(k.strip())
        elif isinstance(keywords, list):
            for k in keywords:
                if k and isinstance(k, str):
                    keywords_set.add(k.strip())
    return list(keywords_set)

def get_distinct_years(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[int]:
    docs = get_all_metadatas(embedding_type)
    return sorted(set(doc.get("Year") for doc in docs if doc.get("Year") is not None))

def get_distinct_titles(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    docs = get_all_metadatas(embedding_type)
    return list(set(doc.get("Title") for doc in docs if doc.get("Title")))

def get_distinct_citation_counts(embedding_type: str = DEFAULT_EMBEDDING_MODEL) -> List[int]:
    docs = get_all_metadatas(embedding_type)
    return sorted(set(doc.get("CitationCounts") for doc in docs if doc.get("CitationCounts") is not None))
