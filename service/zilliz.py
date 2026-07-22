"""
Zilliz Cloud (Milvus-compatible) vector database service.
"""
import json
import math
import sys
import numpy as np
from typing import List, Dict, Any, Optional

from tqdm import tqdm

import config
from model.const import EMBED
from model.retrieval import (
    DEFAULT_RETRIEVAL_PROFILE,
    RETRIEVAL_PROFILES,
    get_retrieval_profile,
)
from model.paper import GetPapersRequest
from logger_config import get_logger
from repositories.zilliz.connection import (
    ensure_collection_loaded,
    get_client as _get_milvus_client,
)
from repositories.zilliz.query_expressions import (
    build_paper_query_expr as _build_paper_query_expr,
    escape_like as _escape_like,
    ids_to_expr as _ids_to_expr,
    query_has_filters as _query_has_filters,
    where_to_expr as _zilliz_where_to_expr,
)
from repositories.zilliz.mappers import (
    SCALAR_FIELDS as _SCALAR_FIELDS,
    entity_to_metadata as _entity_to_meta,
    paper_to_api_response as format_doc_for_frontend,
    row_to_metadata as _row_to_meta,
    rows_to_umap_points as format_umap_points,
    search_hit_to_id_and_distance,
)
from service.metadata_normalizer import parse_string_list

logging = get_logger()

# Collection name mapping; include string keys for agent_tools
COLLECTION_MAPPING = {
    name: profile.collection for name, profile in RETRIEVAL_PROFILES.items()
}


def _profile_or_log(embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
    profile = get_retrieval_profile(embedding_type)
    if not profile:
        logging.error("Unsupported retrieval profile: %s", embedding_type)
    return profile


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

    def search(self, *, data, anns_field, param, limit, output_fields):
        client = _get_milvus_client()
        if not client:
            return []
        return client.search(
            collection_name=self._collection_name,
            data=data,
            anns_field=anns_field,
            search_params=param,
            limit=limit,
            output_fields=output_fields,
        )

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

# Expose for agent_tools
def get_client():
    """Return a client-like object for get_or_create_collection(name).get(include=[...])."""
    return _ZillizClientCompat()

class _ZillizClientCompat:
    def get_or_create_collection(self, collection_name: str):
        return _ZillizCollectionCompat(collection_name)

class _ZillizCollectionCompat:
    def __init__(self, name: str):
        self._name = name
    def get(self, include=None, ids=None, where=None):
        if where:
            expr = _zilliz_where_to_expr(where)
        elif ids:
            expr = _ids_to_expr(ids)
        else:
            expr = 'paper_uid != ""'
        coll = _get_collection(self._name)
        if not coll:
            return {"metadatas": [], "documents": []}
        try:
            res = _query_by_expr_batched(coll, expr, _SCALAR_FIELDS)
        except Exception as e:
            logging.error(f"Zilliz query failed: {e}")
            return {"metadatas": [], "documents": []}
        metadatas = [r for r in res] if res else []
        return {"metadatas": metadatas, "documents": []}

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

def load_all_papers_to_cache(embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
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

def get_cached_papers(embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        return []
    if collection_name not in _all_papers_cache:
        load_all_papers_to_cache(embedding_type)
    return _all_papers_cache.get(collection_name, [])

# --- Query schema & matching ---
def normalize_text(text: str) -> str:
    return str(text or "").strip().lower().rstrip(".!?")


def _parse_string_list(value) -> List[str]:
    return parse_string_list(value)


def _normalized_list(values) -> List[str]:
    normalized = []
    for value in _parse_string_list(values):
        norm = normalize_text(value)
        if norm:
            normalized.append(norm)
    return normalized


def _count_matching_entities(coll, expr: str) -> Optional[int]:
    """Return the exact number of entities matching a Milvus scalar expression."""
    try:
        rows = coll.query(expr=expr, output_fields=["count(*)"]) or []
        if not rows:
            return 0
        row = rows[0]
        for key in ("count(*)", "count()"):
            if key in row:
                return int(row[key])
        return int(next(iter(row.values())))
    except Exception as e:
        logging.warning(f"Zilliz count(*) failed for expr={expr!r}: {e}")
        return None


def _contains_token_phrase(container: str, needle: str) -> bool:
    if not container or not needle:
        return False
    if container == needle:
        return True
    if needle in container:
        return True
    container_tokens = set(container.replace("-", " ").split())
    needle_tokens = [tok for tok in needle.replace("-", " ").split() if tok]
    return bool(needle_tokens) and all(tok in container_tokens for tok in needle_tokens)


def _author_matches(candidate: str, query_author: str) -> bool:
    candidate_norm = normalize_text(candidate)
    query_norm = normalize_text(query_author)
    if not candidate_norm or not query_norm:
        return False
    if candidate_norm == query_norm:
        return True

    candidate_tokens = [tok for tok in candidate_norm.replace("-", " ").split() if tok]
    query_tokens = [tok for tok in query_norm.replace("-", " ").split() if tok]
    if not candidate_tokens or not query_tokens:
        return False

    # Treat reordered full names as equivalent, but avoid loose substring matches
    return sorted(candidate_tokens) == sorted(query_tokens)

def match_doc(doc, query: GetPapersRequest):
    doc_id_str = str(doc.get("ID")) if doc.get("ID") is not None else None
    if query.title:
        # Support comma-separated keywords with AND logic (all keywords must match)
        keywords = [k.strip() for k in query.title.split(',') if k.strip()]
        d_title = normalize_text(doc.get("Title", ""))

        # Check if all keywords are present in the title
        if not all(normalize_text(kw) in d_title for kw in keywords):
            return False

    if query.abstract:
        # Support comma-separated keywords with AND logic (all keywords must match)
        keywords = [k.strip() for k in query.abstract.split(',') if k.strip()]
        d_abstract = str(doc.get("Abstract", "")).lower()

        # Check if all keywords are present in the abstract
        if not all(kw.lower() in d_abstract for kw in keywords):
            return False
    doc_authors = _normalized_list(doc.get("Authors", []))
    if query.author and not any(
        any(_author_matches(author, q_author) for author in doc_authors)
        for q_author in query.author
    ):
        return False
    src = query.source
    if src is not None:
        if isinstance(src, list):
            if not any(s.lower() in str(doc.get("Source", "")).lower() for s in src):
                return False
        else:
            if src.lower() not in str(doc.get("Source", "")).lower():
                return False
    doc_keywords = _normalized_list(doc.get("Keywords", []))
    if query.keyword and not any(
        any(_contains_token_phrase(keyword, normalize_text(q_keyword)) for keyword in doc_keywords)
        for q_keyword in query.keyword
    ):
        return False
    try:
        doc_year = int(doc.get("Year", 0))
        if query.min_year and doc_year < query.min_year:
            return False
        if query.max_year and doc_year > query.max_year:
            return False
    except Exception:
        pass
    try:
        doc_citation_counts = int(doc.get("CitationCounts", 0))
        if getattr(query, "min_citation_counts", None) is not None and doc_citation_counts < query.min_citation_counts:
            return False
        if getattr(query, "max_citation_counts", None) is not None and doc_citation_counts > query.max_citation_counts:
            return False
    except Exception:
        pass
    if query.id_list and (doc_id_str is None or str(doc_id_str) not in [str(i) for i in query.id_list]):
        return False
    return True

def query_docs(query: GetPapersRequest, embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
    """Return one Zilliz-backed page of papers without a process-wide cache."""
    try:
        profile = _profile_or_log(embedding_type)
        if not profile:
            return {"papers": [], "total": 0}
        coll = _get_collection(profile.collection)
        if not coll:
            return {"papers": [], "total": 0}

        # Keep this guard in the service as well as the HTTP route: agent code
        # also calls query_docs() directly.
        limit = min(max(int(query.limit or 100), 1), 100)
        offset = max(int(query.offset or 0), 0)
        expr = _build_paper_query_expr(query)
        rows = coll.query(
            expr=expr,
            output_fields=_SCALAR_FIELDS,
            limit=limit + 1,
            offset=offset,
        ) or []
        has_more = len(rows) > limit
        papers = [format_doc_for_frontend(row) for row in rows[:limit] if row]

        if not _query_has_filters(query):
            # num_entities: total rows in the Zilliz collection (full corpus size).
            try:
                total_count = int(coll.num_entities)
            except Exception:
                total_count = offset + len(papers) + int(has_more)
        else:
            total_count = _count_matching_entities(coll, expr)
            if total_count is None:
                total_count = offset + len(papers) + int(has_more)

        return {"papers": papers, "total": total_count, "has_more": has_more}
    except Exception as e:
        logging.error(f"Error in query_docs(): {e}", exc_info=True)
        return {"papers": [], "total": 0}

def query_docs_with_embeddings(query: GetPapersRequest, embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
    return query_docs(query, embedding_type=embedding_type)

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
        doc["Sim"] = float(sim)
        normalized.append(doc)
    return normalized

def query_doc_by_id(_id: str, embedding_type: str = DEFAULT_RETRIEVAL_PROFILE):
    profile = _profile_or_log(embedding_type)
    if not profile:
        return None
    coll = _get_collection(profile.collection)
    if not coll:
        return None
    try:
        res = coll.query(expr=f'paper_uid == "{str(_id).replace(chr(34), "")}"', output_fields=_SCALAR_FIELDS, limit=1)
        if res and len(res) > 0:
            return format_doc_for_frontend(res[0])
    except Exception as e:
        logging.error(f"Error fetching doc ID {_id} from Zilliz: {e}", exc_info=True)
    return None

def query_doc_by_title(title: str, embedding_type: str = DEFAULT_RETRIEVAL_PROFILE) -> list:
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

def query_doc_by_ids(ids: List[str], embedding_type: str = DEFAULT_RETRIEVAL_PROFILE) -> List[dict]:
    if not ids:
        return []
    profile = _profile_or_log(embedding_type)
    if not profile:
        return []
    coll = _get_collection(profile.collection)
    if not coll:
        return []
    try:
        expr = _ids_to_expr(ids)
        # Use scalar fields only so Title, Abstract, etc. are returned (embedding not needed)
        res = coll.query(expr=expr, output_fields=_SCALAR_FIELDS, limit=len(ids) + 100)
        return [format_doc_for_frontend(r) for r in (res or []) if r]
    except Exception as e:
        logging.error(f"Error in query_doc_by_ids(): {e}", exc_info=True)
        return []

def query_doc_by_embedding(
    paper_ids: Optional[List[str]],
    embedding: List[float],
    embedding_type: str,
    limit: int,
    lang_filter: Dict = None,
) -> List[Dict]:
    if paper_ids is None:
        paper_ids = []
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    profile = _profile_or_log(embedding_type)
    if not profile:
        return []
    if len(embedding) != profile.dimension:
        logging.error(
            "Query vector has %s dimensions for profile %s; expected %s",
            len(embedding), profile.name, profile.dimension,
        )
        return []
    coll = _get_collection(profile.collection)
    if not coll:
        return []
    # Request extra candidates for better precision (then trim to limit after excluding paper_ids)
    mult = getattr(config, "ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER", 1.5)
    top_k = max(limit + len(paper_ids) + 5, int(limit * mult))
    index_type = getattr(config, "ZILLIZ_INDEX_TYPE", "IVF_FLAT")
    if index_type.upper() == "HNSW":
        ef = getattr(config, "ZILLIZ_SEARCH_EF", 64)
        search_params = {"metric_type": profile.metric, "params": {"ef": ef}}
    else:
        nprobe = getattr(config, "ZILLIZ_SEARCH_NPROBE", 128)
        search_params = {"metric_type": profile.metric, "params": {"nprobe": nprobe}}
    # Step 1: vector search returns only ID + distance (Zilliz often omits scalars in search results)
    try:
        res = coll.search(
            data=[embedding],
            anns_field=profile.vector_field,
            param=search_params,
            limit=min(top_k, 16384),
            output_fields=["paper_uid"],
        )
    except Exception as e:
        logging.error(f"Zilliz search failed: {e}", exc_info=True)
        return []
    if not res or len(res) == 0:
        return []
    hits = res[0]
    id_to_distance = {}
    order_ids = []
    for hit in hits:
        doc_id, distance = search_hit_to_id_and_distance(hit)
        if not doc_id or (paper_ids and doc_id in [str(p) for p in paper_ids]):
            continue
        if doc_id not in id_to_distance:
            id_to_distance[doc_id] = distance
            order_ids.append(doc_id)
        if len(order_ids) >= limit:
            break
    if not order_ids:
        return []
    # Step 2: fetch full metadata (Title, Abstract, etc.) by ID via query API
    full_docs = query_doc_by_ids(order_ids, embedding_type)
    doc_by_id = {str(d.get("ID")): d for d in full_docs if d.get("ID")}
    docs = []
    for doc_id in order_ids:
        d = doc_by_id.get(doc_id)
        if not d:
            continue
        dist = id_to_distance.get(doc_id)
        if dist is not None:
            d["_score"] = float(dist)
            try:
                d["score"] = float(dist)
                d["Sim"] = d["score"]
                d["_Sim"] = d["score"]
            except Exception:
                pass
        docs.append(d)
    return docs

def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    paper_ids_to_exclude = [str(p.get("ID")) for p in papers if p.get("ID")]
    profile = _profile_or_log(embedding_type)
    if not profile:
        return []
    coll = _get_collection(profile.collection)
    if not coll:
        return []
    try:
        expr = _ids_to_expr(paper_ids_to_exclude)
        res = coll.query(expr=expr, output_fields=[profile.vector_field], limit=len(paper_ids_to_exclude) + 100)
    except Exception as e:
        logging.error(f"Failed to fetch embeddings from Zilliz: {e}", exc_info=True)
        return []
    vectors_for_mean = []
    for r in (res or []):
        emb = r.get(profile.vector_field)
        if isinstance(emb, (list, np.ndarray)) and (np.any(emb) if hasattr(emb, "__len__") else emb):
            vectors_for_mean.append(emb if isinstance(emb, list) else emb.tolist())
    if not vectors_for_mean:
        return []
    mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
    return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)

def query_similar_doc_by_embedding_2d(
    papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None
):
    umap_field = {EMBED.ADA: "ada_umap", EMBED.SPECTER: "specter_umap"}.get(embedding_type)
    if not umap_field:
        return []
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
                    "ada_umap": doc.get("ada_umap"),
                    "specter_umap": doc.get("specter_umap"),
                    "distance": dist,
                    "score": score,
                    "Sim": score,
                })
            except Exception:
                continue
    results.sort(key=lambda x: x["distance"])
    return results[:limit]

def query_similar_doc_by_paper(paper: dict, embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    return query_similar_doc_by_embedding_full([paper], embedding_type, limit, lang_filter)

_UMAP_FIELDS = [
    "ID",
    "Title",
    "Year",
    "Source",
    "ada_umap",
    "specter_umap",
]
_METADATA_FIELDS = [
    "ID",
    "Title",
    "Authors",
    "Keywords",
    "Source",
    "Year",
    "CitationCounts",
]
_STATIC_CACHE_FIELDS = list(dict.fromkeys(_METADATA_FIELDS + _UMAP_FIELDS))


def get_all_umap_points(embedding_type: str = EMBED.SPECTER):
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    try:
        rows = _query_all_batched(coll, _UMAP_FIELDS, desc="UMAP")
        return format_umap_points(rows or [])
    except Exception as e:
        logging.error(f"Failed to load UMAP points from Zilliz: {e}", exc_info=True)
        return []


def get_all_static_cache_rows(embedding_type: str = EMBED.SPECTER) -> List[dict]:
    """Fetch all fields needed for metadata and UMAP cache snapshots in one pass."""
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    try:
        return list(
            _query_all_batched(coll, _STATIC_CACHE_FIELDS, desc="static cache") or []
        )
    except Exception as e:
        logging.error(f"Failed to fetch static cache rows from Zilliz: {e}")
        return []


def get_all_metadatas(
    embedding_type: str = EMBED.SPECTER,
    limit: Optional[int] = None,
) -> List[dict]:
    """Return metadata rows from Zilliz (batched). Optional limit samples for cheap calls."""
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    try:
        if limit is not None:
            safe_limit = max(1, int(limit))
            res = coll.query(
                expr='ID != ""',
                output_fields=_METADATA_FIELDS,
                limit=safe_limit,
            )
        else:
            res = _query_all_batched(coll, _METADATA_FIELDS, desc="metadata")
        return list(res or [])
    except Exception as e:
        logging.error(f"Failed to fetch metadatas from Zilliz: {e}")
        return []

def _aggregate_count(field: str, embedding_type: str = EMBED.SPECTER) -> List[Dict[str, Any]]:
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

def get_distinct_authors_with_counts(embedding_type: str = EMBED.SPECTER):
    return _aggregate_count("Authors", embedding_type)
def get_distinct_sources_with_counts(embedding_type: str = EMBED.SPECTER):
    return _aggregate_count("Source", embedding_type)
def get_distinct_keywords_with_counts(embedding_type: str = EMBED.SPECTER):
    return _aggregate_count("Keywords", embedding_type)
def get_distinct_years_with_counts(embedding_type: str = EMBED.SPECTER):
    return sorted(_aggregate_count("Year", embedding_type), key=lambda x: x["_id"])
def get_distinct_titles_with_counts(embedding_type: str = EMBED.SPECTER):
    return _aggregate_count("Title", embedding_type)
def get_distinct_citation_counts_with_counts(embedding_type: str = EMBED.SPECTER):
    return _aggregate_count("CitationCounts", embedding_type)

def get_distinct_authors(embedding_type: str = EMBED.SPECTER) -> List[str]:
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

def get_distinct_sources(embedding_type: str = EMBED.SPECTER):
    docs = get_all_metadatas(embedding_type)
    formatted = [format_doc_for_frontend(d) for d in docs]
    return list(set(d.get("Source") for d in formatted if d.get("Source")))

def get_distinct_keywords(embedding_type: str = EMBED.SPECTER) -> List[str]:
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

def get_distinct_years(embedding_type: str = EMBED.SPECTER) -> List[int]:
    docs = get_all_metadatas(embedding_type)
    return sorted(set(doc.get("Year") for doc in docs if doc.get("Year") is not None))

def get_distinct_titles(embedding_type: str = EMBED.SPECTER) -> List[str]:
    docs = get_all_metadatas(embedding_type)
    return list(set(doc.get("Title") for doc in docs if doc.get("Title")))

def get_distinct_citation_counts(embedding_type: str = EMBED.SPECTER) -> List[int]:
    docs = get_all_metadatas(embedding_type)
    return sorted(set(doc.get("CitationCounts") for doc in docs if doc.get("CitationCounts") is not None))

# For agent_tools
_zilliz_client = get_client()
