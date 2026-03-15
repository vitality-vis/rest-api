"""
Zilliz Cloud (Milvus-compatible) vector database service.
"""
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional

import config
from model.const import EMBED
from model.query import QuerySchema
from logger_config import get_logger

logging = get_logger()

# Collection name mapping; include string keys for agent_tools
COLLECTION_MAPPING = {
    EMBED.ADA: "paper_ada_localized",
    EMBED.GLOVE: "paper_glove_localized",
    EMBED.SPECTER: "paper_specter",
    "specter": "paper_specter",
    "ada": "paper_ada_localized",
    "glove": "paper_glove_localized",
}

# Lazy pymilvus imports and connection
_pymilvus = None
_connected = False
_collection_cache = {}  # name -> Collection

def _get_pymilvus():
    global _pymilvus
    if _pymilvus is None:
        try:
            from pymilvus import connections, Collection, utility
            _pymilvus = (connections, Collection, utility)
        except ImportError as e:
            logging.error(f"pymilvus not installed: {e}")
    return _pymilvus

def _ensure_connection():
    global _connected
    if _connected:
        return True
    if not config.ZILLIZ_URI or not config.ZILLIZ_TOKEN:
        logging.error("ZILLIZ_URI and ZILLIZ_TOKEN must be set (e.g. in .env)")
        return False
    pym = _get_pymilvus()
    if not pym:
        return False
    connections, _, _ = pym
    try:
        connections.connect(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
        _connected = True
        logging.info("Zilliz Cloud connection established")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Zilliz: {e}", exc_info=True)
        return False

def _get_collection(collection_name: str):
    """Get and load collection; cache the instance."""
    if not _ensure_connection():
        return None
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]
    pym = _get_pymilvus()
    if not pym:
        return None
    _, Collection, utility = pym
    try:
        if not utility.has_collection(collection_name):
            logging.error(f"Collection '{collection_name}' does not exist. Run load_to_zilliz.py first.")
            return None
        coll = Collection(collection_name)
        coll.load()
        _collection_cache[collection_name] = coll
        return coll
    except Exception as e:
        logging.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        return None

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
            expr = 'ID != ""'
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

def _ids_to_expr(ids: List[str]) -> str:
    if not ids:
        return 'ID != ""'
    escaped = [f'"{str(i).replace(chr(34), "")}"' for i in ids]
    return "ID in [" + ", ".join(escaped) + "]"

_SCALAR_FIELDS = ["ID", "Title", "Abstract", "Authors", "Keywords", "Source", "Year", "CitationCounts", "Lang", "ada_umap", "glove_umap", "specter_umap"]


def _entity_to_meta(entity) -> Optional[Dict[str, Any]]:
    """Convert pymilvus hit.entity to a plain dict with scalar fields (Title, Abstract, etc.)."""
    if entity is None:
        return None
    meta = {}
    if isinstance(entity, dict):
        for k, v in entity.items():
            if k in _SCALAR_FIELDS or k == "_score":
                meta[k] = v
            elif k.lower() in [f.lower() for f in _SCALAR_FIELDS]:
                meta[next(f for f in _SCALAR_FIELDS if f.lower() == k.lower())] = v
    else:
        for key in _SCALAR_FIELDS:
            if hasattr(entity, "get") and callable(entity.get):
                meta[key] = entity.get(key) or entity.get(key.lower())
            else:
                meta[key] = getattr(entity, key, None) or getattr(entity, key.lower(), None)
    if meta.get("id") is not None and meta.get("ID") is None:
        meta["ID"] = meta["id"]
    return meta if meta.get("ID") is not None else None

_QUERY_BATCH_SIZE = 2000
_ID_BATCH_SIZE = 5000

def _query_all_batched(coll, output_fields: List[str]):
    """
    Fetch all rows in batches to avoid gRPC 'message larger than max' (4MB).
    Phase 1: collect all IDs (ID only, small response). Phase 2: for each chunk
    of IDs query full rows with 'ID in [chunk]' (small request/response).
    """
    return _query_by_expr_batched(coll, 'ID != ""', output_fields)

def _query_by_expr_batched(coll, base_expr: str, output_fields: List[str]):
    """
    Fetch all rows matching base_expr in batches to stay under gRPC 4MB.
    Phase 1: collect IDs matching base_expr (ID only, batched with ID not in seen).
    Phase 2: fetch full rows by ID chunks.
    """
    all_ids = []
    seen_ids = []
    while True:
        if seen_ids:
            expr = f"({base_expr}) and {_zilliz_where_to_expr({'ID': {'$nin': seen_ids}})}"
        else:
            expr = base_expr
        try:
            res = coll.query(expr=expr, output_fields=["ID"], limit=_ID_BATCH_SIZE)
        except Exception as e:
            logging.warning(f"Batch ID fetch failed: {e}. Collected {len(all_ids)} IDs.")
            break
        if not res:
            break
        ids_batch = [r["ID"] for r in res if r.get("ID")]
        all_ids.extend(ids_batch)
        seen_ids.extend(ids_batch)
        if len(res) < _ID_BATCH_SIZE:
            break
    if not all_ids:
        return []
    all_rows = []
    for i in range(0, len(all_ids), _QUERY_BATCH_SIZE):
        chunk = all_ids[i : i + _QUERY_BATCH_SIZE]
        expr = _ids_to_expr(chunk)
        try:
            res = coll.query(expr=expr, output_fields=output_fields, limit=len(chunk) + 10)
            all_rows.extend(res or [])
        except Exception as e:
            logging.warning(f"Batch row fetch failed: {e}")
    return all_rows

def _escape_like(s: str) -> str:
    """Escape % and _ for Milvus LIKE pattern."""
    s = str(s).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return s.replace('"', '\\"')

def _zilliz_where_to_expr(where: dict) -> str:
    if not where:
        return 'ID != ""'
    parts = []
    for k, v in where.items():
        if isinstance(v, dict):
            if "$eq" in v:
                parts.append(f'{k} == "{str(v["$eq"]).replace(chr(34), "")}"')
            elif "$in" in v:
                in_list = v["$in"]
                escaped = [f'"{str(i).replace(chr(34), "")}"' for i in in_list]
                parts.append(f"{k} in [{', '.join(escaped)}]")
            elif "$nin" in v:
                nin_list = v["$nin"]
                escaped = [f'"{str(i).replace(chr(34), "")}"' for i in nin_list]
                parts.append(f"{k} not in [{', '.join(escaped)}]")
            elif "$gte" in v:
                parts.append(f"{k} >= {int(v['$gte'])}")
            elif "$lte" in v:
                parts.append(f"{k} <= {int(v['$lte'])}")
            elif "$contains" in v:
                esc = _escape_like(v["$contains"])
                parts.append(f'{k} like "%{esc}%"')
            elif "$contains_all" in v:
                for val in v["$contains_all"]:
                    esc = _escape_like(val)
                    parts.append(f'{k} like "%{esc}%"')
        else:
            parts.append(f'{k} == "{str(v).replace(chr(34), "")}"')
    return " and ".join(parts) if parts else 'ID != ""'

# --- Cache ---
_all_papers_cache = {}

def load_all_papers_to_cache(embedding_type: str = EMBED.SPECTER):
    global _all_papers_cache
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        return
    coll = _get_collection(collection_name)
    if not coll:
        return
    try:
        logging.info(f"Loading all papers from {collection_name} into memory cache (batched)...")
        res = _query_all_batched(coll, _SCALAR_FIELDS)
        cached_papers = []
        for r in (res or []):
            cached_papers.append(format_doc_for_frontend(_row_to_meta(r)))
        _all_papers_cache[collection_name] = cached_papers
        logging.info(f"Cached {len(cached_papers)} papers for {collection_name}")
    except Exception as e:
        logging.error(f"Failed to load papers to cache: {e}", exc_info=True)
        _all_papers_cache[collection_name] = []

def _row_to_meta(row: dict) -> dict:
    """Convert Milvus row (field names as returned) to metadata dict."""
    return row

def get_cached_papers(embedding_type: str = EMBED.SPECTER):
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        return []
    if collection_name not in _all_papers_cache:
        load_all_papers_to_cache(embedding_type)
    return _all_papers_cache.get(collection_name, [])

# --- Query schema & matching ---
def normalize_text(text: str) -> str:
    return str(text or "").strip().lower().rstrip(".!?")

def match_doc(doc, query: QuerySchema):
    doc_id_str = str(doc.get("ID")) if doc.get("ID") is not None else None
    if query.title:
        q_title = normalize_text(query.title)
        d_title = normalize_text(doc.get("Title", ""))
        if q_title != d_title and q_title not in d_title:
            try:
                from rapidfuzz import fuzz
                if fuzz.ratio(q_title, d_title) < 80:
                    return False
            except ImportError:
                return False
    if query.abstract and query.abstract.lower() not in str(doc.get("Abstract", "")).lower():
        return False
    authors = doc.get("Authors", [])
    if isinstance(authors, str):
        authors = [authors]
    authors_str = " ".join(authors).lower()
    if query.author and not any(a.lower() in authors_str for a in query.author):
        return False
    src = query.source
    if src is not None:
        if isinstance(src, list):
            if not any(s.lower() in str(doc.get("Source", "")).lower() for s in src):
                return False
        else:
            if src.lower() not in str(doc.get("Source", "")).lower():
                return False
    keywords = doc.get("Keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords_str = " ".join(keywords).lower()
    if query.keyword and not any(k.lower() in keywords_str for k in query.keyword):
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

def query_docs(query: QuerySchema, embedding_type: str = EMBED.SPECTER):
    try:
        all_papers = get_cached_papers(embedding_type)
        if not all_papers:
            return {"papers": [], "total": 0}
        docs = [p for p in all_papers if match_doc(p, query)]
        total_count = len(docs)
        offset = int(query.offset or 0)
        limit = int(query.limit or 100)
        if limit == -1:
            paginated_docs = docs[offset:]
        else:
            paginated_docs = docs[offset : offset + limit]
        return {"papers": paginated_docs, "total": total_count}
    except Exception as e:
        logging.error(f"Error in query_docs(): {e}", exc_info=True)
        return {"papers": [], "total": 0}

def query_docs_with_embeddings(query: QuerySchema, embedding_type: str = EMBED.SPECTER):
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

def query_doc_by_id(_id: str, embedding_type: str = EMBED.SPECTER):
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return None
    try:
        res = coll.query(expr=f'ID == "{str(_id).replace(chr(34), "")}"', output_fields=_SCALAR_FIELDS, limit=1)
        if res and len(res) > 0:
            return format_doc_for_frontend(res[0])
    except Exception as e:
        logging.error(f"Error fetching doc ID {_id} from Zilliz: {e}", exc_info=True)
    return None

def query_doc_by_title(title: str, embedding_type: str = EMBED.SPECTER) -> list:
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

def query_doc_by_ids(ids: List[str], embedding_type: str = EMBED.SPECTER) -> List[dict]:
    if not ids:
        return []
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
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
    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"No collection mapped for embedding type: {embedding_type}")
        return []
    coll = _get_collection(collection_name)
    if not coll:
        return []
    # Request extra candidates for better precision (then trim to limit after excluding paper_ids)
    mult = getattr(config, "ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER", 1.5)
    top_k = max(limit + len(paper_ids) + 5, int(limit * mult))
    index_type = getattr(config, "ZILLIZ_INDEX_TYPE", "IVF_FLAT")
    if index_type.upper() == "HNSW":
        ef = getattr(config, "ZILLIZ_SEARCH_EF", 64)
        search_params = {"metric_type": "L2", "params": {"ef": ef}}
    else:
        nprobe = getattr(config, "ZILLIZ_SEARCH_NPROBE", 128)
        search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}
    # Step 1: vector search returns only ID + distance (Zilliz often omits scalars in search results)
    try:
        res = coll.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=min(top_k, 16384),
            output_fields=["ID"],
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
        entity = getattr(hit, "entity", hit)
        doc_id = None
        if hasattr(entity, "get") and entity.get("ID"):
            doc_id = str(entity.get("ID"))
        elif getattr(entity, "ID", None) is not None:
            doc_id = str(entity.ID)
        elif isinstance(entity, dict) and entity.get("id"):
            doc_id = str(entity.get("id"))
        if not doc_id or (paper_ids and doc_id in [str(p) for p in paper_ids]):
            continue
        if doc_id not in id_to_distance:
            id_to_distance[doc_id] = getattr(hit, "distance", None)
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
                d["score"] = 1.0 / (1.0 + float(dist))
                d["Sim"] = d["score"]
                d["_Sim"] = d["score"]
            except Exception:
                pass
        docs.append(d)
    return docs

def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    paper_ids_to_exclude = [str(p.get("ID")) for p in papers if p.get("ID")]
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    try:
        expr = _ids_to_expr(paper_ids_to_exclude)
        res = coll.query(expr=expr, output_fields=["embedding"], limit=len(paper_ids_to_exclude) + 100)
    except Exception as e:
        logging.error(f"Failed to fetch embeddings from Zilliz: {e}", exc_info=True)
        return []
    vectors_for_mean = []
    for r in (res or []):
        emb = r.get("embedding")
        if isinstance(emb, (list, np.ndarray)) and (np.any(emb) if hasattr(emb, "__len__") else emb):
            vectors_for_mean.append(emb if isinstance(emb, list) else emb.tolist())
    if not vectors_for_mean:
        return []
    mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
    return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)

def query_similar_doc_by_embedding_2d(
    papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None
):
    umap_field = {EMBED.ADA: "ada_umap", EMBED.GLOVE: "glove_umap", EMBED.SPECTER: "specter_umap"}.get(embedding_type)
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
                    "glove_umap": doc.get("glove_umap"),
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

def get_all_umap_points(embedding_type: str = EMBED.SPECTER):
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    umap_fields = ["ID", "Title", "Year", "Source", "ada_umap", "glove_umap", "specter_umap"]
    try:
        res = _query_all_batched(coll, umap_fields)
        points = []
        for r in (res or []):
            def parse_coords(val):
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return None
                return val
            points.append({
                "ID": str(r.get("ID")) if r.get("ID") else None,
                "Title": r.get("Title", ""),
                "Year": r.get("Year"),
                "Source": r.get("Source", ""),
                "ada_umap": parse_coords(r.get("ada_umap")),
                "glove_umap": parse_coords(r.get("glove_umap")),
                "specter_umap": parse_coords(r.get("specter_umap")),
            })
        return points
    except Exception as e:
        logging.error(f"Failed to load UMAP points from Zilliz: {e}", exc_info=True)
        return []

def get_all_metadatas(embedding_type: str = EMBED.SPECTER) -> List[dict]:
    """Return all metadata from Zilliz for the given embedding type (batched, no embedding vector)."""
    coll = _get_collection(COLLECTION_MAPPING.get(embedding_type, "paper_specter"))
    if not coll:
        return []
    try:
        res = _query_all_batched(coll, _SCALAR_FIELDS)
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
        if not isinstance(values, list):
            if field in ("Authors", "Keywords") and isinstance(values, str):
                values = [v.strip() for v in values.split(",") if v.strip()]
            else:
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

def format_doc_for_frontend(doc: dict, score_key: str = "_score") -> dict:
    distance = doc.get(score_key)
    final_sim_value = 0.0
    try:
        float_distance = float(distance) if distance is not None else float("nan")
        if not math.isnan(float_distance):
            final_sim_value = 1.0 / (1.0 + float_distance)
    except Exception:
        pass
    # Support both "Title" and "title" (some backends return lowercase)
    def _get(k):
        return doc.get(k) or doc.get(k.lower()) or ""
    authors = doc.get("Authors") or doc.get("authors") or ""
    if isinstance(authors, list):
        authors = [a.strip() for a in authors if a and isinstance(a, str)]
    elif isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    keywords = doc.get("Keywords") or doc.get("keywords") or ""
    if isinstance(keywords, list):
        keywords = [k.strip() for k in keywords if k and isinstance(k, str)]
    elif isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    def parse_coords(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return None
        return val
    return {
        "ID": doc.get("ID") or doc.get("id"),
        "Title": _get("Title"),
        "Abstract": _get("Abstract"),
        "Authors": authors,
        "Keywords": keywords,
        "Source": _get("Source"),
        "Year": doc.get("Year") if doc.get("Year") is not None else doc.get("year"),
        "CitationCounts": doc.get("CitationCounts") if doc.get("CitationCounts") is not None else doc.get("citationcounts"),
        "_Sim": final_sim_value,
        "Sim": final_sim_value,
        "score": final_sim_value,
        "ada_umap": parse_coords(doc.get("ada_umap")),
        "glove_umap": parse_coords(doc.get("glove_umap")),
        "specter_umap": parse_coords(doc.get("specter_umap")),
    }

# For agent_tools
_zilliz_client = get_client()
