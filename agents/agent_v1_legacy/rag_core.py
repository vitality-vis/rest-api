"""
Legacy search agent helpers: session docs, CrossEncoder rerank, formatting.

Owned by ``agents.agent_v1_legacy``. Shared paper retrieval goes through
``service.search``; this module is agent-specific.
"""
import json
import logging
from copy import deepcopy
from typing import List, Dict, Any, Sequence, Optional
from langchain_core.documents import Document
from model.paper import SearchRequest
from service.search import search
from sentence_transformers import CrossEncoder

# How many ranked papers to send to the frontend paper list after a search.
# TODO: Replace this fixed cap with score-based truncation (relative threshold /
# elbow on cross-encoder scores), once cutoffs are tuned. Keep retrieve/rerank
# candidate pools larger than this emit limit.
EMIT_PAPER_LIMIT = 100

# Machine-readable block appended to the chat stream (stripped in the UI).
PAPERS_PAYLOAD_START = "[[VITALITY_PAPERS_JSON]]"
PAPERS_PAYLOAD_END = "[[/VITALITY_PAPERS_JSON]]"

# First N papers included in the tool text for the LLM / chat narrative.
CHAT_PREVIEW_LIMIT = 5
def _doc_paper_id(doc: Any) -> Optional[str]:
    if isinstance(doc, dict):
        md = doc.get("metadata", doc)
    else:
        md = getattr(doc, "metadata", None) or {}
    raw = md.get("id") or md.get("ID") or md.get("paper_uid")
    if raw is None:
        return None
    paper_id = str(raw).strip()
    return paper_id or None


def paper_ids_for_emit(docs: Sequence[Any], *, limit: int = EMIT_PAPER_LIMIT) -> List[str]:
    """Ordered unique paper IDs to send to the frontend (capped)."""
    ids: List[str] = []
    seen: set[str] = set()
    for doc in docs or []:
        paper_id = _doc_paper_id(doc)
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ids.append(paper_id)
        if len(ids) >= limit:
            break
    return ids


def format_papers_payload(docs: Sequence[Any], *, limit: int = EMIT_PAPER_LIMIT) -> str:
    """Build a hidden stream marker with ranked paper IDs for the frontend.

    Shape (forward-compatible with storing loaded papers on chat messages later):
      {
        "ranked_ids": [...],   # full ranked list for this turn (capped)
        "count_known": false   # not a corpus total
      }
    """
    ranked_ids = paper_ids_for_emit(docs, limit=limit)
    if not ranked_ids:
        return ""
    body = json.dumps(
        {
            "ranked_ids": ranked_ids,
            # Backward-compatible alias while clients migrate.
            "ids": ranked_ids,
            "count_known": False,
        },
        separators=(",", ":"),
    )
    return f"\n\n{PAPERS_PAYLOAD_START}{body}{PAPERS_PAYLOAD_END}"


def save_session_docs(chat_id: str, docs: List[Document]) -> None:
    """
    Save retrieved docs into the current chat session so they can be reused
    (e.g., for follow-up questions about already listed papers).
    """
    from service.session_state import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return

    # --- Append docs to per-turn buffer ---
    turn_buffer = session.setdefault("_turn_docs", [])
    turn_buffer.extend(deepcopy(docs or []))

    # --- Keep a flat 'docs' cache for quick lookup by tools ---
    existing = session.get("docs", [])
    session["docs"] = existing + deepcopy(docs or [])

    # --- Keep structured memory synced (optional) ---
    mem = session.get("mem")
    if mem:
        mem.set_docs(turn_buffer)


def get_session_docs(chat_id: str) -> List[Document]:
    """Return current docs for this chat (fallback to memory cache if empty)."""
    from service.session_state import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return []

    docs = session.get("docs", [])
    if not docs:
        mem = session.get("mem")
        if mem and getattr(mem, "doc_cache", None):
            docs = mem.doc_cache
    return deepcopy(docs)


def clear_session_docs(chat_id: str) -> None:
    """Clear doc cache for one chat (and structured memory if exists)."""
    from service.session_state import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return

    session["docs"] = []
    logging.info(f"[rag_core] Cleared docs for chat_id={chat_id}")

    # Also clear memory doc cache if present
    mem = session.get("mem")
    if mem and hasattr(mem, "clear_docs"):
        try:
            mem.clear_docs()
            logging.info(f"[rag_core] Cleared structured memory docs for chat_id={chat_id}")
        except Exception as e:
            logging.error(f"[rag_core] MemoryManager clear failed: {e}")

# =====================================================
# Embedding + Zilliz setup
# =====================================================
CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def format_docs(docs: Sequence[Document], *, include_abstract: bool = True, include_score: bool = True) -> str:
    """
    Format retrieved or recalled docs into Markdown for LLM context (memory-aware).

    - Handles both `Document` and raw `dict` entries.
    - Truncates long abstracts to stay token-safe when include_abstract=True.
    - Optionally includes stable IDs for follow-ups.
    - Set include_abstract=False, include_score=False for compact paper lists.
    """
    if not docs:
        return "_(No documents found or remembered.)_"

    formatted_blocks = []
    for i, doc in enumerate(docs):
        # Support both dict and LangChain Document
        if isinstance(doc, dict):
            md = doc.get("metadata", doc)
        else:
            md = getattr(doc, "metadata", {}) or {}

        # Normalize key casing
        md_l = {str(k).lower(): v for k, v in md.items()}

        # Stable ID for later recall
        doc_id = str(md_l.get("id", f"doc_{i}"))

        # Authors normalization
        authors_raw = md_l.get("authors", "")
        authors = ", ".join(authors_raw) if isinstance(authors_raw, list) else authors_raw

        # Abstract (omit for compact paper lists)
        abstract = (md_l.get("abstract", "") or "") if include_abstract else ""

        # Score / ranking
        score_val = md_l.get("score", 0.0)
        try:
            score_val = float(score_val)
        except Exception:
            score_val = 0.0

        # Format block
        block = (
            f"- **Title:** {md_l.get('title', '(No title)')} [[ID:{doc_id}]]\n"
            f"  - Authors: {authors or '(Unknown)'}\n"
            f"  - Year: {md_l.get('year', '(N/A)')}\n"
            f"  - Source: {md_l.get('source', '(N/A)')}\n"
        )
        if include_score:
            block += f"  - Score: {score_val:.4f}\n"
        if abstract:
            block += f"  - Abstract: {abstract}\n"
        formatted_blocks.append(block)

    # Optionally store formatted text in memory for the generator
    formatted_text = "\n".join(formatted_blocks)
    return formatted_text


def _rows_to_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw rows into LangChain Documents."""
    docs = []
    for i, m in enumerate(items or []):
        docs.append(
            Document(
                page_content=m.get("Abstract", "") or m.get("Title", ""),
                metadata={
                    "title": m.get("Title", ""),
                    "abstract": m.get("Abstract", ""),
                    "authors": m.get("Authors", []),
                    "keywords": m.get("Keywords", []),
                    "source": m.get("Source", ""),
                    "year": m.get("Year", ""),
                    "id": str(m.get("ID", f"doc_{i}")),
                    "score": float(m.get("score", 0.0)),
                },
            )
        )
    return docs

# =====================================================
# Query Functions
# =====================================================

def _run_metadata_search(plan, chat_id: str) -> List[Document]:
    """Run metadata-based search and save docs to session. plan may be a dict of filters or an object with .filters."""
    if isinstance(plan, dict):
        filters = plan
    else:
        filters = getattr(plan, "filters", {}) or {}
    q = SearchRequest(
        title=filters.get("title"),
        author=filters.get("authors"),
        # keyword=filters.get("keywords"),
        source=filters.get("sources"),
        min_year=filters.get("year_min"),
        max_year=filters.get("year_max"),
        id_list=filters.get("paper_ids"),
    )
    result = search(q)
    items = result.papers
    docs = _rows_to_documents(items)
    save_session_docs(chat_id, docs)
    return docs


# ==========================================
# 1. Helper: Reciprocal Rank Fusion (RRF)
# ==========================================

def reciprocal_rank_fusion(results_lists: List[List[Document]], k=60) -> List[Document]:
    """
    Combines multiple ranked lists (Vector + Keyword) into one.
    Score = 1 / (k + rank).
    """
    fused_scores = {}
    doc_map = {}

    for distinct_list in results_lists:
        for rank, doc in enumerate(distinct_list):
            # Use a unique identifier. Fallback to title if ID missing.
            # Ideally, your docs have a unique 'id' or 'paper_id' in metadata
            # doc_id = doc.metadata.get("paper_id") or doc.metadata.get("title")     
            doc_id = doc.metadata.get("id") or doc.metadata.get("ID") or doc.metadata.get("title")
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                fused_scores[doc_id] = 0.0
            
            # The RRF formula
            fused_scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by the fused score (Highest score = Best match)
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[did] for did in sorted_ids]

# ==========================================
# 2. Semantic search (vector retrieval, cross-encoder rerank)
# ==========================================


def _run_semantic_search(
    query_text: str, 
    chat_id: str, 
    top_k: int = 100  # Increase this to 100
) -> List[Document]:
    # --- Stage 1: Retrieval ---
    # Fetch more candidates than needed so the reranker has room to work
    vector_result = search(
        SearchRequest(
            search_query=query_text,
            search_mode="vector",
            limit=120,
        )
    )
    vector_docs = _rows_to_documents(vector_result.papers)

    rerank_candidates = vector_docs[:100] # Keep the top 100

    # --- Stage 3: Deep Reranking ---
    pairs = []
    for d in rerank_candidates:
        doc_text = f"{d.metadata.get('title', '')} [SEP] {d.page_content}"
        pairs.append((query_text, doc_text))

    scores = CROSS_ENCODER_MODEL.predict(pairs)
    for d, s in zip(rerank_candidates, scores):
        d.metadata["_rerank_score"] = float(s)
        
    rerank_candidates.sort(key=lambda d: d.metadata["_rerank_score"], reverse=True)
    
    # Store the full ranked candidate list for the frontend papers payload
    from service.session_state import SESSIONS
    if chat_id in SESSIONS:
        sess = SESSIONS[chat_id]
        sess["search_cache"] = rerank_candidates

    return rerank_candidates[:top_k]


def _rerank_docs_by_query(docs: List[Document], query_text: str, top_k: Optional[int] = None) -> List[Document]:
    """
    Re-rank a list of documents by relevance to query_text using the cross-encoder.
    Returns the full list sorted by score (or first top_k if top_k is set).
    """
    if not docs:
        return []
    pairs = [
        (query_text, f"{d.metadata.get('title', '')} [SEP] {d.page_content}")
        for d in docs
    ]
    scores = CROSS_ENCODER_MODEL.predict(pairs)
    for d, s in zip(docs, scores):
        d.metadata["_rerank_score"] = float(s)
    docs_sorted = sorted(docs, key=lambda d: d.metadata["_rerank_score"], reverse=True)
    if top_k is not None:
        return docs_sorted[:top_k]
    return docs_sorted


# TODO(hybrid-refine): currently unused — keep for design reference, do not wire
# back without fixing scores. Production mixed_search uses metadata candidates +
# _rerank_docs_by_query instead.
#
# Idea worth revisiting later (prefer SearchService policy, not this function as-is):
# 1. Run FILTER (or DENSE+filters) and DENSE as two ranked lists.
# 2. Prefer ID intersection as high-confidence hits; if empty, fall back to dense
#    (or to filtered+reranked) rather than pretending they fused.
# 3. Only compute a hybrid score when both sides have real, normalized scores
#    (_semantic_score / _meta_score). Missing scores must not default to 1.0.
# 4. Optional CrossEncoder after fusion; keep retrieval_score / hybrid_score /
#    rerank_score distinct (see notes/rag-hybrid-fallback-followup.md).
# 5. Evaluate vs current mixed_search before replacing it; do not expose alpha /
#    candidate sizes to the client.
def hybrid_refine(meta_docs, sem_docs, query_text, top_k=5, alpha=0.7):
    """Legacy metadata∩semantic refine. Not called by production tools."""

    # --- 1. Build ID sets (use ID only) ---
    meta_ids = {d.metadata.get("id") or d.metadata.get("ID") for d in meta_docs}
    sem_ids  = {d.metadata.get("id") or d.metadata.get("ID") for d in sem_docs}

    intersection_ids = meta_ids & sem_ids
    found_overlap = len(intersection_ids) > 0

    # --- 2. PERFECT MATCHES ---
    if found_overlap:
        final_docs = []
        for d in sem_docs:
            d_id = d.metadata.get("id") or d.metadata.get("ID")
            if d_id in intersection_ids:
                d.metadata["_match_type"] = "perfect_match"
                final_docs.append(d)

    else:
        # --- 3. FALLBACK ---
        final_docs = sem_docs[:top_k]
        for d in final_docs:
            d.metadata["_match_type"] = "topic_fallback"

    # --- 4. Hybrid score ---
    for d in final_docs:
        sem_score = d.metadata.get("_semantic_score", 1.0)
        meta_score = d.metadata.get("_meta_score", 1.0)
        d.metadata["_hybrid_score"] = alpha * sem_score + (1 - alpha) * meta_score

    # --- 5. Rerank ---
    if CROSS_ENCODER_MODEL is not None:
        pairs = [(query_text, d.page_content) for d in final_docs]
        scores = CROSS_ENCODER_MODEL.predict(pairs)
        for d, s in zip(final_docs, scores):
            d.metadata["_rerank_score"] = float(s)
        final_docs.sort(key=lambda x: x.metadata["_rerank_score"], reverse=True)
    else:
        final_docs.sort(key=lambda x: x.metadata["_hybrid_score"], reverse=True)

    # --- 6. Truncate + ensure ID exists ---
    final_docs = final_docs[:top_k]

    for d in final_docs:
        if not d.metadata.get("ID"):
            if d.metadata.get("id"):
                d.metadata["ID"] = d.metadata["id"]

    return final_docs, found_overlap
