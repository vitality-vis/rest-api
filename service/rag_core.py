"""
RAG core: session doc storage, embedding/rerank setup, semantic search, hybrid refine.
"""
import logging
import numpy as np
from copy import deepcopy
from typing import List, Dict, Any, Sequence, Optional
from langchain_core.documents import Document
from model.query import QuerySchema
from service.zilliz import query_docs
from service.embed import LocalSpecterEmbedding, LocalSentenceTransformerEmbedding
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

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
COLLECTION_MAPPING = {
    "specter": "paper_specter",
    "ada": "paper_ada_localized",
    "glove": "paper_glove_localized",
}

EMBEDDING_MODELS = {
    "specter": LocalSpecterEmbedding(model_name="allenai/specter"),
    "ada": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "glove": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
}

def format_docs(docs: Sequence[Document], *, include_abstract: bool = True, include_score: bool = True) -> str:
    """
    Format retrieved or recalled docs into Markdown for LLM context (memory-aware).

    - Handles both `Document` and raw `dict` entries.
    - Truncates long abstracts to stay token-safe when include_abstract=True.
    - Optionally includes stable IDs for follow-ups.
    - Set include_abstract=False, include_score=False for compact lists (e.g. load more papers).
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

        # Abstract (omit for compact "load more" lists)
        abstract = (md_l.get("abstract", "") or "") if include_abstract else ""

        # Score / ranking
        score_val = md_l.get("_score", md_l.get("score", 0.0))
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
                    "_score": float(m.get("score", 0.0)),
                },
            )
        )
    return docs

# =====================================================
# Query Functions
# =====================================================

def _query_zilliz_by_embedding(query_text: str, embedding_type: str = "specter", k: int = 5) -> List[Document]:
    """Embed query and perform Zilliz vector search."""
    from service import zilliz
    embedding_type = embedding_type.lower()
    if embedding_type not in EMBEDDING_MODELS:
        embedding_type = "specter"

    embedder = EMBEDDING_MODELS[embedding_type]
    qvec = embedder.embed_query(query_text)
    raw = zilliz.query_doc_by_embedding(
        paper_ids=None,
        embedding=qvec,
        embedding_type=embedding_type,
        limit=int(k),
    )
    docs = []
    for i, m in enumerate(raw):
        authors = m.get("Authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        docs.append(
            Document(
                page_content=m.get("Abstract", "") or m.get("Title", ""),
                metadata={
                    "title": m.get("Title", ""),
                    "authors": authors,
                    "source": m.get("Source", ""),
                    "year": m.get("Year", ""),
                    "abstract": m.get("Abstract", m.get("abstract", "")),
                    "id": str(m.get("ID", f"doc_{i}")),
                    "_score": float(m.get("score", 0)),
                },
            )
        )
    return docs


def _run_metadata_search(plan, chat_id: str) -> List[Document]:
    """Run metadata-based search and save docs to session. plan may be a dict of filters or an object with .filters."""
    if isinstance(plan, dict):
        filters = plan
    else:
        filters = getattr(plan, "filters", {}) or {}
    q = QuerySchema(
        title=filters.get("title"),
        author=filters.get("authors"),
        # keyword=filters.get("keywords"),
        source=filters.get("sources"),
        min_year=filters.get("year_min"),
        max_year=filters.get("year_max"),
        id_list=filters.get("paper_ids"),
    )
    result = query_docs(q)
    items = result.get("papers", [])
    docs = _rows_to_documents(items)
    save_session_docs(chat_id, docs)
    return docs


# ==========================================
# 1. Global State & Initialization
# ==========================================

# These hold the keyword index in memory
BM25_INDEX = None
BM25_DOC_MAP = {} 

def initialize_bm25_index(all_documents: List[Document]):
    """
    Call this ONCE when your app starts. 
    It pulls docs from Zilliz (or your cache) to build the keyword index.
    """
    global BM25_INDEX, BM25_DOC_MAP
    
    logging.info(f"Building BM25 index for {len(all_documents)} documents...")
    
    corpus_tokens = []
    BM25_DOC_MAP = {}

    # for idx, doc in enumerate(all_documents):
    #     # Combine Title + Abstract for rich keyword matching
    #     # We map the integer index 'idx' to the actual Document object
    #     BM25_DOC_MAP[idx] = doc
        
    #     content_text = (doc.metadata.get("title", "") + " " + doc.page_content).lower()
    #     # Simple tokenization (split by whitespace). 
    #     # For production, consider spaCy or NLTK for better stemming.
    #     tokens = content_text.split() 
    #     corpus_tokens.append(tokens)

    # BM25_INDEX = BM25Okapi(corpus_tokens)
    # logging.info(" BM25 Index built successfully.")

    corpus_tokens = []
    BM25_DOC_MAP = {}

    for idx, doc in enumerate(all_documents):

        BM25_DOC_MAP[idx] = doc

        # --- Extract metadata fields safely ---
        title_text = doc.metadata.get("title", "")
        abstract_text = doc.page_content or doc.metadata.get("abstract", "")
        keywords_text = doc.metadata.get("keywords", "")

        # --- Build the BM25-optimized text field ---
        # Important: BM25 is sensitive to duplicate words and noise.
        # We keep formatting extremely simple: "title keywords abstract"
        content_text = f"{title_text} {keywords_text} {abstract_text}".lower()

        # Basic tokenization. (You can later switch to spaCy.)
        tokens = content_text.split()

        corpus_tokens.append(tokens)

    BM25_INDEX = BM25Okapi(corpus_tokens)
    logging.info(" BM25 Index built successfully (title + keywords + abstract).")

# ==========================================
# 2. Helper: Reciprocal Rank Fusion (RRF)
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
# 3. Semantic search (vector + BM25 fusion, cross-encoder rerank)
# ==========================================


def _run_semantic_search(
    query_text: str, 
    chat_id: str, 
    top_k: int = 100  # Increase this to 100
) -> List[Document]:
    # --- Stage 1: Parallel Retrieval ---
    # Fetch more candidates to ensure quality after fusion
    vector_docs = _query_zilliz_by_embedding(query_text, embedding_type="specter", k=120)
    
    keyword_docs = []
    if BM25_INDEX is not None:
        tokenized_query = query_text.lower().split()
        keyword_docs = BM25_INDEX.get_top_n(tokenized_query, list(BM25_DOC_MAP.values()), n=120)

    # --- Stage 2: Fusion ---
    fused_docs = reciprocal_rank_fusion([vector_docs, keyword_docs])
    rerank_candidates = fused_docs[:100] # Keep the top 100

    # --- Stage 3: Deep Reranking ---
    pairs = []
    for d in rerank_candidates:
        doc_text = f"{d.metadata.get('title', '')} [SEP] {d.page_content}"
        pairs.append((query_text, doc_text))

    scores = CROSS_ENCODER_MODEL.predict(pairs)
    for d, s in zip(rerank_candidates, scores):
        d.metadata["_rerank_score"] = float(s)
        
    rerank_candidates.sort(key=lambda d: d.metadata["_rerank_score"], reverse=True)
    
    # Store the full 100 in the session cache for pagination / load-more
    from service.session_state import SESSIONS
    if chat_id in SESSIONS:
        sess = SESSIONS[chat_id]
        # Backend pagination cache used by load_more_papers in agent_tools.py
        sess["search_cache"] = rerank_candidates
        # First 5 papers are already shown in the initial semantic_search response
        sess["last_offset"] = 5

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


def hybrid_refine(meta_docs, sem_docs, query_text, top_k=5, alpha=0.7):

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
