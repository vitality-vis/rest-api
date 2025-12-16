import logging
import numpy as np
from copy import deepcopy
from typing import List, Dict, Any, Sequence, Optional
from langchain_core.documents import Document
from service.chroma import ChromaQuerySchema, query_docs
from service.embed_chroma import LocalSpecterEmbedding, LocalSentenceTransformerEmbedding
import chromadb
from sentence_transformers import CrossEncoder

def save_session_docs(chat_id: str, docs: List[Document]) -> None:
    from service.agent_runner import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return

    # --- Append docs to turn buffer ---
    turn_buffer = session.setdefault("_turn_docs", [])
    turn_buffer.extend(deepcopy(docs or []))

    # --- Keep structured memory synced (optional) ---
    mem = session.get("mem")
    if mem:
        mem.set_docs(turn_buffer)


def get_session_docs(chat_id: str) -> List[Document]:
    """Return current docs for this chat (fallback to memory cache if empty)."""
    from service.agent_runner import SESSIONS
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
    from service.agent_runner import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return

    session["docs"] = []
    logging.info(f"[rag_core] Cleared docs for chat_id={chat_id}")

    # ✅ Also clear memory doc cache if present
    mem = session.get("mem")
    if mem and hasattr(mem, "clear_docs"):
        try:
            mem.clear_docs()
            logging.info(f"[rag_core] Cleared structured memory docs for chat_id={chat_id}")
        except Exception as e:
            logging.error(f"[rag_core] MemoryManager clear failed: {e}")

# =====================================================
# 🧠 Embedding + Chroma setup
# =====================================================
CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_MAPPING = {
    "specter": "paper_specter",
    "ada": "paper_ada_localized",
    "glove": "paper_glove_localized",
}

EMBEDDING_MODELS = {
    "specter": LocalSpecterEmbedding(model_name="allenai/specter"),
    "ada": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "glove": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
}

# def format_docs(docs: Sequence[Document]) -> str:
#     """Format retrieved docs into Markdown for LLM context (including abstract)."""
#     if not docs:
#         return "_(No documents found.)_"

#     blocks = []
#     for i, doc in enumerate(docs):
#         # Support both dict and Document, then normalize keys to lower-case
#         if isinstance(doc, dict):
#             md = doc.get("metadata", doc)
#         else:
#             md = getattr(doc, "metadata", {}) or {}

#         md_l = {str(k).lower(): v for k, v in md.items()}

#         doc_id = str(md_l.get("id", f"doc_{i}"))

#         authors_raw = md_l.get("authors", "")
#         if isinstance(authors_raw, list):
#             authors = ", ".join(authors_raw)
#         else:
#             authors = authors_raw

#         abstract = md_l.get("abstract", "") or ""
#         if isinstance(abstract, str) and len(abstract) > 500:
#             abstract = abstract[:500].rstrip() + "..."  # avoid flooding context

#         blocks.append(
#             f"- **Title:** {md_l.get('title', '')} [[ID:{doc_id}]]\n"
#             f"  - Authors: {authors}\n"
#             f"  - Year: {md_l.get('year', '')}\n"
#             f"  - Source: {md_l.get('source', '')}\n"
#             f"  - Score: {float(md_l.get('_score', 0.0)):.4f}\n"
#             f"  - Abstract: {abstract}\n"
#         )

#     return "\n".join(blocks)

def format_docs(docs: Sequence[Document]) -> str:
    """
    Format retrieved or recalled docs into Markdown for LLM context (memory-aware).

    - Handles both `Document` and raw `dict` entries.
    - Truncates long abstracts to stay token-safe.
    - Optionally includes stable IDs for follow-ups.
    """
    if not docs:
        return "_(No documents found or remembered.)_"

    formatted_blocks = []
    for i, doc in enumerate(docs):
        # 🔹 Support both dict and LangChain Document
        if isinstance(doc, dict):
            md = doc.get("metadata", doc)
        else:
            md = getattr(doc, "metadata", {}) or {}

        # 🔹 Normalize key casing
        md_l = {str(k).lower(): v for k, v in md.items()}

        # 🔹 Stable ID for later recall
        doc_id = str(md_l.get("id", f"doc_{i}"))

        # 🔹 Authors normalization
        authors_raw = md_l.get("authors", "")
        authors = ", ".join(authors_raw) if isinstance(authors_raw, list) else authors_raw

        # # 🔹 Abstract trimming
        # abstract = md_l.get("abstract", "") or ""
        # if isinstance(abstract, str) and len(abstract) > 400:
        #     abstract = abstract[:400].rstrip() + "..."  # keep concise for token efficiency

        # 🔹 Abstract (full, intact)
        abstract = md_l.get("abstract", "") or ""

        # 🔹 Score / ranking
        score_val = md_l.get("_score", md_l.get("score", 0.0))
        try:
            score_val = float(score_val)
        except Exception:
            score_val = 0.0

        # 🔹 Format block
        formatted_blocks.append(
            f"- **Title:** {md_l.get('title', '(No title)')} [[ID:{doc_id}]]\n"
            f"  - Authors: {authors or '(Unknown)'}\n"
            f"  - Year: {md_l.get('year', '(N/A)')}\n"
            f"  - Source: {md_l.get('source', '(N/A)')}\n"
            f"  - Score: {score_val:.4f}\n"
            f"  - Abstract: {abstract}\n"
        )

    # ✅ Optionally store formatted text in memory for the generator
    formatted_text = "\n".join(formatted_blocks)
    return formatted_text


def _rows_to_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw Chroma rows into LangChain Documents."""
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
# 🔍 Query Functions
# =====================================================

def _query_chroma_by_embedding(query_text: str, embedding_type: str = "specter", k: int = 5) -> List[Document]:
    """Embed query and perform Chroma vector search."""
    embedding_type = embedding_type.lower()
    if embedding_type not in EMBEDDING_MODELS:
        embedding_type = "specter"

    embedder = EMBEDDING_MODELS[embedding_type]
    collection_name = CHROMA_COLLECTION_MAPPING[embedding_type]
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        logging.error(f"[retriever] Failed to get collection {collection_name}: {e}")
        return []

    qvec = embedder.embed_query(query_text)
    results = collection.query(
        query_embeddings=[qvec],
        n_results=int(k),
        include=["metadatas", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    docs = []
    for i, (m, d) in enumerate(zip(metadatas, distances)):
        score = 1 / (1 + float(d or 0))
        docs.append(
            Document(
                page_content=m.get("Abstract", "") or m.get("Title", ""),
                metadata={
                    "title": m.get("Title", ""),
                    "authors": m.get("Authors", []),
                    "source": m.get("Source", ""),
                    "year": m.get("Year", ""),
                    "abstract": m.get("Abstract", m.get("abstract", "")),
                    "id": str(m.get("ID", f"doc_{i}")),
                    "_score": score,
                },
            )
        )
    return docs


def _run_metadata_search(plan, chat_id: str) -> List[Document]:
    """Run metadata-based search and save docs to session."""
    filters = getattr(plan, "filters", {}) or {}
    q = ChromaQuerySchema(
        title=filters.get("title"),
        author=filters.get("authors"),
        # keyword=filters.get("keywords"),
        source=filters.get("sources"),
        min_year=filters.get("year_min"),
        max_year=filters.get("year_max"),
        id_list=filters.get("paper_ids"),
    )
    items = query_docs(q)
    docs = _rows_to_documents(items)
    save_session_docs(chat_id, docs)
    return docs

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi

# ==========================================
# 1. Global State & Initialization
# ==========================================

# These hold the keyword index in memory
BM25_INDEX = None
BM25_DOC_MAP = {} 

def initialize_bm25_index(all_documents: List[Document]):
    """
    Call this ONCE when your app starts. 
    It pulls docs from Chroma (or your cache) to build the keyword index.
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
    # logging.info("✅ BM25 Index built successfully.")

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
    logging.info("✅ BM25 Index built successfully (title + keywords + abstract).")

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
            # Ideally, your docs in Chroma should have a unique 'id' or 'paper_id' in metadata
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
# 3. Main Search Function
# ==========================================

def _run_semantic_search(
    query_text: str, 
    chat_id: str, 
    top_k: int = 5
) -> List[Document]:
    """
    Robust Hybrid Search Pipeline:
    1. Parallel Recall: Specter (Vector) + BM25 (Keyword)
    2. Fusion: Reciprocal Rank Fusion
    3. Precision: Cross-Encoder Reranking
    """
    
    # --- Stage 1: Parallel Retrieval (Recall) ---
    
    # A. Vector Search (Semantic Meaning)
    # Note: We request 50 candidates here
    vector_docs = _query_chroma_by_embedding(query_text, embedding_type="specter", k=50)
    
    # B. Keyword Search (Exact Match)
    keyword_docs = []
    if BM25_INDEX is not None:
        tokenized_query = query_text.lower().split()
        # Retrieve top 50 strictly by keyword overlap
        # .get_top_n returns the actual items from our corpus map
        raw_bm25_docs = BM25_INDEX.get_top_n(tokenized_query, list(BM25_DOC_MAP.values()), n=50)
        keyword_docs = raw_bm25_docs
    else:
        logging.warning("BM25 Index not initialized. Running pure Vector search.")

    # --- Stage 2: Fusion (RRF) ---
    
    if not vector_docs and not keyword_docs:
        return []

    # Combine the lists. Docs found by BOTH methods get pushed to the top.
    fused_docs = reciprocal_rank_fusion([vector_docs, keyword_docs])
    
    # Take top 50 fused results for the expensive reranking step
    rerank_candidates = fused_docs[:50]

    # --- Stage 3: Deep Reranking (Precision) ---
    
    # Prepare pairs for the Cross Encoder
    # Format: (Query, Title + [SEP] + Abstract) for maximum context
    pairs = []
    for d in rerank_candidates:
        abstract = d.page_content or d.metadata.get("abstract", "")
        title = d.metadata.get("title", "")
        # Combine title and abstract for the model
        doc_text = f"{title} [SEP] {abstract}"
        pairs.append((query_text, doc_text))

    # Predict scores
    scores = CROSS_ENCODER_MODEL.predict(pairs)
    
    # Attach scores and sort
    for d, s in zip(rerank_candidates, scores):
        d.metadata["_rerank_score"] = float(s)
        
    rerank_candidates.sort(key=lambda d: d.metadata["_rerank_score"], reverse=True)
    
    # Select final Top K
    final_docs = rerank_candidates[:top_k]

    # Save and Return
    save_session_docs(chat_id, final_docs)
    logging.info(f"✅ Retrieved {len(final_docs)} docs for chat {chat_id}")
    
    return final_docs



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


# =====================================================
# 🧩 Session-based Cache Helpers
# =====================================================

def get_session_docs(chat_id: str) -> List[Document]:
    """Return current docs for this chat."""
    from service.agent_runner import SESSIONS
    session = SESSIONS.get(chat_id)
    return session.get("docs", []) if session else []

def clear_session_docs(chat_id: str) -> None:
    """Clear doc cache for one chat."""
    from service.agent_runner import SESSIONS
    session = SESSIONS.get(chat_id)
    if session:
        session["docs"] = []
        logging.info(f"[rag_core] Cleared docs for chat_id={chat_id}")