"""
Agent tools for the research assistant: metadata/semantic/mixed search, RAG Q&A, load more.
"""
import json
import re
import logging
from typing import Union, List

from langchain_core.tools import tool
from langchain.schema import Document
from pydantic import BaseModel, Field

from agents.agent_v1_legacy import rag_core
from agents.agent_v1_legacy.rag_core import (
    _run_metadata_search,
    _run_semantic_search,
    format_docs,
    save_session_docs,
    get_session_docs,
)


class MetadataSearchInput(BaseModel):
    filters: dict = Field(..., description="Metadata filters dictionary")
    user_request: str = ""
    chat_id: str = "default"


@tool(
    "metadata_search",
    args_schema=MetadataSearchInput,
    return_direct=False
)
def metadata_search(filters: Union[str, dict], user_request: str = "", chat_id: str = "default") -> str:
    """
    [PAPER SEARCH — list only] Search papers by metadata filters only: author, year, venue/source, title, paper IDs. Do NOT accept topic or keywords — use semantic_search or mixed_search for topic/keyword. Authors: AND logic. Source: AND logic. Title: scalar only.
    """

    # ============================================================
    # Step 0 — Lazy imports
    # ============================================================
    try:
        from model.paper import SearchRequest
        from service.search import search
    except Exception as e:
        return f"Failed to load metadata search dependencies: {e}"

    # ============================================================
    # Step 1 — Parse filters
    # ============================================================
    try:
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except json.JSONDecodeError:
                logging.info(f"[metadata_search] key:value fallback parse: {filters}")
                pattern = r'(\w+)\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([\w\s\-\_]+))'
                matches = re.findall(pattern, filters)
                filters = {
                    k.lower(): next((v for v in vals if v), "").strip()
                    for k, *vals in matches
                }

        if not isinstance(filters, dict):
            return "Invalid filters format (expect dict or JSON string)"

    except Exception as e:
        logging.error(f"[metadata_search] Parse error: {e}")
        return f"Invalid filter format: {e}"

    # ============================================================
    # Step 2 — Normalize incoming filters
    # ============================================================
    filters_norm = {}

    # Only allow structured metadata: title, authors, sources/venues, year_min, year_max, ids
    allowed_keys = {"title", "authors", "sources", "source", "venues", "venue", "year_min", "year_max", "ids", "id_list", "paper_ids"}
    for k, v in filters.items():
        key = k.lower().strip()
        if v is None or key not in allowed_keys:
            continue
        if key in ("topic", "keywords", "keyword"):
            continue  # metadata_search does not use topic or keywords

        # TITLE: enforce scalar
        if key == "title":
            if isinstance(v, list) and v:
                v = v[0]       # take first item
            filters_norm["title"] = str(v).lower().strip()
            continue

        if isinstance(v, str) and not v.strip():
            continue

        if isinstance(v, list):
            if not v:
                continue
            filters_norm[key] = [str(i).lower().strip() for i in v]
        else:
            filters_norm[key] = str(v).lower().strip()

    # Parse year filters
    for num_key in ["year_min", "year_max"]:
        if num_key in filters_norm:
            try:
                filters_norm[num_key] = int(re.findall(r"\d{4}", str(filters_norm[num_key]))[0])
            except Exception:
                filters_norm.pop(num_key, None)

    # ============================================================
    # Step 3 — Build a SearchRequest for metadata filtering
    # ============================================================
    # Authors: always a list
    authors_val = filters_norm.get("authors")
    if authors_val:
        if isinstance(authors_val, list):
            authors = authors_val
        else:
            authors = [str(authors_val)]
    else:
        authors = None

    # Source / sources: normalize to list
    source_val = filters_norm.get("sources") or filters_norm.get("source")
    if source_val:
        if isinstance(source_val, list):
            sources = source_val
        else:
            sources = [str(source_val)]
    else:
        sources = None

    # metadata_search does not use topic or keywords — use semantic_search or mixed_search for those
    keywords = None

    # IDs: support ids / id_list / paper_ids from filters
    ids_val = (
        filters.get("ids")
        or filters.get("id_list")
        or filters.get("paper_ids")
        or filters_norm.get("ids")
        or filters_norm.get("id_list")
        or filters_norm.get("paper_ids")
    )
    if ids_val:
        if isinstance(ids_val, list):
            id_list = [str(i) for i in ids_val]
        else:
            id_list = [str(ids_val)]
    else:
        id_list = None

    q = SearchRequest(
        title=filters_norm.get("title"),
        author=authors,
        source=sources,
        keyword=keywords,
        min_year=filters_norm.get("year_min"),
        max_year=filters_norm.get("year_max"),
        id_list=id_list,
        limit=100,
        offset=0,
    )

    logging.info(
        "[metadata_search] Running SearchService with filters=%s",
        q.model_dump(exclude_none=True) if hasattr(q, "model_dump") else q,
    )

    # ============================================================
    # Step 4 — Execute the shared metadata search
    # ============================================================
    result = search(q)
    items = result.papers

    # ============================================================
    # Step 5 — Save to session + format output (reuse rag_core format)
    # ============================================================
    # Convert rows to LangChain Documents so downstream code (memory, etc.) keeps working.
    from agents.agent_v1_legacy.rag_core import _rows_to_documents, save_session_docs, format_docs

    docs = _rows_to_documents(items)
    save_session_docs(chat_id, docs)

    if not docs:
        return "_(No matching papers found.)_"

    # Full ranked set for the frontend paper list (streamed as a hidden payload).
    from .session_state import get_session, save_session
    from agents.agent_v1_legacy.rag_core import CHAT_PREVIEW_LIMIT

    sess = get_session(chat_id) or {}
    sess["search_cache"] = docs
    save_session(chat_id, sess)

    # Chat/LLM preview only — full list goes to the frontend via papers payload.
    initial_docs = docs[:CHAT_PREVIEW_LIMIT]
    return format_docs(initial_docs, include_abstract=True, include_score=False)


# @tool("semantic_search", return_direct=True)
# def semantic_search(query: str, chat_id: str = "default") -> str:
@tool("semantic_search", return_direct=False)
def semantic_search(query: str, chat_id: str = "default") -> str:
    """
    [PAPER SEARCH — list only] Retrieve papers by topic/semantic similarity. Use ONLY when the user wants to see/find papers (e.g. "give me papers on X"). Returns a short chat preview and caches the ranked list for the frontend paper panel. For answering a question using papers, use rag_semantic_qa instead.
    """
    from agents.agent_v1_legacy.rag_core import CHAT_PREVIEW_LIMIT, EMIT_PAPER_LIMIT

    # Fetch a large candidate pool for rerank; emit cap is separate (see EMIT_PAPER_LIMIT).
    docs = rag_core._run_semantic_search(
        query_text=query, chat_id=chat_id, top_k=EMIT_PAPER_LIMIT
    )

    if not docs:
        return "_(No relevant papers found.)_"

    initial_docs = docs[:CHAT_PREVIEW_LIMIT]
    return rag_core.format_docs(initial_docs)



@tool("mixed_search", return_direct=False)
def mixed_search(query_text, filters, chat_id, top_k=None):
    """
    [PAPER SEARCH — list only] First run metadata search (filters), then re-rank those results by topic/semantic relevance to query_text. Use when the user wants papers that match both metadata (author, year, venue, etc.) and topic — or when they ask a question that combines topic + filters (e.g. "What do CHI papers say about usability?"); call mixed_search then answer from the results.
    """
    from .session_state import SESSIONS
    from agents.agent_v1_legacy.rag_core import (
        CHAT_PREVIEW_LIMIT,
        _run_metadata_search,
        _rerank_docs_by_query,
        save_session_docs,
        format_docs,
    )

    preview_k = CHAT_PREVIEW_LIMIT if top_k is None else top_k

    # 1. Run metadata search first
    meta_docs = _run_metadata_search(filters, chat_id)

    if not meta_docs:
        logging.info("[mixed_search] No papers matched metadata filters.")
        author = filters.get("authors", ["the specified author"])
        author = author[0] if isinstance(author, list) and author else author
        year_min = filters.get("year_min", "N/A")
        year_max = filters.get("year_max", "N/A")
        topic = query_text
        return (
            f"SYSTEM_NOTICE: No papers were found matching the filters.\n"
            f"- Author filter: {author}\n"
            f"- Year range: {year_min}–{year_max}\n"
            f"- Query topic: {topic}\n"
            "Please respond politely, explain that no results were found, "
            "and offer to broaden the search (e.g., drop author or topic filter)."
        )

    # 2. Re-rank metadata results by semantic relevance to query_text
    reranked = _rerank_docs_by_query(meta_docs, query_text, top_k=None)
    # Store full reranked list for the frontend papers payload
    if chat_id in SESSIONS:
        sess = SESSIONS[chat_id]
        sess["search_cache"] = reranked
    save_session_docs(chat_id, reranked[:preview_k])

    logging.info(
        f"[mixed_search] Metadata returned {len(meta_docs)} papers; "
        f"returning chat preview top {preview_k} after semantic re-rank for chat_id={chat_id}"
    )
    return format_docs(reranked[:preview_k])


# =====================================================
# RAG Q&A tools (retrieve then answer — use when user asks a question)
# =====================================================

_RAG_QA_HEADER = (
    "Use the following retrieved papers to answer the question. "
    "Provide a direct answer; do not just list paper titles.\n\n"
)

@tool("rag_semantic_qa", return_direct=False)
def rag_semantic_qa(query: str, question: str, chat_id: str = "default") -> str:
    """
    [RAG Q&A] Retrieve papers by topic/semantic similarity, then answer the given question using those papers. Use when the user asks a question that should be answered from retrieved content (e.g. 'What methods do RAG papers use?'). Do not use for simply listing papers — use semantic_search for that.
    """
    docs = rag_core._run_semantic_search(query_text=query, chat_id=chat_id, top_k=5)
    if not docs:
        return "_(No relevant papers found.)_ Use this to politely say no papers were retrieved and suggest broadening the query."
    formatted = rag_core.format_docs(docs[:5], include_abstract=True, include_score=False)
    return _RAG_QA_HEADER + "Papers:\n" + formatted + "\n\nQuestion: " + question


# =====================================================
# Tool Registration
# =====================================================
ALL_AGENT_TOOLS = [
    metadata_search,
    semantic_search,
    mixed_search,
    rag_semantic_qa,
]
