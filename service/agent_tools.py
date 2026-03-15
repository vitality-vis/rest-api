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

from service import rag_core
from service.rag_core import (
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
        # Use high-level query_docs API over cached papers (fast, in-memory)
        from model.query import QuerySchema
        from service.zilliz import query_docs
        from service.session_state import SESSIONS
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
    # Step 3 — Build QuerySchema for fast in-memory filtering
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

    q = QuerySchema(
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

    logging.info(f"[metadata_search] Running query_docs with filters={q.dict(exclude_none=True)}")

    # ============================================================
    # Step 4 — Execute query against cached papers
    # ============================================================
    result = query_docs(q)
    items = result.get("papers", []) or []

    # ============================================================
    # Step 5 — Save to session + format output (reuse rag_core format)
    # ============================================================
    # Convert rows to LangChain Documents so downstream code (memory, etc.) keeps working.
    from service.rag_core import _rows_to_documents, save_session_docs, format_docs

    docs = _rows_to_documents(items)
    save_session_docs(chat_id, docs)

    if not docs:
        return "_(No matching papers found.)_"

    # Load-more: store full result set in session (same as semantic_search)
    from service.session_state import get_session, save_session
    sess = get_session(chat_id) or {}
    sess["search_cache"] = docs
    sess["last_offset"] = 5
    save_session(chat_id, sess)

    # Show first 5 with abstracts so the model can answer questions about the papers
    initial_docs = docs[:5]
    formatted_text = format_docs(initial_docs, include_abstract=True, include_score=False)
    if len(docs) > 5:
        formatted_text += "\n\n[SIGNAL:SHOW_LOAD_MORE]"
    return formatted_text


# @tool("semantic_search", return_direct=True)
# def semantic_search(query: str, chat_id: str = "default") -> str:
@tool("semantic_search", return_direct=False)
def semantic_search(query: str, chat_id: str = "default") -> str:
    """
    [PAPER SEARCH — list only] Retrieve a list of papers by topic/semantic similarity. Use ONLY when the user wants to see/find papers (e.g. "give me papers on X"). Returns top 5 and caches the rest for Load More. For answering a question using papers, use rag_semantic_qa instead.
    """
    # 1. Fetch 100 results from Zilliz + Reranker
    # We pass top_k=100 so the core function knows to fetch a large batch
    docs = rag_core._run_semantic_search(query_text=query, chat_id=chat_id, top_k=100)
    
    if not docs:
        return "_(No relevant papers found.)_"

    # 2. Slice the first 5 for the LLM/User to see immediately
    initial_docs = docs[:5]
    formatted_text = rag_core.format_docs(initial_docs)

    # 3. Add the Frontend Signal
    # Your frontend code will look for this exact string to render the button
    if len(docs) > 5:
        formatted_text += "\n\n[SIGNAL:SHOW_LOAD_MORE]"
    
    return formatted_text



@tool("mixed_search", return_direct=False)
def mixed_search(query_text, filters, chat_id, top_k=5):
    """
    [PAPER SEARCH — list only] First run metadata search (filters), then re-rank those results by topic/semantic relevance to query_text. Use when the user wants papers that match both metadata (author, year, venue, etc.) and topic — or when they ask a question that combines topic + filters (e.g. "What do CHI papers say about usability?"); call mixed_search then answer from the results.
    """
    from service.session_state import SESSIONS
    from service.rag_core import (
        _run_metadata_search,
        _rerank_docs_by_query,
        save_session_docs,
        format_docs,
    )

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
    # Store full reranked list for load_more_papers
    if chat_id in SESSIONS:
        sess = SESSIONS[chat_id]
        sess["search_cache"] = reranked
        sess["last_offset"] = top_k
    save_session_docs(chat_id, reranked[:top_k])

    logging.info(
        f"[mixed_search] Metadata returned {len(meta_docs)} papers; "
        f"returning top {top_k} after semantic re-rank for chat_id={chat_id}"
    )
    return format_docs(reranked[:top_k])


@tool("load_more_papers", return_direct=True)
def load_more_papers(chat_id: str = "default") -> str:
    """
    Fetch the next 10 papers from the session cache.
    """
    from service.session_state import SESSIONS
    session = SESSIONS.get(chat_id)
    if not session:
        return "No more papers found in this search."
    all_docs = session.get("search_cache", [])
    current_offset = session.get("last_offset", 5) 
    
    next_batch = all_docs[current_offset : current_offset + 10]
    
    if not next_batch:
        return "No more papers found in this search."

    # Update the offset for the NEXT "Load More" click
    session["last_offset"] = current_offset + len(next_batch)
    
    formatted = rag_core.format_docs(next_batch, include_abstract=False, include_score=False)
    # Short intro; no need to invoke the model for a simple list
    result = "Here are more papers from this search:\n\n" + formatted
    # If there are still more left in the 100, keep the signal alive
    if session["last_offset"] < len(all_docs):
        result += "\n\n[SIGNAL:SHOW_LOAD_MORE]"
        
    return result


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
    load_more_papers,
    rag_semantic_qa,
]