import json
import re
import logging
from dataclasses import dataclass
from typing import Union

from langchain_core.tools import tool
from langchain.schema import Document
from service import rag_core
from typing import Union
from service.rag_core import (
    _run_metadata_search,
    _run_semantic_search,
    hybrid_refine,
    format_docs,
    save_session_docs,
    get_session_docs,
)

from pydantic.v1 import BaseModel, Field

# def save_session_docs(chat_id, docs):
#     """Save retrieved docs to the right session (and sync structured memory)."""
#     from service.agent_runner import SESSIONS  # lazy import to avoid circular dependency
#     if chat_id not in SESSIONS:
#         return

#     # Update raw session cache
#     SESSIONS[chat_id]["docs"] = docs

#     # ✅ Also update structured memory if available
#     session = SESSIONS[chat_id]
#     if "mem" in session and hasattr(session["mem"], "set_docs"):
#         session["mem"].set_docs(docs)


# @tool("metadata_search", return_direct=True)
# def metadata_search(filters: Union[str, dict], user_request: str = "", chat_id: str = "default") -> str:
#     """
#     🔎 Search academic papers using structured metadata filters in ChromaDB.

#     Supports both JSON strings and 'key:value' input formats.
#     Each chat tab keeps its own cached results in SESSIONS[chat_id]['docs'].

#     Supported filter keys:
#       - title
#       - authors
#       - year_min, year_max
#       - source / venue
#     """
#     # ============================================================
#     # 🧠 Step 0 — Lazy imports to avoid circular dependencies
#     # ============================================================
#     try:
#         from service.chroma import _chroma_client, COLLECTION_MAPPING
#     except Exception as e:
#         return f"❌ Failed to load Chroma client: {e}"

#     try:
#         from service.agent_runner import SESSIONS
#     except Exception:
#         SESSIONS = {}

#     # ============================================================
#     # 🧩 Step 1 — Parse input flexibly
#     # ============================================================
#     try:
#         if isinstance(filters, str):
#             try:
#                 filters = json.loads(filters)  # Try strict JSON first
#             except json.JSONDecodeError:
#                 logging.info(f"[metadata_search] Non-JSON filters → using key:value parse: {filters}")
#                 # Match patterns like title:"Data Mining" authors:Jiawei Han
#                 pattern = r'(\w+)\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([\w\s\-\_]+))'
#                 matches = re.findall(pattern, filters)
#                 filters = {
#                     k.lower(): next((v for v in vals if v), "").strip()
#                     for k, *vals in matches
#                 }

#         elif not isinstance(filters, dict):
#             return "❌ Invalid input: filters must be a JSON string or dict."
#     except Exception as e:
#         logging.error(f"[metadata_search] Parse error: {e}")
#         return f"❌ Invalid filter format: {e}"

#     # ============================================================
#     # 🧹 Step 2 — Normalize filters
#     # ============================================================
#     filters_norm = {}
#     for k, v in filters.items():
#         key = k.lower().strip()
#         if v in [None, ""]:
#             continue
#         # Handle list vs. string
#         if isinstance(v, list):
#             filters_norm[key] = [str(i).lower().strip() for i in v]
#         else:
#             filters_norm[key] = str(v).lower().strip()

#     # Parse year limits safely
#     for num_key in ["year_min", "year_max"]:
#         if num_key in filters_norm:
#             try:
#                 filters_norm[num_key] = int(re.findall(r"\d{4}", str(filters_norm[num_key]))[0])
#             except Exception:
#                 filters_norm.pop(num_key, None)

#     # ============================================================
#     # 🧩 Step 3 — Get collection dynamically (no hardcoding)
#     # ============================================================
#     if _chroma_client is None:
#         return "❌ Chroma client not initialized."

#     # Pick default collection (Specter is usually the semantic base)
#     collection_name = COLLECTION_MAPPING.get("specter", "paper_specter")
#     chroma_collection = _chroma_client.get_or_create_collection(collection_name)

#     # ============================================================
#     # 🧠 Step 4 — Fetch and filter metadata
#     # ============================================================
#     all_docs = chroma_collection.get(include=["metadatas", "documents"])
#     metadatas = all_docs.get("metadatas", [])
#     logging.info(f"[metadata_search] Checking {len(metadatas)} metadata entries...")

#     results = []
#     for meta in metadatas:
#         m = {k.lower(): str(v).lower() for k, v in meta.items()}
#         year = 0
#         try:
#             year = int(re.findall(r"\d{4}", m.get("year", ""))[0])
#         except Exception:
#             pass

#         # --- Title ---
#         if "title" in filters_norm:
#             title_val = filters_norm["title"]
#             if isinstance(title_val, list):
#                 if not any(t in m.get("title", "") for t in title_val):
#                     continue
#             else:
#                 if title_val not in m.get("title", ""):
#                     continue

#         if "authors" in filters_norm:
#             author_val = filters_norm["authors"]
#             author_field = m.get("authors", "")
#             if isinstance(author_val, list):
#                 if not any(a in author_field for a in author_val):
#                     continue
#             else:
#                 if author_val not in author_field:
#                     continue

#         # --- Year range ---
#         if "year_min" in filters_norm and year < filters_norm["year_min"]:
#             continue
#         if "year_max" in filters_norm and year > filters_norm["year_max"]:
#             continue

#         # --- Source/Venue ---
#         if "source" in filters_norm and filters_norm["source"] not in m.get("source", ""):
#             continue
#         if "venue" in filters_norm and filters_norm["venue"] not in m.get("source", ""):
#             continue

#         results.append(meta)

#     def make_doc(meta: dict) -> Document:
#         meta_norm = {k.lower(): v for k, v in meta.items()}  # 👈 normalize keys
#         parts = []
#         if meta_norm.get("title"):
#             parts.append(f"Title: {meta_norm['title']}")
#         if meta_norm.get("authors"):
#             parts.append(f"Authors: {meta_norm['authors']}")
#         if meta_norm.get("year"):
#             parts.append(f"Year: {meta_norm['year']}")
#         if meta_norm.get("source"):
#             parts.append(f"Source: {meta_norm['source']}")
#         if meta_norm.get("abstract"):
#             parts.append(f"Abstract: {meta_norm['abstract']}")

#         return Document(
#             page_content="\n".join(parts),
#             metadata=meta_norm
#         )

#     # ✅ Save to SESSIONS
#     if chat_id not in SESSIONS:
#         SESSIONS[chat_id] = {"docs": []}
#     SESSIONS[chat_id]["docs"] = [make_doc(d) for d in results]
#     logging.info(f"[metadata_search] ✅ Saved {len(results)} docs to chat_id={chat_id}")

#     # ✅ Also sync with structured memory
#     session = SESSIONS.get(chat_id)
#     if session and "mem" in session:
#         session["mem"].set_docs(session["docs"])

#     # Format output...
#     if not results:
#         return "_(No matching papers found.)_"

#     # ============================================================
#     # 🧾 Step 6 — Format output
#     # ============================================================
#     if not results:
#         return "_(No matching papers found.)_"

#     formatted = []
#     for d in results[:10]:  # show up to 10 results
#         formatted.append(
#             f"- **Title:** {d.get('Title', d.get('title', 'Unknown'))}\n"
#             f"  - Authors: {d.get('Authors', d.get('authors', 'Unknown'))}\n"
#             f"  - Year: {d.get('Year', d.get('year', 'N/A'))}\n"
#             f"  - Source: {d.get('Source', d.get('source', 'N/A'))}\n"
#             f"  - Abstract: {d.get('Abstract', d.get('abstract', 'N/A'))}\n"
#         )

#     return "\n".join(formatted)

class MetadataSearchInput(BaseModel):
    filters: dict = Field(..., description="Metadata filters dictionary")
    user_request: str = ""
    chat_id: str = "default"

# @tool("metadata_search", return_direct=True)
@tool(
    "metadata_search",
    args_schema=MetadataSearchInput,
    return_direct=True
)
def metadata_search(filters: Union[str, dict], user_request: str = "", chat_id: str = "default") -> str:
    """
    🔎 Search academic papers using structured metadata filters in ChromaDB.

    Supports both JSON strings and 'key:value' input formats.
    Each chat tab keeps its own cached results in SESSIONS[chat_id]['docs'].

    Supported filter keys:
      - title
      - authors
      - year_min, year_max
      - source / venue
    """
    # ============================================================
    # 🧠 Step 0 — Lazy imports
    # ============================================================
    try:
        from service.chroma import _chroma_client, COLLECTION_MAPPING
    except Exception as e:
        return f"❌ Failed to load Chroma client: {e}"

    try:
        from service.agent_runner import SESSIONS
    except Exception:
        SESSIONS = {}

    import json, re, logging
    from langchain.schema import Document

    # ============================================================
    # 🧩 Step 1 — Parse filters flexibly
    # ============================================================
    try:
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except json.JSONDecodeError:
                logging.info(f"[metadata_search] Non-JSON filters → using key:value parse: {filters}")
                pattern = r'(\w+)\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([\w\s\-\_]+))'
                matches = re.findall(pattern, filters)
                filters = {
                    k.lower(): next((v for v in vals if v), "").strip()
                    for k, *vals in matches
                }
        
        # Handle case where filters is a list (malformed agent call)
        if isinstance(filters, list):
            logging.warning(f"[metadata_search] Received list instead of dict, attempting conversion: {filters}")
            # Try to convert list to dict if possible
            if len(filters) > 0 and isinstance(filters[0], dict):
                filters = filters[0]  # Use first dict in list
            else:
                filters = {}  # Empty dict as fallback
        
        if not isinstance(filters, dict):
            return "❌ Invalid input: filters must be a JSON string or dict."
    except Exception as e:
        logging.error(f"[metadata_search] Parse error: {e}")
        return f"❌ Invalid filter format: {e}"

    # ============================================================
    # 🧹 Step 2 — Normalize filters
    # ============================================================
    filters_norm = {}

    for k, v in filters.items():
        key = k.lower().strip()

        # Skip None
        if v is None:
            continue

        # Skip empty strings / whitespace strings
        if isinstance(v, str) and not v.strip():
            continue

        # Skip empty lists
        if isinstance(v, list):
            if not v:          # ← CRITICAL FIX
                continue
            filters_norm[key] = [str(i).lower().strip() for i in v]
            continue

        # Normal case
        filters_norm[key] = str(v).lower().strip()

    # Parse year numbers
    for num_key in ["year_min", "year_max"]:
        if num_key in filters_norm:
            try:
                filters_norm[num_key] = int(re.findall(r"\d{4}", str(filters_norm[num_key]))[0])
            except Exception:
                filters_norm.pop(num_key, None)

    # ============================================================
    # 🧠 Step 3 — Get Chroma collection and metadata
    # ============================================================
    if _chroma_client is None:
        return "❌ Chroma client not initialized."

    collection_name = COLLECTION_MAPPING.get("specter", "paper_specter")
    chroma_collection = _chroma_client.get_or_create_collection(collection_name)
    all_docs = chroma_collection.get(include=["metadatas", "documents"])
    metadatas = all_docs.get("metadatas", [])
    logging.info(f"[metadata_search] Checking {len(metadatas)} metadata entries...")

    # ============================================================
    # 🧮 Step 4 — Robust filtering
    # ============================================================
    # results = []
    # for meta in metadatas:
    #     # Normalize all keys/values
    #     m = {}
    #     for k, v in meta.items():
    #         key = str(k).lower()
    #         if isinstance(v, list):
    #             val = ", ".join(map(str, v)).lower()
    #         else:
    #             val = str(v).lower()
    #         m[key] = val

    #     # --- Title match ---
    #     if "title" in filters_norm:
    #         title_val = filters_norm["title"]
    #         if isinstance(title_val, list):
    #             if not any(t in m.get("title", "") for t in title_val):
    #                 continue
    #         else:
    #             # if title_val not in m.get("title", ""):
    #             #     continue
    #             def normalize_title(x):
    #                 return re.sub(r"[^a-z0-9 ]+", " ", x.lower()).strip()

    #             q_title = normalize_title(title_val)
    #             db_title = normalize_title(m.get("title", ""))

    #             if q_title not in db_title:
    #                 continue

    #     # --- Authors match (robust)
    #     if "authors" in filters_norm:
    #         author_val = filters_norm["authors"]
    #         author_field = m.get("authors", "")
    #         if isinstance(author_val, list):
    #             if not any(a in author_field for a in author_val):
    #                 continue
    #         else:
    #             if author_val not in author_field:
    #                 continue

    #     # --- Year range ---
    #     try:
    #         year = int(re.findall(r"\d{4}", m.get("year", ""))[0])
    #     except Exception:
    #         year = 0
    #     if "year_min" in filters_norm and year < filters_norm["year_min"]:
    #         continue
    #     if "year_max" in filters_norm and year > filters_norm["year_max"]:
    #         continue

    #     # --- Venue / Source ---
    #     if "source" in filters_norm and filters_norm["source"] not in m.get("source", ""):
    #         continue
    #     if "venue" in filters_norm and filters_norm["venue"] not in m.get("source", ""):
    #         continue

    #     results.append(meta)

    results = []

    for meta in metadatas:

        # Convert metadata keys to lowercase for lookup,
        # but PRESERVE original values.
        m = {str(k).lower(): v for k, v in meta.items()}

        # ----- Filter: title -----
        if "title" in filters_norm:
            query_title = filters_norm["title"]
            db_title = str(m.get("title", "")).lower()

            # Normalize both
            def norm(x):
                return re.sub(r"[^a-z0-9 ]+", " ", x.lower()).strip()

            if norm(query_title) not in norm(db_title):
                continue

        # ----- Filter: authors -----
        if "authors" in filters_norm:
            values = filters_norm["authors"]
            db_authors = str(m.get("authors", "")).lower()
            if isinstance(values, list):
                if not any(v.lower() in db_authors for v in values):
                    continue
            else:
                if values.lower() not in db_authors:
                    continue

        # ----- Filter: keywords -----
        if "keywords" in filters_norm:
            values = filters_norm["keywords"]
            db_keywords = str(m.get("keywords", "")).lower()
            if isinstance(values, list):
                if not any(v.lower() in db_keywords for v in values):
                    continue
            else:
                if values.lower() not in db_keywords:
                    continue

        # ----- Filter: source / venue -----
        if "source" in filters_norm:
            if filters_norm["source"].lower() not in str(m.get("source", "")).lower():
                continue

        # ----- Filter: id_list -----
        if "id_list" in filters_norm:
            db_id = str(m.get("id", "")).strip()
            if db_id not in filters_norm["id_list"]:
                continue

        # ----- Filter: year_min / year_max -----
        if "year_min" in filters_norm or "year_max" in filters_norm:
            try:
                year = int(re.findall(r"\d{4}", str(m.get("year", "")))[0])
            except Exception:
                continue

            if "year_min" in filters_norm and year < filters_norm["year_min"]:
                continue
            if "year_max" in filters_norm and year > filters_norm["year_max"]:
                continue

        # If passed all filters → keep
        results.append(meta)

    # ============================================================
    # 🧾 Step 5 — Save to session + format output
    # ============================================================
    # def make_doc(meta: dict) -> Document:
    #     meta_norm = {k.lower(): v for k, v in meta.items()}
    #     parts = []
    #     if meta_norm.get("title"):
    #         parts.append(f"Title: {meta_norm['title']}")
    #     if meta_norm.get("authors"):
    #         parts.append(f"Authors: {meta_norm['authors']}")
    #     if meta_norm.get("year"):
    #         parts.append(f"Year: {meta_norm['year']}")
    #     if meta_norm.get("source"):
    #         parts.append(f"Source: {meta_norm['source']}")
    #     if meta_norm.get("abstract"):
    #         parts.append(f"Abstract: {meta_norm['abstract']}")
    #     return Document(page_content="\n".join(parts), metadata=meta_norm)

    def make_doc(meta: dict) -> Document:
        meta_norm = {k.lower(): v for k, v in meta.items()}

        # Ensure ID preserved
        if meta_norm.get("id"):
            meta_norm["ID"] = meta_norm["id"]

        parts = []
        if meta_norm.get("title"):
            parts.append(f"Title: {meta_norm['title']}")
        if meta_norm.get("authors"):
            parts.append(f"Authors: {meta_norm['authors']}")
        if meta_norm.get("year"):
            parts.append(f"Year: {meta_norm['year']}")
        if meta_norm.get("source"):
            parts.append(f"Source: {meta_norm['source']}")
        if meta_norm.get("abstract"):
            parts.append(f"Abstract: {meta_norm['abstract']}")

        return Document(page_content="\n".join(parts), metadata=meta_norm)

    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = {"docs": [], "mem": None}
    
    # Ensure docs key exists in session
    session = SESSIONS.get(chat_id)
    if "docs" not in session:
        session["docs"] = []
    
    # Add docs to session
    session["docs"].extend([make_doc(d) for d in results])
    
    # Also add to turn buffer
    turn_buffer = session.setdefault("_turn_docs", [])
    turn_buffer.extend([make_doc(d) for d in results])
    logging.info(f"[metadata_search] ✅ Saved {len(results)} docs to chat_id={chat_id}")

    # Sync with structured memory
    if session and "mem" in session and session["mem"] is not None:
        session["mem"].set_docs(session["docs"])

    # Format output for display
    if not results:
        return "_(No matching papers found.)_"

    formatted = []
    # for d in results[:10]:
    #     formatted.append(
    #         f"- **Title:** {d.get('Title', d.get('title', 'Unknown'))}\n"
    #         f"  - Authors: {d.get('Authors', d.get('authors', 'Unknown'))}\n"
    #         f"  - Year: {d.get('Year', d.get('year', 'N/A'))}\n"
    #         f"  - Source: {d.get('Source', d.get('source', 'N/A'))}\n"
    #         f"  - Abstract: {d.get('Abstract', d.get('abstract', 'N/A'))}\n"
    #     )
    for d in results[:10]:
        ID = d.get("ID") or d.get("id") or "Unknown"
        formatted.append(
            f"- **Title:** {d.get('Title', d.get('title', 'Unknown'))} [[ID:{ID}]]\n"
            f"  - Authors: {d.get('Authors', d.get('authors', 'Unknown'))}\n"
            f"  - Year: {d.get('Year', d.get('year', 'N/A'))}\n"
            f"  - Source: {d.get('Source', d.get('source', 'N/A'))}\n"
            f"  - Abstract: {d.get('Abstract', d.get('abstract', 'N/A'))}\n"
        )

    return "\n".join(formatted)

@tool("semantic_search", return_direct=True)
def semantic_search(query: str, chat_id: str = "default") -> str:
    """
    Perform semantic retrieval (Specter + CrossEncoder rerank),  
    scoped to a specific chat session.
    """
    # log(f"[semantic_search] 🔎 Query='{query}' in chat_id={chat_id}")
    logging.info(f"[semantic_search] 🔎 Query='{query}' in chat_id={chat_id}")

    docs = rag_core._run_semantic_search(query_text=query, chat_id=chat_id)
    if not docs:
        return "_(No relevant papers found.)_"

    # Docs are already stored in session within rag_core
    return rag_core.format_docs(docs)



@tool("mixed_search", return_direct=True)
def mixed_search(query_text, filters, chat_id, top_k=5):
    """
    Hybrid search combining metadata + semantic retrieval.
    Returns formatted results if overlap found,
    otherwise a polite LLM-controllable fallback message (no hallucination).
    """
    from service.rag_core import hybrid_refine, _run_metadata_search, _run_semantic_search, save_session_docs, format_docs

    meta_docs = _run_metadata_search(filters, chat_id)
    sem_docs = _run_semantic_search(query_text, chat_id, top_k=20)

    final_docs, found_overlap = hybrid_refine(
        meta_docs,
        sem_docs,
        query_text,
        top_k=top_k,
        alpha=0.7
    )

    if not found_overlap:
        logging.info("[hybrid] ❌ No overlapping papers found — returning fallback message.")
        author = filters.get("authors", ["the specified author"])[0]
        year_min = filters.get("year_min", "N/A")
        year_max = filters.get("year_max", "N/A")
        topic = query_text
        # Instead of hardcoding a full sentence, we tell the LLM to handle it
        return (
            f"SYSTEM_NOTICE: No papers were found.\n"
            f"- Author filter: {author}\n"
            f"- Year range: {year_min}–{year_max}\n"
            f"- Query topic: {topic}\n"
            "Please respond politely, explain that no results were found, "
            "and offer to broaden the search (e.g., drop author or topic filter)."
        )

    save_session_docs(chat_id, final_docs)
    logging.info(f"[hybrid] ✅ Retrieved {len(final_docs)} hybrid-matched papers for chat_id={chat_id}")
    return format_docs(final_docs)


# =====================================================
# 🧠 Recall Memory Tool
# =====================================================
# @tool("recall_memory", return_direct=True)
# def recall_memory(query: str, chat_id: str = "default") -> str:
#     """Recall documents or facts cached in this chat session."""
#     import logging
#     log = lambda msg: logging.info(msg)

#     # 🧠 Import lazily to avoid circular import
#     try:
#         from service.agent_runner import SESSIONS
#     except ImportError:
#         log("[recall_memory] ⚠️ Could not import SESSIONS, using empty fallback.")
#         SESSIONS = {}

#     session = SESSIONS.get(chat_id)
#     if not session or not session.get("docs"):
#         return "_(No cached memory found for this chat.)_"

#     docs = session["docs"]
#     log(f"[recall_memory] Retrieved {len(docs)} docs from chat_id={chat_id}")

#     formatted = []
#     for d in docs[:10]:
#         formatted.append(
#             f"- **Title:** {d.get('Title', 'Unknown')}\n"
#             f"  - Authors: {d.get('Authors', 'Unknown')}\n"
#             f"  - Year: {d.get('Year', 'N/A')}\n"
#             f"  - Source: {d.get('Source', 'N/A')}\n"
#         )
#     return "\n".join(formatted)

# @tool("recall_memory", return_direct=True)
# def recall_memory(query: str, chat_id: str = "default") -> str:
#     """Recall documents or facts cached in this chat session or structured memory."""
#     import logging
#     log = lambda msg: logging.info(msg)

#     from service.agent_runner import SESSIONS
#     session = SESSIONS.get(chat_id, {})

#     docs = session.get("docs", [])
#     if not docs:
#         mem = session.get("mem")
#         if mem and getattr(mem, "doc_cache", None):
#             docs = mem.doc_cache

#     if not docs:
#         return "_(No cached or remembered papers found in this chat.)_"

#     log(f"[recall_memory] Retrieved {len(docs)} docs from chat_id={chat_id}")
#     return format_docs(docs[:10])


# @tool("select_top_k", return_direct=False)
# def select_top_k_tool(count: int = 5, chat_id: str = "default") -> str:
#     """
#     Select top-N most relevant papers from the current chat session's cached docs.
#     Each chat window keeps its own independent doc cache.
#     """
#     import service.rag_core as rag_core
#     from service.agent_runner import SESSIONS

#     # 🧠 Check per-chat session cache
#     session = SESSIONS.get(chat_id)
#     if not session:
#         return f"⚠️ No active session found for chat_id={chat_id}."

#     docs = session.get("docs", [])
#     if not docs:
#         return "⚠️ No cached papers available to select from."

#     # 🧩 Trim results safely
#     try:
#         count = max(1, min(int(count), len(docs)))
#     except Exception:
#         count = 5

#     selected = docs[:count]

#     # ✅ Update the per-session doc cache
#     session["docs"] = selected
#     rag_core.save_session_docs(chat_id, selected)

#     return rag_core.format_docs(selected)

# =====================================================
# 🧩 Tool Registration
# =====================================================
ALL_AGENT_TOOLS = [
    metadata_search,
    semantic_search,
    # recall_memory,
    # select_top_k_tool,
    mixed_search,
]