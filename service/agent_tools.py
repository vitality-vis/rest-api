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
from pydantic import BaseModel, Field
from typing import Union, List
from langchain.tools import tool


class MetadataSearchInput(BaseModel):
    filters: dict = Field(..., description="Metadata filters dictionary")
    user_request: str = ""
    chat_id: str = "default"


@tool(
    "metadata_search",
    args_schema=MetadataSearchInput,
    return_direct=True
)
def metadata_search(filters: Union[str, dict], user_request: str = "", chat_id: str = "default") -> str:
    """
    Search academic papers using structured metadata filters in ChromaDB.

    Authors: AND logic  
    Source: AND logic  
    Title: scalar only (list ignored -> first element)  
    Full normalization of punctuation, periods, hyphens (fixes 'Jenny T. Liang' search issue)
    """

    # ============================================================
    # Step 0 — Lazy imports
    # ============================================================
    try:
        from service.chroma import _chroma_client, COLLECTION_MAPPING
    except Exception as e:
        return f"Failed to load Chroma client: {e}"

    try:
        from service.agent_runner import SESSIONS
    except Exception:
        SESSIONS = {}

    import json, re, logging
    from langchain.schema import Document

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

    for k, v in filters.items():
        key = k.lower().strip()
        if v is None:
            continue

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
    # Step 3 — Load metadata
    # ============================================================
    if _chroma_client is None:
        return "Chroma client not initialized."

    collection_name = COLLECTION_MAPPING.get("specter", "paper_specter")
    coll = _chroma_client.get_or_create_collection(collection_name)

    all_docs = coll.get(include=["metadatas", "documents"])
    metadatas = all_docs.get("metadatas", [])

    logging.info(f"[metadata_search] Checking {len(metadatas)} metadata entries...")

    # ============================================================
    # Step 4 — Normalization helper (critical)
    # ============================================================
    def norm_text(x):
        """
        Normalize text by:
        - lowercasing
        - removing punctuation (.,-/ etc.)
        - compressing spaces
        """
        x = str(x).lower()
        x = re.sub(r"[^a-z0-9 ]+", " ", x)
        x = re.sub(r"\s+", " ", x)
        return x.strip()

    # ============================================================
    # Step 5 — Apply AND logic filtering
    # ============================================================
    results = []

    for meta in metadatas:
        m = {str(k).lower(): v for k, v in meta.items()}
        m_norm = {
            "title": norm_text(m.get("title", "")),
            "authors": norm_text(m.get("authors", "")),
            # "keywords": norm_text(m.get("keywords", "")),
            "source": norm_text(m.get("source", "")),
            "year": norm_text(m.get("year", "")),
            "id": norm_text(m.get("id", "")),
        }

        # --------------------
        # TITLE (scalar ONLY)
        # --------------------
        if "title" in filters_norm:
            q = norm_text(filters_norm["title"])
            if q not in m_norm.get("title", ""):
                continue

        # --------------------
        # AUTHORS (AND logic)
        # --------------------
        if "authors" in filters_norm:
            db_val = m_norm.get("authors", "")
            vals = filters_norm["authors"]
            if isinstance(vals, list):
                if not all(norm_text(v) in db_val for v in vals):
                    continue
            else:
                if norm_text(vals) not in db_val:
                    continue

        # --------------------
        # SOURCE (AND logic)
        # --------------------
        if "source" in filters_norm:
            db_val = m_norm.get("source", "")
            vals = filters_norm["source"]
            if isinstance(vals, list):
                if not all(norm_text(v) in db_val for v in vals):
                    continue
            else:
                if norm_text(vals) not in db_val:
                    continue

        # --------------------
        # YEAR RANGE
        # --------------------
        if "year_min" in filters_norm or "year_max" in filters_norm:
            try:
                year = int(re.findall(r"\d{4}", m_norm.get("year", ""))[0])
            except Exception:
                continue
            if "year_min" in filters_norm and year < filters_norm["year_min"]:
                continue
            if "year_max" in filters_norm and year > filters_norm["year_max"]:
                continue

        # results.append(meta)
        results.append(m)

    # ============================================================
    # Step 6 — Save to session + format output
    # ============================================================
    def make_doc(meta: dict) -> Document:
        meta_norm = {k.lower(): v for k, v in meta.items()}
        if "id" in meta_norm:
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

        return Document("\n".join(parts), metadata=meta_norm)

    session = SESSIONS.setdefault(chat_id, {})
    turn_buffer = session.setdefault("_turn_docs", [])
    turn_buffer.extend([make_doc(d) for d in results])

    logging.info(f"[metadata_search] Saved {len(results)} docs to chat_id={chat_id}")

    # Use sliding-window memory if exists
    if session.get("mem"):
        session["mem"].set_docs(session["_turn_docs"])

    # ============================================================
    # Step 7 — Output
    # ============================================================
    if not results:
        return "_(No matching papers found.)_"

    formatted = []
    for d in results[:10]:
        ID = d.get("ID") or d.get("id", "Unknown")
        formatted.append(
            f"- **Title:** {d.get('title', 'Unknown')} [[ID:{ID}]]\n"
            f"  - Authors: {d.get('authors', 'Unknown')}\n"
            f"  - Year: {d.get('year', 'N/A')}\n"
            f"  - Source: {d.get('source', 'N/A')}\n"
            f"  - Abstract: {d.get('abstract', 'N/A')}\n"
        )

    return "\n".join(formatted)


@tool("semantic_search", return_direct=True)
def semantic_search(query: str, chat_id: str = "default") -> str:
    """
    Perform semantic retrieval (Specter + CrossEncoder rerank),  
    scoped to a specific chat session.
    """
    # log(f"[semantic_search]  Query='{query}' in chat_id={chat_id}")
    logging.info(f"[semantic_search]  Query='{query}' in chat_id={chat_id}")

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
        logging.info("[hybrid] No overlapping papers found — returning fallback message.")
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
    logging.info(f"[hybrid] Retrieved {len(final_docs)} hybrid-matched papers for chat_id={chat_id}")
    return format_docs(final_docs)

ALL_AGENT_TOOLS = [
    metadata_search,
    semantic_search,
    mixed_search,
]