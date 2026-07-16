# """
# OpenAlex API client for live academic paper search.

# Replaces the Semantic Scholar client — identical function signature, zero
# changes required in rag_core.py.

# Why OpenAlex:
#   - No API key required
#   - Generous rate limits (100k req/day unauthenticated; higher with mailto)
#   - Full abstract + rich metadata (venue, citations, year)
#   - Abstract stored as inverted index → reconstructed here before returning

# Polite pool: set OPENALEX_MAILTO in .env to get higher rate limits and
# priority support from the OpenAlex team.  No key needed — just an email.
# """


# import os
# from typing import List, Optional

# import httpx
# from langchain_core.documents import Document
# from logger_config import get_logger

# logger = get_logger()

# _OA_SEARCH_URL = "https://api.openalex.org/works"

# # Fields requested from the API.  abstract_inverted_index is the only way
# # OpenAlex exposes abstracts — it must be reconstructed from word→[positions].
# _OA_SELECT = (
#     "id,doi,title,abstract_inverted_index,"
#     "authorships,publication_year,"
#     "primary_location,cited_by_count"
# )


# def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
#     """
#     Rebuild plain-text abstract from OpenAlex inverted-index format.

#     Format: {"word": [pos1, pos2, ...], ...}
#     Empty dict or None → empty string.
#     """
#     if not inverted_index:
#         return ""
#     position_word: dict[int, str] = {}
#     for word, positions in inverted_index.items():
#         for pos in positions:
#             position_word[pos] = word
#     return " ".join(position_word[i] for i in sorted(position_word))


# async def semantic_scholar_search(
#     query: str,
#     limit: int = 20,
#     year_min: Optional[int] = None,
#     year_max: Optional[int] = None,
# ) -> List[Document]:
#     """
#     Search OpenAlex for papers matching *query*.

#     Function name kept as semantic_scholar_search for drop-in compatibility
#     with the import in rag_core.py.

#     Args:
#         query:    free-text search string
#         limit:    max results (capped at 200 by the API)
#         year_min: optional lower bound on publication year
#         year_max: optional upper bound on publication year

#     Returns:
#         List[Document] with metadata._source="openalex".
#         Returns [] on any network / API error — non-blocking.
#     """
#     mailto = os.getenv("OPENALEX_MAILTO", "").strip()

#     params: dict = {
#         "search":   query,
#         "per-page": min(limit, 200),
#         "select":   _OA_SELECT,
#     }

#     # Year range filter (OpenAlex filter syntax)
#     if year_min and year_max:
#         params["filter"] = f"publication_year:{year_min}-{year_max}"
#     elif year_min:
#         params["filter"] = f"publication_year:{year_min}-"
#     elif year_max:
#         params["filter"] = f"publication_year:-{year_max}"

#     # Polite pool: include mailto for higher rate limits
#     if mailto:
#         params["mailto"] = mailto

#     headers = {
#         "User-Agent": (
#             f"RAGSearchBot/1.0 (mailto:{mailto})" if mailto else "RAGSearchBot/1.0"
#         ),
#     }

#     logger.info(
#         "[OpenAlex] query=%r limit=%d mailto=%s",
#         query[:60], limit, bool(mailto),
#     )

#     try:
#         async with httpx.AsyncClient(
#             timeout=httpx.Timeout(connect=4.0, read=3.5, write=4.0, pool=4.0)
#         ) as client:
#             resp = await client.get(_OA_SEARCH_URL, params=params, headers=headers)

#         if resp.status_code == 429:
#             logger.warning("[OpenAlex] rate-limited (429) for query=%r — skipping source", query[:60])
#             return []

#         resp.raise_for_status()
#         data = resp.json()

#     except httpx.TimeoutException:
#         logger.warning("[OpenAlex] timed out for query=%r — skipping source", query[:60])
#         return []
#     except Exception as exc:
#         logger.warning("[OpenAlex] search failed for query=%r: %s — skipping source", query[:60], exc)
#         return []

#     docs: List[Document] = []
#     for work in data.get("results") or []:
#         # Strip the URL prefix from the OpenAlex ID  (e.g. "https://openalex.org/W123" → "W123")
#         raw_id   = str(work.get("id") or "")
#         paper_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

#         title    = str(work.get("title") or "")
#         abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
#         year     = work.get("publication_year")
#         cited_by = int(work.get("cited_by_count") or 0)
#         raw_doi  = str(work.get("doi") or "")
#         doi      = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

#         authors = [
#             str(a.get("author", {}).get("display_name", ""))
#             for a in (work.get("authorships") or [])
#             if a.get("author", {}).get("display_name")
#         ]

#         # Venue: primary_location → source → display_name
#         primary_loc = work.get("primary_location") or {}
#         source_obj  = primary_loc.get("source") or {}
#         venue       = str(source_obj.get("display_name") or "")

#         docs.append(Document(
#             page_content=abstract or title,
#             metadata={
#                 "title":         title,
#                 "abstract":      abstract,
#                 "authors":       authors,
#                 "year":          year,
#                 "source":        venue,
#                 "id":            paper_id,
#                 "doi":           doi,
#                 "_score":        0.0,
#                 "_source":       "openalex",
#                 "citationCount": cited_by,
#             },
#         ))

#     logger.info("[OpenAlex] query=%r → %d results", query[:60], len(docs))
#     return docs


# ── Semantic Scholar snippet search (active implementation) ───────────────────

import os
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from logger_config import get_logger

logger = get_logger()

_SS_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
# Request text + kind (abstract / body) + section name.
# Paper info (corpusId, title, authors, openAccessInfo) and score are always returned.
_SS_SNIPPET_FIELDS = "snippet.text,snippet.snippetKind,snippet.section"


async def semantic_scholar_search(
    query: str,
    limit: int = 20,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Document]:
    """
    Search Semantic Scholar via the snippet search endpoint.

    Returns ~500-word text excerpts (title, abstract, body) ranked by relevance.
    Results are deduplicated by corpus ID — only the highest-scoring snippet per
    paper is kept — then the top `limit` papers are returned as Documents.

    Set SEMANTIC_SCHOLAR_API_KEY in .env for authenticated (higher rate-limit) access.
    Function signature is identical to the old OpenAlex version for drop-in compatibility.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

    # Fetch 3× more snippets than needed so deduplication can still return `limit` papers.

    requested_limit = min(limit, 5)
    params: dict = {
        "query":  query,
        # "limit":  min(limit * 3, 500),
        "limit": requested_limit,  # Snippet endpoint max limit is 100
        "fields": _SS_SNIPPET_FIELDS,
    }

    # Year filter — Semantic Scholar accepts "YYYY", "YYYY-YYYY", "YYYY-", or "-YYYY"
    if year_min and year_max:
        params["year"] = f"{year_min}-{year_max}"
    elif year_min:
        params["year"] = f"{year_min}-"
    elif year_max:
        params["year"] = f"-{year_max}"

    headers: dict = {"User-Agent": "RAGSearchBot/1.0"}
    if api_key:
        headers["x-api-key"] = api_key

    logger.info(
        "[SemanticScholar] query=%r limit=%d auth=%s",
        query[:60], limit, bool(api_key),
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=4.0, read=6.0, write=4.0, pool=4.0)
        ) as client:
            resp = await client.get(_SS_SNIPPET_URL, params=params, headers=headers)

        if resp.status_code == 429:
            logger.warning(
                "[SemanticScholar] rate-limited (429) for query=%r — skipping source",
                query[:60],
            )
            return []

        resp.raise_for_status()
        data = resp.json()

        # Deduplicate by corpus ID, keeping the highest-scoring snippet per paper.
        best: dict = {}
        for item in (data.get("data") or [] if isinstance(data, dict) else []):
            if not isinstance(item, dict):
                continue
            paper     = item.get("paper") or {}
            if not isinstance(paper, dict):
                continue
            corpus_id = str(paper.get("corpusId") or "")
            if not corpus_id:
                continue
            score = float(item.get("score") or 0.0)
            if corpus_id not in best or score > best[corpus_id]["score"]:
                best[corpus_id] = {"item": item, "score": score}

        # Sort descending by score; take top `limit` papers.
        ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:limit]

        docs: List[Document] = []
        for entry in ranked:
            item    = entry["item"]
            paper   = item.get("paper") or {}
            snippet = item.get("snippet") or {}
            if not isinstance(snippet, dict):
                snippet = {}

            title     = str(paper.get("title") or "")
            text      = str(snippet.get("text") or "")
            corpus_id = str(paper.get("corpusId") or "")
            score     = entry["score"]

            authors = [
                str(a.get("name", ""))
                for a in (paper.get("authors") or [])
                if isinstance(a, dict) and a.get("name")
            ]

            docs.append(Document(
                page_content=text or title,
                metadata={
                    "title":         title,
                    "abstract":      text,
                    "authors":       authors,
                    "year":          None,   # not returned by snippet endpoint
                    "source":        "",     # venue not returned by snippet endpoint
                    "id":            corpus_id,
                    "doi":           "",
                    "_score":        score,
                    "_source":       "semantic_scholar",
                    "citationCount": 0,
                },
            ))

        logger.info(
            "[SemanticScholar] query=%r → %d results (after dedup)", query[:60], len(docs)
        )
        return docs

    except httpx.TimeoutException:
        logger.warning(
            "[SemanticScholar] timed out for query=%r — skipping source", query[:60]
        )
        return []
    except Exception as exc:
        logger.warning(
            "[SemanticScholar] search failed for query=%r: %s — skipping source",
            query[:60], exc,
        )
        return []
