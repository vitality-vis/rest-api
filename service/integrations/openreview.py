"""
OpenReview API v2 client for live paper search.

Uses the public OpenReview notes/search endpoint — no API key required.
Returns [] on any error so the caller can continue with other sources.
"""
import asyncio
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from logger_config import get_logger

logger = get_logger()

_OR_SEARCH_URL = "https://api2.openreview.net/notes/search"
# Short timeout: OpenReview is an optional source; fail fast rather than block the pipeline.
_OR_TIMEOUT = httpx.Timeout(connect=4.0, read=3.5, write=4.0, pool=4.0)


async def openreview_search(
    query: str,
    limit: int = 20,
    year_min: Optional[int] = None,
) -> List[Document]:
    """
    Search OpenReview for papers matching *query*.

    Args:
        query:    free-text search string
        limit:    max results to return
        year_min: if set, skip notes whose creation year is before this value

    Returns:
        List[Document] with metadata._source="openreview".
        Returns [] on any network / API error — non-blocking.
    """
    params = {
        "term":   query,
        "limit":  limit,
        "offset": 0,
    }

    logger.info(
        "[OpenReview] query=%r limit=%d year_min=%s", query[:60], limit, year_min
    )

    try:
        async with httpx.AsyncClient(timeout=_OR_TIMEOUT) as client:
            resp = await client.get(_OR_SEARCH_URL, params=params)

        if resp.status_code == 429:
            logger.warning("[OpenReview] rate-limited (429) for query=%r — skipping source", query[:60])
            return []

        resp.raise_for_status()
        data = resp.json()

    except httpx.TimeoutException:
        logger.warning("[OpenReview] timed out for query=%r — skipping source", query[:60])
        return []
    except Exception as exc:
        logger.warning("[OpenReview] search failed for query=%r: %s — skipping source", query[:60], exc)
        return []

    docs: List[Document] = []
    for note in data.get("notes") or []:
        try:
            content  = note.get("content") or {}

            title    = (content.get("title")    or {}).get("value", "")
            abstract = (content.get("abstract") or {}).get("value", "")

            if not title or not abstract:
                continue

            # OpenReview returns authors as list[str], list[dict], or {"value": [...]}
            authors_field = content.get("authors") or {}
            if isinstance(authors_field, dict):
                raw_list = authors_field.get("value") or []
            elif isinstance(authors_field, list):
                raw_list = authors_field
            else:
                raw_list = []
            authors: List[str] = []
            for a in raw_list:
                if isinstance(a, dict):
                    name = (a.get("name") or a.get("fullname") or a.get("username") or "").strip()
                elif isinstance(a, str):
                    name = a.strip()
                else:
                    name = str(a).strip()
                if name:
                    authors.append(name)

            # Year from cdate (Unix ms timestamp) or creation_date string
            cdate     = note.get("cdate") or note.get("creation_date") or ""
            year_str  = str(cdate)[:4]
            try:
                year = int(year_str) if year_str.isdigit() else None
            except ValueError:
                year = None

            if year_min and year is not None and year < year_min:
                continue

            # Venue: prefer note-level, fall back to content field
            venue_content = content.get("venue") or {}
            venue = (
                note.get("venue")
                or (venue_content.get("value") if isinstance(venue_content, dict) else venue_content)
                or "OpenReview"
            )

            note_id = note.get("id", "")

            # DOI: explicitly published papers may carry one; try common field names
            doi_field = (
                content.get("doi")
                or content.get("paper_doi")
                or content.get("_doi")
                or {}
            )
            if isinstance(doi_field, dict):
                doi = str(doi_field.get("value") or "").strip()
            else:
                doi = str(doi_field or "").strip()
            # Strip URL prefix if present
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

            docs.append(Document(
                page_content=abstract or title,
                metadata={
                    "title":    title,
                    "abstract": abstract,
                    "authors":  authors,
                    "year":     year,
                    "source":   str(venue),
                    "id":       f"or_{note_id}",
                    "doi":      doi,
                    "_score":   0.0,
                    "_source":  "openreview",
                },
            ))
        except Exception:
            continue  # skip malformed notes, never raise to caller

    logger.info("[OpenReview] query=%r → %d results", query[:60], len(docs))
    return docs
