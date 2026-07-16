"""
ArXiv paper search via the official ArXiv Atom/XML API.

No API key required. DOI is extracted from the <link title="doi"> element
when a paper is published to a journal; otherwise the canonical arXiv
preprint DOI (10.48550/arXiv.<id>) is returned as fallback.

Circuit-breaker pattern (CLOSED → OPEN → HALF_OPEN → CLOSED):
  CLOSED    — normal operation; requests are allowed.
  OPEN      — ArXiv is considered unavailable; returns [] immediately without
              making any network request.
  HALF_OPEN — cooldown expired; exactly one probe request is allowed.
              Any failure immediately reopens the circuit (no second chance).

All circuit-breaker state mutations happen inside _ARXIV_SEMAPHORE to
prevent concurrent-write races in async code.
"""
import asyncio
import os
import time
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from logger_config import get_logger

logger = get_logger()

_ARXIV_URL = "https://export.arxiv.org/api/query"
_NS_ATOM   = "http://www.w3.org/2005/Atom"
_NS_ARXIV  = "http://arxiv.org/schemas/atom"

# ── Circuit-breaker configuration ──────────────────────────────────────────
_ARXIV_FAILURE_THRESHOLD = 2    # consecutive qualifying failures before opening
_ARXIV_COOLDOWN_SECONDS  = 120  # seconds the circuit stays OPEN
_ARXIV_MIN_INTERVAL      = 3.2  # minimum seconds between actual network requests

# Serialises all ArXiv calls and gates every state mutation.
_ARXIV_SEMAPHORE = asyncio.Semaphore(1)

# User-Agent built once at import time; reuses OPENALEX_MAILTO if configured.
_mailto     = os.getenv("OPENALEX_MAILTO", "").strip()
_USER_AGENT = (
    f"ArXivRAGSearch/0.1 (mailto:{_mailto})" if _mailto else "ArXivRAGSearch/0.1"
)


class _CircuitState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# All mutations occur inside _ARXIV_SEMAPHORE — no concurrent-write races.
_cb_state         : _CircuitState = _CircuitState.CLOSED
_cb_failures      : int           = 0
_cb_open_until    : float         = 0.0  # monotonic: when OPEN → HALF_OPEN
_cb_last_req_time : float         = 0.0  # monotonic: last actual request fire time


# ── Circuit-breaker helpers ─────────────────────────────────────────────────

def _transition_to_open(from_state: _CircuitState, cooldown: float) -> None:
    """Open the circuit for *cooldown* seconds and log the transition."""
    global _cb_state, _cb_open_until
    _cb_state      = _CircuitState.OPEN
    _cb_open_until = time.monotonic() + cooldown
    logger.warning(
        "[ArXiv] circuit %s → OPEN (cooldown=%.0fs)",
        from_state.value, cooldown,
    )


def _transition_to_closed() -> None:
    """
    Return the circuit to normal operation.
    Resets failure count and clears the cooldown timestamp.
    Only logs when leaving a non-CLOSED state to avoid per-request noise.
    """
    global _cb_state, _cb_failures, _cb_open_until
    prev           = _cb_state
    _cb_state      = _CircuitState.CLOSED
    _cb_failures   = 0
    _cb_open_until = 0.0
    if prev != _CircuitState.CLOSED:
        logger.info("[ArXiv] circuit %s → CLOSED", prev.value)


def _record_failure(label: str, cooldown: float = _ARXIV_COOLDOWN_SECONDS) -> None:
    """
    Increment the failure counter and open the circuit when appropriate.

    HALF_OPEN: always opens immediately — a failed probe gets no second chance.
    CLOSED:    opens only when the failure count reaches _ARXIV_FAILURE_THRESHOLD.
    """
    global _cb_failures
    _cb_failures += 1
    logger.warning(
        "[ArXiv] %s — failure %d/%d",
        label, _cb_failures, _ARXIV_FAILURE_THRESHOLD,
    )
    if _cb_state == _CircuitState.HALF_OPEN or _cb_failures >= _ARXIV_FAILURE_THRESHOLD:
        _transition_to_open(_cb_state, cooldown)


# ── DOI extraction ──────────────────────────────────────────────────────────

def _entry_doi(entry: ET.Element, arxiv_id: str) -> str:
    """
    Extract DOI for an ArXiv entry.

    Priority:
      1. <link title="doi" rel="related" href="https://doi.org/..."/>
         — present when the paper is published to a journal
      2. <arxiv:doi>10.xxxx/yyyy</arxiv:doi>
         — alternative journal-DOI field
      3. Canonical arXiv preprint DOI: 10.48550/arXiv.<clean_id>
         — always available; strip version suffix (e.g. v2) first
    """
    # Priority 1 — link element
    for link in entry.findall(f"{{{_NS_ATOM}}}link"):
        if link.get("title") == "doi" and link.get("href"):
            href = link.get("href", "")
            return href.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

    # Priority 2 — arxiv:doi element
    doi_el = entry.find(f"{{{_NS_ARXIV}}}doi")
    if doi_el is not None and doi_el.text:
        return doi_el.text.strip()

    # Priority 3 — canonical preprint DOI (strip version suffix e.g. "2301.00001v2")
    clean_id = arxiv_id.split("v")[0] if arxiv_id else ""
    if clean_id:
        return f"10.48550/arXiv.{clean_id}"

    return ""


# ── Public search function ──────────────────────────────────────────────────

async def arxiv_search(
    query: str,
    limit: int = 20,
    year_min: Optional[int] = None,
) -> List[Document]:
    """
    Search ArXiv for papers matching *query*.

    Returns List[Document] with metadata._source="arxiv" and doi field.
    Returns [] on any failure or when the circuit is OPEN — non-blocking.
    """
    global _cb_state, _cb_failures, _cb_open_until, _cb_last_req_time

    keywords = " ".join(query.split()[:8])

    if year_min:
        search_query = (
            f"all:{keywords} AND "
            f"submittedDate:[{year_min}01010000 TO 99991231235959]"
        )
    else:
        search_query = f"all:{keywords}"

    params = {
        "search_query": search_query,
        "max_results":  limit,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }

    logger.info(
        "[ArXiv] query=%r keywords=%r limit=%d year_min=%s",
        query[:60], keywords[:40], limit, year_min,
    )

    root = None
    async with _ARXIV_SEMAPHORE:
        now = time.monotonic()

        # ── Circuit-breaker gate ────────────────────────────────────────
        # OPEN: return [] immediately so asyncio.gather() doesn't accumulate
        # latency from a known-down source on every concurrent query.
        if _cb_state == _CircuitState.OPEN:
            if now < _cb_open_until:
                logger.info(
                    "[ArXiv] circuit OPEN — skipping request (%.0fs remaining) for query=%r",
                    _cb_open_until - now, query[:60],
                )
                return []
            # Cooldown expired → allow exactly one probe to test recovery.
            # Any failure in HALF_OPEN reopens immediately (no second chance).
            logger.info("[ArXiv] circuit OPEN → HALF_OPEN for query=%r", query[:60])
            _cb_state = _CircuitState.HALF_OPEN

        # ── Polite rate limiting ────────────────────────────────────────
        elapsed = time.monotonic() - _cb_last_req_time
        if elapsed < _ARXIV_MIN_INTERVAL:
            await asyncio.sleep(_ARXIV_MIN_INTERVAL - elapsed)

        _cb_last_req_time = time.monotonic()

        # ── Network request ─────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=4.0, read=5.0, write=4.0, pool=4.0),
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(_ARXIV_URL, params=params)

            # ── 429 ─────────────────────────────────────────────────────
            # Repeated 429s mean the service is overloaded, not just our
            # query — counts toward the breaker so we back off automatically.
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0) or 0)
                cooldown = (
                    max(60, min(300, retry_after)) if retry_after
                    else _ARXIV_COOLDOWN_SECONDS
                )
                _record_failure(
                    f"429 rate-limited for query={query[:60]!r}",
                    cooldown=cooldown,
                )
                return []

            # ── 5xx ─────────────────────────────────────────────────────
            if resp.status_code >= 500:
                _record_failure(f"HTTP {resp.status_code} for query={query[:60]!r}")
                return []

            # ── Other 4xx (non-429) ──────────────────────────────────────
            # These reflect query/client issues, not ArXiv availability —
            # log and skip without tripping the breaker.
            if 400 <= resp.status_code < 500:
                logger.warning(
                    "[ArXiv] HTTP %d client error for query=%r — skipping source",
                    resp.status_code, query[:60],
                )
                return []

            resp.raise_for_status()

            # ── XML parse ───────────────────────────────────────────────
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as exc:
                _record_failure(
                    f"XML parse error for query={query[:60]!r}: {exc}",
                )
                return []

            # ── Success ──────────────────────────────────────────────────
            _transition_to_closed()

        except httpx.TimeoutException:
            _record_failure(f"timed out for query={query[:60]!r}")
            return []

        except (httpx.NetworkError, httpx.TransportError) as exc:
            _record_failure(f"network error for query={query[:60]!r}: {exc}")
            return []

        except Exception as exc:
            logger.warning(
                "[ArXiv] unexpected error for query=%r: [%s] %s — skipping source",
                query[:60], type(exc).__name__, exc,
            )
            return []

    if root is None:
        return []

    docs: List[Document] = []
    for entry in root.findall(f"{{{_NS_ATOM}}}entry"):
        try:
            # ArXiv ID — strip URL prefix and keep e.g. "2301.00001v1"
            raw_id   = (entry.findtext(f"{{{_NS_ATOM}}}id") or "").strip()
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

            title     = (entry.findtext(f"{{{_NS_ATOM}}}title")     or "").strip()
            abstract  = (entry.findtext(f"{{{_NS_ATOM}}}summary")   or "").strip()
            published = (entry.findtext(f"{{{_NS_ATOM}}}published") or "").strip()

            try:
                year = int(published[:4]) if published else None
            except ValueError:
                year = None

            authors = [
                (name.text or "").strip()
                for author in entry.findall(f"{{{_NS_ATOM}}}author")
                for name   in author.findall(f"{{{_NS_ATOM}}}name")
                if name.text
            ]

            categories = [
                cat.get("term", "")
                for cat in entry.findall(f"{{{_NS_ATOM}}}category")
                if cat.get("term")
            ]

            doi = _entry_doi(entry, arxiv_id)

            if not title:
                continue

            docs.append(Document(
                page_content=abstract or title,
                metadata={
                    "title":      title,
                    "abstract":   abstract,
                    "authors":    authors,
                    "year":       year,
                    "source":     "arXiv",
                    "id":         f"arxiv_{arxiv_id}",
                    "doi":        doi,
                    "_score":     0.0,
                    "_source":    "arxiv",
                    "categories": categories,
                },
            ))
        except Exception:
            continue

    logger.info("[ArXiv] query=%r → %d results", query[:60], len(docs))
    return docs
