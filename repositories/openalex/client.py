"""OpenAlex Works API adapter for citation neighbor lookup."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import quote

import requests

import config

OPENALEX_API_BASE = "https://api.openalex.org"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
MAX_PER_PAGE = 100
# Keep OR-filter batches small enough for typical URL length limits.
REFERENCE_BATCH_SIZE = 50

WORK_SUMMARY_SELECT = "id,doi,title,publication_year,cited_by_count"
SOURCE_WORK_SELECT = f"{WORK_SUMMARY_SELECT},referenced_works"


class OpenAlexError(RuntimeError):
    """Base error for OpenAlex API failures."""


class OpenAlexConfigurationError(OpenAlexError):
    """Raised when OPENALEX_API_KEY is missing."""


class OpenAlexTransientError(OpenAlexError):
    """Raised for timeouts, rate limits, and OpenAlex 5xx responses."""


def get_openalex_api_key(api_key: Optional[str] = None) -> str:
    """Return a configured OpenAlex API key or raise if unavailable."""
    resolved = (api_key if api_key is not None else config.OPENALEX_API_KEY).strip()
    if not resolved:
        raise OpenAlexConfigurationError("OpenAlex is not configured")
    return resolved


def normalize_doi(doi: str) -> str:
    """Strip common DOI prefixes down to the bare DOI string."""
    trimmed = (doi or "").strip()
    if not trimmed:
        return ""
    lowered = trimmed.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            return trimmed[len(prefix) :].strip()
    return trimmed


def normalize_openalex_id(work_id: str) -> str:
    """Normalize a work id to the short OpenAlex form, e.g. ``W2741809807``."""
    trimmed = (work_id or "").strip()
    if not trimmed:
        return ""
    if trimmed.startswith("https://openalex.org/"):
        return trimmed[len("https://openalex.org/") :].strip()
    if trimmed.startswith("https://api.openalex.org/works/"):
        return trimmed[len("https://api.openalex.org/works/") :].strip()
    if trimmed.startswith("openalex.org/"):
        return trimmed[len("openalex.org/") :].strip()
    return trimmed.lstrip("/")


def summarize_work(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project an OpenAlex work into the MVP citation-neighbor shape."""
    openalex_id = normalize_openalex_id(str(raw.get("id") or ""))
    doi = normalize_doi(str(raw.get("doi") or ""))
    title = raw.get("title")
    year = raw.get("publication_year")
    cited_by_count = raw.get("cited_by_count")
    summary: dict[str, Any] = {
        "openalex_id": openalex_id or None,
        "title": title if isinstance(title, str) and title.strip() else None,
        "year": year if isinstance(year, int) else None,
        "doi": doi or None,
        "cited_by_count": cited_by_count if isinstance(cited_by_count, int) else None,
    }
    return summary


def resolve_work_by_doi(
    doi: str,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> Optional[dict[str, Any]]:
    """Resolve one OpenAlex work by DOI.

    Returns a summarized work plus ``referenced_works`` (OpenAlex ids) for
    backward expansion. Returns ``None`` when OpenAlex has no match.
    """
    normalized = normalize_doi(doi)
    if not normalized:
        return None

    payload = _get_json(
        "/works",
        params={
            "filter": f"doi:https://doi.org/{normalized}",
            "per-page": 1,
            "select": SOURCE_WORK_SELECT,
        },
        api_key=api_key,
        session=session,
        timeout=timeout,
    )
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results:
        return None

    first = results[0]
    if not isinstance(first, Mapping):
        return None

    summary = summarize_work(first)
    referenced = first.get("referenced_works") or []
    summary["referenced_works"] = [
        normalize_openalex_id(str(item))
        for item in referenced
        if isinstance(item, str) and normalize_openalex_id(item)
    ]
    return summary


def list_referenced_works(
    work_id: str,
    limit: int,
    *,
    referenced_work_ids: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch works cited by ``work_id`` (backward / references).

    Pass ``referenced_work_ids`` when the caller already resolved the source
    work, to avoid a second lookup.
    """
    safe_limit = _bounded_limit(limit)
    if safe_limit <= 0:
        return []

    ids = [
        normalize_openalex_id(item)
        for item in (referenced_work_ids or [])
        if normalize_openalex_id(item)
    ]
    if referenced_work_ids is None:
        source = _get_work_raw(
            work_id,
            select=SOURCE_WORK_SELECT,
            api_key=api_key,
            session=session,
            timeout=timeout,
        )
        if source is None:
            return []
        referenced = source.get("referenced_works") or []
        ids = [
            normalize_openalex_id(str(item))
            for item in referenced
            if isinstance(item, str) and normalize_openalex_id(item)
        ]

    ids = ids[:safe_limit]
    if not ids:
        return []

    works_by_id: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), REFERENCE_BATCH_SIZE):
        chunk = ids[start : start + REFERENCE_BATCH_SIZE]
        filter_value = "openalex:" + "|".join(chunk)
        payload = _get_json(
            "/works",
            params={
                "filter": filter_value,
                "per-page": min(MAX_PER_PAGE, len(chunk)),
                "select": WORK_SUMMARY_SELECT,
            },
            api_key=api_key,
            session=session,
            timeout=timeout,
        )
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            continue
        for raw in results:
            if not isinstance(raw, Mapping):
                continue
            summary = summarize_work(raw)
            if summary.get("openalex_id"):
                works_by_id[str(summary["openalex_id"])] = summary

    # Preserve the source paper's reference order.
    return [works_by_id[item] for item in ids if item in works_by_id]


def list_citing_works(
    work_id: str,
    limit: int,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch works that cite ``work_id`` (forward / cited-by)."""
    safe_limit = _bounded_limit(limit)
    normalized_id = normalize_openalex_id(work_id)
    if safe_limit <= 0 or not normalized_id:
        return []

    collected: list[dict[str, Any]] = []
    page = 1
    while len(collected) < safe_limit:
        remaining = safe_limit - len(collected)
        per_page = min(MAX_PER_PAGE, remaining)
        payload = _get_json(
            "/works",
            params={
                "filter": f"cites:{normalized_id}",
                "per-page": per_page,
                "page": page,
                "select": WORK_SUMMARY_SELECT,
            },
            api_key=api_key,
            session=session,
            timeout=timeout,
        )
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list) or not results:
            break

        for raw in results:
            if not isinstance(raw, Mapping):
                continue
            summary = summarize_work(raw)
            if summary.get("openalex_id"):
                collected.append(summary)
            if len(collected) >= safe_limit:
                break

        if len(results) < per_page:
            break
        page += 1

    return collected


def _bounded_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 0
    return max(0, min(value, 200))


def _get_work_raw(
    work_id: str,
    *,
    select: str,
    api_key: Optional[str],
    session: Optional[requests.Session],
    timeout: float,
) -> Optional[Mapping[str, Any]]:
    normalized_id = normalize_openalex_id(work_id)
    if not normalized_id:
        return None
    payload = _get_json(
        f"/works/{quote(normalized_id, safe='')}",
        params={"select": select},
        api_key=api_key,
        session=session,
        timeout=timeout,
        allow_404=True,
    )
    if payload is None or not isinstance(payload, Mapping):
        return None
    return payload


def _get_json(
    path: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    allow_404: bool = False,
) -> Any:
    key = get_openalex_api_key(api_key)
    query: MutableMapping[str, Any] = dict(params or {})
    query["api_key"] = key
    url = f"{OPENALEX_API_BASE}{path}"
    http = session or requests

    try:
        response = http.get(url, params=query, timeout=timeout)
    except requests.Timeout as error:
        raise OpenAlexTransientError("OpenAlex request timed out") from error
    except requests.RequestException as error:
        raise OpenAlexTransientError("Could not reach OpenAlex") from error

    if allow_404 and response.status_code == 404:
        return None
    if response.status_code in {429, 500, 502, 503, 504}:
        raise OpenAlexTransientError(
            f"OpenAlex is temporarily unavailable ({response.status_code})"
        )
    if response.status_code >= 400:
        raise OpenAlexError(f"OpenAlex request failed with {response.status_code}")

    try:
        return response.json()
    except ValueError as error:
        raise OpenAlexError("OpenAlex returned an invalid JSON response") from error
