"""Budgeted, model-free evidence access for papers selected by the user."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from service.paper_registry import (
    LibraryPaperResolutionError,
    ResolvedLibraryPaper,
    resolve_selected_papers,
)

DetailLevel = Literal["auto", "titles", "abstracts"]
SamplingMode = Literal["representative", "selected_order"]

MAX_TITLE_RECORDS = 100
MAX_ABSTRACT_RECORDS = 20
MAX_INSPECTION_RECORDS = 12
SMALL_SELECTION_LIMIT = 8
SMALL_ABSTRACT_CHARS = 2_500
LARGE_ABSTRACT_CHARS = 1_000


class SelectedPaperEvidenceError(RuntimeError):
    """User-safe selected-paper evidence failure."""


@dataclass(frozen=True)
class EvidenceRequest:
    detail_level: DetailLevel = "auto"
    limit: int | None = None
    sampling: SamplingMode = "representative"
    paper_ids: list[str] | None = None


def inspect_selected_papers(
    *, user_id: str | None, selected_paper_ids: list[str]
) -> dict[str, object]:
    """Return a cheap overview and a small representative metadata sample."""
    selected = _normalize_ids(selected_paper_ids)
    if not selected:
        return {
            "selected_count": 0,
            "sampled_count": 0,
            "sampled_papers": [],
            "instruction": "No papers are selected.",
        }

    sampled_ids = _sample_ids(selected, min(len(selected), MAX_INSPECTION_RECORDS))
    papers = _resolve(user_id=user_id, paper_ids=sampled_ids)
    sampled = [_title_record(paper, include_availability=True) for paper in papers]
    years = [
        int(record["year"])
        for record in sampled
        if isinstance(record.get("year"), int)
    ]
    return {
        "selected_count": len(selected),
        "sampled_count": len(sampled),
        "sample_policy": "deterministic_even_coverage",
        "sampled_papers": sampled,
        "sample_year_range": [min(years), max(years)] if years else None,
        "instruction": (
            "Use get_selected_paper_evidence to retrieve bounded titles or abstracts. "
            "The sample is an overview, not the complete selection."
        ),
    }


def get_selected_paper_evidence(
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
    request: EvidenceRequest,
) -> dict[str, object]:
    """Return bounded metadata from the trusted current selection, without an LLM call."""
    selected = _normalize_ids(selected_paper_ids)
    if not selected:
        raise SelectedPaperEvidenceError("Select at least one paper first.")

    candidates = _requested_subset(selected, request.paper_ids)
    detail = request.detail_level
    if detail not in {"auto", "titles", "abstracts"}:
        raise SelectedPaperEvidenceError("detail_level must be auto, titles, or abstracts.")
    if request.sampling not in {"representative", "selected_order"}:
        raise SelectedPaperEvidenceError(
            "sampling must be representative or selected_order."
        )

    title_limit = _bounded_limit(request.limit, maximum=MAX_TITLE_RECORDS)
    if detail == "abstracts":
        title_limit = min(title_limit, MAX_ABSTRACT_RECORDS)
    chosen_ids = _choose_ids(candidates, title_limit, request.sampling)
    resolved = _resolve(user_id=user_id, paper_ids=chosen_ids)

    if detail == "auto":
        abstract_count = min(len(resolved), MAX_ABSTRACT_RECORDS)
    elif detail == "abstracts":
        abstract_count = len(resolved)
    else:
        abstract_count = 0

    detailed_positions = set(
        _sample_positions(len(resolved), abstract_count)
        if abstract_count and abstract_count < len(resolved)
        else range(abstract_count)
    )
    abstract_chars = (
        SMALL_ABSTRACT_CHARS
        if len(candidates) <= SMALL_SELECTION_LIMIT
        else LARGE_ABSTRACT_CHARS
    )
    records = [
        _evidence_record(
            paper,
            include_abstract=index in detailed_positions,
            abstract_chars=abstract_chars,
        )
        for index, paper in enumerate(resolved)
    ]
    detailed_count = sum("abstract" in record for record in records)
    return {
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "returned_count": len(records),
        "detailed_count": detailed_count,
        "omitted_count": max(0, len(candidates) - len(records)),
        "detail_level": detail,
        "sampling": request.sampling,
        "policy": {
            "max_titles": MAX_TITLE_RECORDS,
            "max_abstracts": MAX_ABSTRACT_RECORDS,
            "abstract_chars": abstract_chars if detailed_count else 0,
        },
        "papers": records,
        "instruction": _evidence_instruction(
            candidate_count=len(candidates),
            returned_count=len(records),
            detailed_count=detailed_count,
        ),
    }


def _normalize_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        raise SelectedPaperEvidenceError("paper IDs must be an array of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SelectedPaperEvidenceError("paper IDs must be an array of strings.")
        paper_id = item.strip()
        if paper_id not in seen:
            seen.add(paper_id)
            normalized.append(paper_id)
    return normalized


def _requested_subset(selected: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return selected
    normalized = _normalize_ids(requested)
    if not normalized:
        raise SelectedPaperEvidenceError("paper_ids cannot be empty when supplied.")
    selected_set = set(selected)
    outside = [paper_id for paper_id in normalized if paper_id not in selected_set]
    if outside:
        raise SelectedPaperEvidenceError(
            "Evidence can only be retrieved for papers in the current selection."
        )
    return normalized


def _bounded_limit(value: int | None, *, maximum: int) -> int:
    if value is None:
        return maximum
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectedPaperEvidenceError("limit must be an integer.")
    if value < 1:
        raise SelectedPaperEvidenceError("limit must be at least 1.")
    return min(value, maximum)


def _choose_ids(ids: list[str], limit: int, sampling: SamplingMode) -> list[str]:
    if len(ids) <= limit:
        return ids
    if sampling == "selected_order":
        return ids[:limit]
    return [ids[index] for index in _sample_positions(len(ids), limit)]


def _sample_ids(ids: list[str], limit: int) -> list[str]:
    return _choose_ids(ids, limit, "representative")


def _sample_positions(count: int, limit: int) -> list[int]:
    if limit <= 0 or count <= 0:
        return []
    if limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    return [(index * (count - 1)) // (limit - 1) for index in range(limit)]


def _resolve(
    *, user_id: str | None, paper_ids: list[str]
) -> list[ResolvedLibraryPaper]:
    try:
        return resolve_selected_papers(user_id=user_id, paper_ids=paper_ids)
    except LibraryPaperResolutionError as error:
        raise SelectedPaperEvidenceError(str(error)) from error


def _title_record(
    paper: ResolvedLibraryPaper, *, include_availability: bool = False
) -> dict[str, object]:
    metadata = paper.metadata
    record: dict[str, object] = {
        "paper_id": paper.paper_id,
        "title": str(metadata.get("Title") or ""),
        "year": metadata.get("Year"),
        "venue": str(metadata.get("Source") or ""),
    }
    if include_availability:
        library = paper.library_paper or {}
        record["full_text_available"] = bool(
            library.get("vs_file_status") == "completed"
            and library.get("azure_file_id")
        )
    return record


def _evidence_record(
    paper: ResolvedLibraryPaper, *, include_abstract: bool, abstract_chars: int
) -> dict[str, object]:
    metadata = paper.metadata
    record = _title_record(paper)
    record.update(
        {
            "authors": list(metadata.get("Authors") or [])[:12],
            "keywords": list(metadata.get("Keywords") or [])[:20],
            "citation_count": metadata.get("CitationCounts"),
            "doi": metadata.get("doi"),
        }
    )
    if include_abstract:
        abstract = str(metadata.get("Abstract") or "")
        record["abstract"] = abstract[:abstract_chars]
        record["abstract_truncated"] = len(abstract) > abstract_chars
    return record


def _evidence_instruction(
    *, candidate_count: int, returned_count: int, detailed_count: int
) -> str:
    scope = (
        f"Evidence covers {returned_count} of {candidate_count} requested selected papers; "
        f"{detailed_count} include abstracts."
    )
    if returned_count < candidate_count or detailed_count < returned_count:
        scope += " Disclose this limited evidence scope in the answer."
    return (
        f"{scope} Generate the answer yourself from these records. "
        "Cite factual paper claims as [[ID:paper_id]] using only supplied IDs."
    )
