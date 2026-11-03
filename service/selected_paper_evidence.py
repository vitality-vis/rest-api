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
    paper_ids: list[str] | None = None
    group: str = "all"


def inspect_selected_papers(
    *,
    user_id: str | None,
    project_id: str | None = None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None = None,
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

    groups = _normalize_groups(selected, selected_paper_groups)
    sampled_ids, sample_policy = _choose_ids(
        selected,
        min(len(selected), MAX_INSPECTION_RECORDS),
        groups=groups,
    )
    papers = _resolve(user_id=user_id, project_id=project_id, paper_ids=sampled_ids)
    sampled = [_title_record(paper, include_availability=True) for paper in papers]
    years = [
        int(record["year"])
        for record in sampled
        if isinstance(record.get("year"), int)
    ]
    return {
        "selected_count": len(selected),
        "groups": {name: {"count": len(ids)} for name, ids in groups.items()},
        "sampled_count": len(sampled),
        "sample_policy": sample_policy,
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
    project_id: str | None = None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None = None,
    request: EvidenceRequest,
) -> dict[str, object]:
    """Return bounded metadata from the trusted current selection, without an LLM call."""
    selected = _normalize_ids(selected_paper_ids)
    if not selected:
        raise SelectedPaperEvidenceError("Select at least one paper first.")

    groups = _normalize_groups(selected, selected_paper_groups)
    if request.group == "all":
        group_candidates = selected
    elif request.group in groups:
        group_candidates = groups[request.group]
    else:
        raise SelectedPaperEvidenceError(
            f"Unknown paper group: {request.group}. Available groups: "
            f"{', '.join(groups) or 'none'}."
        )
    candidates = _requested_subset(group_candidates, request.paper_ids)
    detail = request.detail_level
    if detail not in {"auto", "titles", "abstracts"}:
        raise SelectedPaperEvidenceError("detail_level must be auto, titles, or abstracts.")
    title_limit = _bounded_limit(request.limit, maximum=MAX_TITLE_RECORDS)
    if detail == "abstracts":
        title_limit = min(title_limit, MAX_ABSTRACT_RECORDS)
    chosen_ids, sampling_policy = _choose_ids(
        candidates,
        title_limit,
        group=request.group,
        groups=groups,
    )
    resolved = _resolve(user_id=user_id, project_id=project_id, paper_ids=chosen_ids)

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
        "group": request.group,
        "groups": {name: {"count": len(ids)} for name, ids in groups.items()},
        "candidate_count": len(candidates),
        "returned_count": len(records),
        "detailed_count": detailed_count,
        "omitted_count": max(0, len(candidates) - len(records)),
        "detail_level": detail,
        "sampling_policy": sampling_policy,
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


def _normalize_groups(
    selected: list[str], groups: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    if not groups:
        return {}
    selected_set = set(selected)
    normalized: dict[str, list[str]] = {}
    for raw_name, raw_ids in groups.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise SelectedPaperEvidenceError("Paper group names must be non-empty strings.")
        ids = _normalize_ids(raw_ids)
        if any(paper_id not in selected_set for paper_id in ids):
            raise SelectedPaperEvidenceError(
                "Paper groups can only contain papers in the current selection."
            )
        normalized[raw_name.strip()] = ids
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


def select_representative_paper_ids(
    paper_ids: list[str], *, limit: int, prefer_map_coverage: bool = True
) -> tuple[list[str], str]:
    """Return a bounded deterministic seed set for trusted server-side tools."""
    normalized = _normalize_ids(paper_ids)
    bounded = _bounded_limit(limit, maximum=max(1, limit))
    return _choose_ids(
        normalized,
        min(len(normalized), bounded),
        group="map" if prefer_map_coverage else "all",
    )


def _choose_ids(
    ids: list[str],
    limit: int,
    *,
    group: str = "all",
    groups: dict[str, list[str]] | None = None,
) -> tuple[list[str], str]:
    if len(ids) <= limit:
        return ids, "complete_selection"
    if group.casefold() == "map":
        return _map_coverage_or_even(ids, limit)
    if group != "all" or not groups:
        return _even_coverage(ids, limit)
    return _stratified_representative_ids(ids, limit, groups)


def _even_coverage(ids: list[str], limit: int) -> tuple[list[str], str]:
    return (
        [ids[index] for index in _sample_positions(len(ids), limit)],
        "deterministic_even_coverage",
    )


def _map_coverage_or_even(ids: list[str], limit: int) -> tuple[list[str], str]:
    spatial = _map_coverage_ids(ids, limit)
    if spatial is not None:
        return spatial, "deterministic_map_coverage"
    return _even_coverage(ids, limit)


def _stratified_representative_ids(
    ids: list[str], limit: int, groups: dict[str, list[str]]
) -> tuple[list[str], str]:
    """Sample named sources separately so a union does not hide smaller groups."""
    candidates = set(ids)
    assigned: set[str] = set()
    strata: list[tuple[str, list[str]]] = []
    # Assign overlap to Map first so those papers retain spatial sampling semantics.
    ordered_groups = sorted(
        groups.items(), key=lambda item: (item[0].casefold() != "map",)
    )
    for name, group_ids in ordered_groups:
        stratum = [
            paper_id
            for paper_id in group_ids
            if paper_id in candidates and paper_id not in assigned
        ]
        if stratum:
            strata.append((name, stratum))
            assigned.update(stratum)
    ungrouped = [paper_id for paper_id in ids if paper_id not in assigned]
    if ungrouped:
        strata.append(("__ungrouped__", ungrouped))
    if len(strata) == 1:
        name, stratum = strata[0]
        if name.casefold() == "map":
            return _map_coverage_or_even(stratum, limit)
        return _even_coverage(stratum, limit)

    quotas = _stratum_quotas([len(stratum) for _, stratum in strata], limit)
    chosen: list[str] = []
    for (name, stratum), quota in zip(strata, quotas, strict=True):
        if name.casefold() == "map":
            sampled, _ = _map_coverage_or_even(stratum, quota)
        else:
            sampled, _ = _even_coverage(stratum, quota)
        chosen.extend(sampled)
    return chosen, "deterministic_stratified_coverage"


def _stratum_quotas(sizes: list[int], limit: int) -> list[int]:
    """Allocate proportionally while representing every stratum when possible."""
    quotas = [0] * len(sizes)
    if limit >= len(sizes):
        quotas = [1] * len(sizes)
    for _ in range(limit - sum(quotas)):
        candidates = [index for index, size in enumerate(sizes) if quotas[index] < size]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (limit * sizes[item] / sum(sizes) - quotas[item], -item),
        )
        quotas[index] += 1
    return quotas


def _umap_points() -> list[dict[str, object]]:
    from service.static_cache import cached_data

    points = cached_data.get_umap_points() or []
    return [point for point in points if isinstance(point, dict)]


def _map_coverage_ids(ids: list[str], limit: int) -> list[str] | None:
    """Greedily cover UMAP space; deterministically fill missing-coordinate slots."""
    if limit <= 0:
        return []
    wanted = set(ids)
    coordinates: dict[str, tuple[float, float]] = {}
    for point in _umap_points():
        if not isinstance(point, dict):
            continue
        paper_id = point.get("ID")
        raw = point.get("specter_umap") or point.get("umap")
        if not isinstance(paper_id, str) or paper_id not in wanted or not isinstance(raw, dict):
            continue
        x, y = raw.get("x"), raw.get("y")
        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
        ):
            coordinates[paper_id] = (float(x), float(y))
    # A single mapped point cannot describe spatial coverage.
    if len(coordinates) < 2:
        return None

    mapped = [paper_id for paper_id in ids if paper_id in coordinates]
    order = {paper_id: index for index, paper_id in enumerate(ids)}
    target = min(limit, len(mapped))
    centroid = (
        sum(coordinates[paper_id][0] for paper_id in mapped) / len(mapped),
        sum(coordinates[paper_id][1] for paper_id in mapped) / len(mapped),
    )
    first = min(
        mapped,
        key=lambda paper_id: (
            (coordinates[paper_id][0] - centroid[0]) ** 2
            + (coordinates[paper_id][1] - centroid[1]) ** 2,
            order[paper_id],
        ),
    )
    chosen = [first]
    remaining = set(mapped) - {first}
    while len(chosen) < target:
        next_id = max(
            remaining,
            key=lambda paper_id: (
                min(
                    (coordinates[paper_id][0] - coordinates[selected][0]) ** 2
                    + (coordinates[paper_id][1] - coordinates[selected][1]) ** 2
                    for selected in chosen
                ),
                -order[paper_id],
            ),
        )
        chosen.append(next_id)
        remaining.remove(next_id)

    if len(chosen) < limit:
        unmapped = [paper_id for paper_id in ids if paper_id not in coordinates]
        slots = min(limit - len(chosen), len(unmapped))
        chosen.extend(unmapped[index] for index in _sample_positions(len(unmapped), slots))
    return chosen


def _sample_positions(count: int, limit: int) -> list[int]:
    if limit <= 0 or count <= 0:
        return []
    if limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    return [(index * (count - 1)) // (limit - 1) for index in range(limit)]


def _resolve(
    *, user_id: str | None, project_id: str | None, paper_ids: list[str]
) -> list[ResolvedLibraryPaper]:
    try:
        return resolve_selected_papers(
            user_id=user_id, project_id=project_id, paper_ids=paper_ids
        )
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
