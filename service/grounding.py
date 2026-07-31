"""Pure citation-marker normalization shared by research-answer paths."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def numbered_paper_ids(paper_ids: Sequence[str]) -> dict[int, str]:
    """Return the one-based citation-number map for an ordered evidence set."""
    return {number: paper_id for number, paper_id in enumerate(paper_ids, start=1)}


def replace_numbered_citations(answer: str, paper_ids_by_number: Mapping[int, str]) -> str:
    """Resolve model-only ``[[n]]`` citations to stable paper-ID markers.

    Numbers outside the supplied evidence map remain unchanged.  Compact model
    output such as ``[[1],[2],[4]]`` is normalized to individual markers.
    """
    def marker(number: str) -> str:
        paper_id = paper_ids_by_number.get(int(number))
        return f"[[ID:{paper_id}]]" if paper_id else f"[[{number}]]"

    def replace_group(match: re.Match[str]) -> str:
        return "".join(marker(number) for number in re.findall(r"\d+", match.group(1)))

    def replace_single(match: re.Match[str]) -> str:
        return marker(match.group(1))

    answer = re.sub(r"\[\[(\d+(?:\]\s*,\s*\[\d+)+)\]\]", replace_group, answer)
    return re.sub(r"\[\[(\d+)\]\]", replace_single, answer)


def resolve_file_annotations(
    annotations: Sequence[Mapping[str, object]], file_to_paper_id: Mapping[str, str]
) -> tuple[list[dict[str, object]], list[str]]:
    """Map allowed Azure file annotations to paper IDs without rendering them."""
    resolved: list[dict[str, object]] = []
    unexpected_file_ids: set[str] = set()
    for annotation in annotations:
        file_id = annotation.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            continue
        paper_id = file_to_paper_id.get(file_id)
        if not paper_id:
            unexpected_file_ids.add(file_id)
            continue
        resolved_annotation = dict(annotation)
        resolved_annotation["paper_id"] = paper_id
        resolved.append(resolved_annotation)
    return resolved, sorted(unexpected_file_ids)


def apply_file_citations(
    answer: str, annotations: Sequence[Mapping[str, object]]
) -> str:
    """Render trusted File Search citations without guessing annotation placement.

    Azure annotations with an explicit citation token are replaced in place.  If
    Azure does not provide an unambiguous token, the corresponding paper is
    listed as a full-text source at the end of the answer instead.
    """
    inline_answer = answer
    fallback_paper_ids: list[str] = []
    span_annotations: list[Mapping[str, object]] = []
    for annotation in annotations:
        paper_id = annotation.get("paper_id")
        start_index = annotation.get("start_index")
        end_index = annotation.get("end_index")
        if (
            isinstance(paper_id, str)
            and paper_id
            and isinstance(start_index, int)
            and isinstance(end_index, int)
            and annotation.get("output_text") == answer
            and 0 <= start_index < end_index <= len(answer)
        ):
            span_annotations.append(annotation)

    for annotation in sorted(
        span_annotations, key=lambda item: int(item["start_index"]), reverse=True
    ):
        paper_id = str(annotation["paper_id"])
        start_index = int(annotation["start_index"])
        end_index = int(annotation["end_index"])
        inline_answer = (
            f"{inline_answer[:start_index]}[[ID:{paper_id}]]{inline_answer[end_index:]}"
        )

    span_annotation_ids = {id(annotation) for annotation in span_annotations}
    for annotation in annotations:
        paper_id = annotation.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        if id(annotation) in span_annotation_ids:
            continue
        token = annotation.get("text")
        if isinstance(token, str) and token and inline_answer.count(token) == 1:
            inline_answer = inline_answer.replace(token, f"[[ID:{paper_id}]]", 1)
            continue
        if paper_id not in fallback_paper_ids:
            fallback_paper_ids.append(paper_id)

    if not fallback_paper_ids:
        return inline_answer
    sources = " ".join(f"[[ID:{paper_id}]]" for paper_id in fallback_paper_ids)
    return f"{inline_answer}\n\nFull-text sources: {sources}"
