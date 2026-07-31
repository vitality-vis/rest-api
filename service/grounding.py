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
