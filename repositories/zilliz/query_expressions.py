"""Compilation of application filters into Milvus filter expressions."""
from __future__ import annotations

from typing import List, Optional

from model.paper import PaperFilters


# TODO: Remove this compatibility alias map after Agent tools stop using legacy
# ``where`` dictionaries. Normal paper search uses ``PaperFilters`` below.
_LEGACY_WHERE_FIELD_ALIASES = {
    "ID": "paper_uid",
    "Title": "title",
    "Abstract": "abstract",
    "Authors": "authors",
    "Keywords": "keywords",
    "Source": "source",
    "Year": "year",
    "CitationCounts": "citation_count",
}


def ids_to_expr(ids: List[str]) -> str:
    """Build an ID membership expression, or an expression matching all rows."""
    if not ids:
        return 'paper_uid != ""'
    escaped = [f'"{str(identifier).replace(chr(34), "")}"' for identifier in ids]
    return "paper_uid in [" + ", ".join(escaped) + "]"


def escape_like(value: str) -> str:
    """Escape wildcard characters for a Milvus ``LIKE`` pattern."""
    escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped.replace('"', '\\"')


def escape_text_match(value: str) -> str:
    """Escape a query term embedded in a Milvus ``TEXT_MATCH`` string literal."""
    return str(value).lower().replace("\\", "\\\\").replace('"', '\\"')


def where_to_expr(where: dict) -> str:
    """Convert the legacy agent-tools where syntax into a Milvus expression."""
    if not where:
        return 'paper_uid != ""'

    parts = []
    for raw_field, value in where.items():
        field = _LEGACY_WHERE_FIELD_ALIASES.get(raw_field, raw_field)
        if isinstance(value, dict):
            if "$eq" in value:
                parts.append(f'{field} == "{str(value["$eq"]).replace(chr(34), "")}"')
            elif "$in" in value:
                escaped = [f'"{str(item).replace(chr(34), "")}"' for item in value["$in"]]
                parts.append(f"{field} in [{', '.join(escaped)}]")
            elif "$nin" in value:
                escaped = [f'"{str(item).replace(chr(34), "")}"' for item in value["$nin"]]
                parts.append(f"{field} not in [{', '.join(escaped)}]")
            elif "$gte" in value:
                parts.append(f"{field} >= {int(value['$gte'])}")
            elif "$lte" in value:
                parts.append(f"{field} <= {int(value['$lte'])}")
            elif "$contains" in value:
                parts.append(f'{field} like "%{escape_like(value["$contains"])}%"')
            elif "$contains_all" in value:
                for item in value["$contains_all"]:
                    parts.append(f'{field} like "%{escape_like(item)}%"')
        else:
            parts.append(f'{field} == "{str(value).replace(chr(34), "")}"')
    return " and ".join(parts) if parts else 'paper_uid != ""'


def split_query_terms(value: Optional[str]) -> List[str]:
    """Parse terms for the comma-separated cross-field ``search_query``."""
    if not value:
        return []
    return [term.strip() for term in value.split(",") if term.strip()]


def build_paper_query_expr(
    filters: PaperFilters,
    *,
    query_text: Optional[str] = None,
    include_query_text: bool = True,
) -> str:
    """Translate supported paper filters into a Milvus scalar expression.

    Filtering stays in Zilliz so a page request never materialises the complete
    collection in Python. ``search_query`` uses the analyzed, lower-case
    ``search_text`` field; the remaining field-specific filters retain their
    current Milvus ``like`` / array semantics.
    """
    parts = []

    def like_all(field: str, value: Optional[str]):
        if not value:
            return
        for term in (item.strip() for item in value.split(",")):
            if term:
                parts.append(f'{field} like "%{escape_like(term)}%"')

    def like_any(field: str, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        matches = [
            f'{field} like "%{escape_like(value)}%"'
            for value in values
            if str(value).strip()
        ]
        if matches:
            parts.append("(" + " or ".join(matches) + ")")

    def array_contains_any(field: str, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        matches = [
            f'array_contains({field}, "{escape_like(value)}")'
            for value in values
            if str(value).strip()
        ]
        if matches:
            parts.append("(" + " or ".join(matches) + ")")

    # Each comma-separated search_query term must match the analyzed search_text
    # field. Ingestion lower-cases that field and combines title, abstract,
    # authors, keywords, and source, so this is a case-insensitive cross-field
    # keyword search without pulling the collection into Python.
    if include_query_text:
        for term in split_query_terms(query_text):
            parts.append(f'TEXT_MATCH(search_text, "{escape_text_match(term)}")')

    like_all("title", filters.title)
    like_all("abstract", filters.abstract)
    like_any("source", filters.source)
    array_contains_any("authors", filters.author)
    array_contains_any("keywords", filters.keyword)

    if filters.min_year is not None:
        parts.append(f"year >= {int(filters.min_year)}")
    if filters.max_year is not None:
        parts.append(f"year <= {int(filters.max_year)}")
    if filters.min_citation_counts is not None:
        parts.append(f"citation_count >= {int(filters.min_citation_counts)}")
    if filters.max_citation_counts is not None:
        parts.append(f"citation_count <= {int(filters.max_citation_counts)}")
    if filters.id_list:
        parts.append(ids_to_expr([str(paper_id) for paper_id in filters.id_list]))

    return " and ".join(parts) if parts else 'paper_uid != ""'
