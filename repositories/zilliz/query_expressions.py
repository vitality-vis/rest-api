"""Compilation of application filters into Milvus filter expressions."""
from __future__ import annotations

from typing import List, Optional

from model.paper import GetPapersRequest


# TODO: Migrate agent tools from legacy ``where`` dictionaries to
# GetPapersRequest/repository methods, then remove this compatibility alias map.
# Normal route filters are compiled from GetPapersRequest below.
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


def build_paper_query_expr(query: GetPapersRequest) -> str:
    """Translate supported paper filters into a Milvus scalar expression.

    Filtering stays in Zilliz so a page request never materialises the complete
    collection in Python. Milvus ``like`` and array filters are case-sensitive.

    TODO: At ingestion, add a lowercase ``search_text`` field that concatenates
    title, abstract, authors, keywords, and source. Querying that field will
    make cross-field search case-insensitive and avoid the current mix of
    substring matching for text fields and exact matching for array fields.
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

    # Each comma-separated search_query term must match, but can match a
    # different field.
    # Text fields use substring matching; Authors and Keywords are arrays and
    # therefore use exact element matching in this first implementation.
    for term in split_query_terms(query.search_query):
        escaped = escape_like(term)
        matches = [
            f'title like "%{escaped}%"',
            f'abstract like "%{escaped}%"',
            f'source like "%{escaped}%"',
            f'array_contains(authors, "{escaped}")',
            f'array_contains(keywords, "{escaped}")',
        ]
        parts.append("(" + " or ".join(matches) + ")")

    like_all("title", query.title)
    like_all("abstract", query.abstract)
    like_any("source", query.source)
    array_contains_any("authors", query.author)
    array_contains_any("keywords", query.keyword)

    if query.min_year is not None:
        parts.append(f"year >= {int(query.min_year)}")
    if query.max_year is not None:
        parts.append(f"year <= {int(query.max_year)}")
    if query.min_citation_counts is not None:
        parts.append(f"citation_count >= {int(query.min_citation_counts)}")
    if query.max_citation_counts is not None:
        parts.append(f"citation_count <= {int(query.max_citation_counts)}")
    if query.id_list:
        parts.append(ids_to_expr([str(paper_id) for paper_id in query.id_list]))

    return " and ".join(parts) if parts else 'paper_uid != ""'


def query_has_filters(query: GetPapersRequest) -> bool:
    """Whether a query uses any field that changes the collection-wide total."""
    return any(
        value is not None and value != [] and value != ""
        for value in (
            query.title,
            split_query_terms(query.search_query),
            query.abstract,
            query.author,
            query.source,
            query.keyword,
            query.min_year,
            query.max_year,
            query.min_citation_counts,
            query.max_citation_counts,
            query.id_list,
        )
    )
