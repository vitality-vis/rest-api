"""Compilation of application filters into Milvus filter expressions."""
from __future__ import annotations

from dataclasses import dataclass
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


class BooleanSearchExpressionError(ValueError):
    """Raised when the public exact-search syntax cannot be parsed safely."""


@dataclass(frozen=True)
class _BooleanToken:
    kind: str
    value: str = ""


def _escape_match_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _tokenize_boolean_search(expression: str) -> List[_BooleanToken]:
    tokens: List[_BooleanToken] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if char == "(":
            tokens.append(_BooleanToken("LPAREN"))
            index += 1
            continue
        if char == ")":
            tokens.append(_BooleanToken("RPAREN"))
            index += 1
            continue
        if char == '"':
            index += 1
            phrase: List[str] = []
            while index < len(expression) and expression[index] != '"':
                if expression[index] == "\\":
                    index += 1
                    if index >= len(expression):
                        raise BooleanSearchExpressionError("A quoted phrase ends with an incomplete escape.")
                phrase.append(expression[index])
                index += 1
            if index >= len(expression):
                raise BooleanSearchExpressionError("A quoted phrase is missing its closing quote.")
            index += 1
            value = "".join(phrase).strip()
            if not value:
                raise BooleanSearchExpressionError("Quoted phrases cannot be empty.")
            tokens.append(_BooleanToken("PHRASE", value))
            continue

        end = index
        while end < len(expression) and not expression[end].isspace() and expression[end] not in "()":
            if expression[end] == '"':
                raise BooleanSearchExpressionError("Quotes must start at the beginning of a phrase.")
            end += 1
        value = expression[index:end]
        operator = value.upper()
        tokens.append(_BooleanToken(operator if operator in {"AND", "OR", "NOT"} else "TERM", value))
        index = end
    return tokens


class _BooleanSearchParser:
    def __init__(self, tokens: List[_BooleanToken]):
        self.tokens = tokens
        self.position = 0
        self.operands = 0

    def parse(self) -> str:
        if not self.tokens:
            raise BooleanSearchExpressionError("The expression is empty.")
        result = self._parse_or()
        if self.position != len(self.tokens):
            token = self.tokens[self.position]
            raise BooleanSearchExpressionError(f"Unexpected token {token.value or token.kind!r}.")
        return result

    def _accept(self, kind: str) -> bool:
        if self.position < len(self.tokens) and self.tokens[self.position].kind == kind:
            self.position += 1
            return True
        return False

    def _parse_or(self) -> str:
        result = self._parse_and()
        while self._accept("OR"):
            result = f"({result} or {self._parse_and()})"
        return result

    def _parse_and(self) -> str:
        result = self._parse_not()
        while self._accept("AND"):
            result = f"({result} and {self._parse_not()})"
        return result

    def _parse_not(self) -> str:
        if self._accept("NOT"):
            return f"not ({self._parse_not()})"
        return self._parse_primary()

    def _parse_primary(self) -> str:
        if self._accept("LPAREN"):
            result = self._parse_or()
            if not self._accept("RPAREN"):
                raise BooleanSearchExpressionError("A parenthesized group is missing its closing parenthesis.")
            return f"({result})"
        if self.position >= len(self.tokens):
            raise BooleanSearchExpressionError("The expression ends before a search term.")

        token = self.tokens[self.position]
        if token.kind not in {"TERM", "PHRASE"}:
            raise BooleanSearchExpressionError(f"Expected a search term, found {token.value or token.kind!r}.")
        self.position += 1
        self.operands += 1
        if self.operands > 50:
            raise BooleanSearchExpressionError("The expression may contain at most 50 terms or phrases.")
        # Ingestion lower-cases the combined search_text field. Apply the same
        # normalization explicitly so matching is deterministically
        # case-insensitive rather than dependent on analyzer configuration.
        value = _escape_match_value(token.value.lower())
        if token.kind == "PHRASE":
            return f'PHRASE_MATCH(search_text, "{value}", 0)'
        return f'TEXT_MATCH(search_text, "{value}")'


def compile_boolean_search_expr(expression: str) -> str:
    """Compile plain text or Boolean syntax to a search_text-only filter.

    A plain multi-word input is treated as one exact phrase. Once explicit
    Boolean syntax or quoting is present, the full parser is used.
    """
    normalized = str(expression or "").strip()
    if len(normalized) > 1_000:
        raise BooleanSearchExpressionError("The expression must be at most 1000 characters.")
    tokens = _tokenize_boolean_search(normalized)
    if len(tokens) > 1 and all(token.kind == "TERM" for token in tokens):
        phrase = _escape_match_value(" ".join(token.value for token in tokens).lower())
        return f'PHRASE_MATCH(search_text, "{phrase}", 0)'
    return _BooleanSearchParser(tokens).parse()


def ids_to_expr(ids: List[str]) -> str:
    """Build an ID membership expression, or an expression matching all rows."""
    if not ids:
        return 'paper_uid != ""'
    escaped = [f'"{str(identifier).replace(chr(34), "")}"' for identifier in ids]
    return "paper_uid in [" + ", ".join(escaped) + "]"


def dois_to_expr(dois: List[str]) -> str:
    """Match papers whose ``doi`` or ``doi:``-prefixed ``paper_uid`` is in ``dois``.

    Callers should pass bare, casefolded DOIs to match Vitality ingestion
    (``paper_uid = doi:{doi.casefold()}``).
    """
    normalized: List[str] = []
    seen = set()
    for raw in dois:
        doi = str(raw or "").strip().casefold().replace('"', "")
        if not doi or doi in seen:
            continue
        seen.add(doi)
        normalized.append(doi)
    if not normalized:
        return 'paper_uid == ""'

    doi_values = ", ".join(f'"{doi}"' for doi in normalized)
    uid_values = ", ".join(f'"doi:{doi}"' for doi in normalized)
    return f"(doi in [{doi_values}] or paper_uid in [{uid_values}])"


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


def build_paper_query_expr(
    filters: PaperFilters,
    *,
    query_text: Optional[str] = None,
    include_query_text: bool = True,
) -> str:
    """Translate supported paper filters into a Milvus scalar expression.

    Filtering stays in Zilliz so a page request never materialises the complete
    collection in Python. ``search_query`` performs case-insensitive substring
    matching against the lower-case ``search_text`` field; the remaining
    field-specific filters retain their current Milvus ``like`` / array
    semantics.
    """
    parts = []

    def like_all(field: str, value: Optional[str]):
        if not value:
            return
        for term in (item.strip() for item in value.split(",")):
            if term:
                parts.append(f'{field} like "%{escape_like(term)}%"')

    def equals_any(field: str, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        matches = [
            f'{field} == "{escape_like(value)}"'
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

    # Each comma-separated term must occur as a contiguous substring. Ingestion
    # lower-cases search_text and combines title, abstract, authors, keywords,
    # and source, so terms can match case-insensitively across paper metadata.
    if include_query_text:
        for term in split_query_terms(query_text):
            parts.append(f'search_text like "%{escape_like(term.lower())}%"')

    like_all("title", filters.title)
    like_all("abstract", filters.abstract)
    # Source values are selected from the venue/source facet and must match the
    # complete field: filtering for "CHI" must not also include "TOCHI".
    equals_any("source", filters.source)
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
