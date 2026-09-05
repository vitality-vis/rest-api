"""Typed retrieval actions and adapters shared by low and medium search."""
from __future__ import annotations

import json
from collections import Counter
from typing import Annotated, Any

from model.paper import SearchRequest
from pydantic import BaseModel, ConfigDict, Field

from .models import (
    BM25RetrievalAction,
    ExactTermsRetrievalAction,
    MAX_CALLS_BY_TOOL,
    MAX_EXACT_TERMS,
    MAX_RETRIEVAL_CALLS,
    MAX_SEARCH_QUERY_LENGTH,
    MetadataRetrievalAction,
    RetrievalAction,
    RetrievalPlan,
    SearchIntent,
    VectorRetrievalAction,
)

MAX_EXACT_TERM_LENGTH = 200
ExactToolTerm = Annotated[
    str,
    Field(min_length=1, max_length=MAX_EXACT_TERM_LENGTH, pattern=r"^[^,]+$"),
]


class SearchBM25ToolInput(BaseModel):
    """Search for words and phrases likely to occur in relevant papers."""

    model_config = ConfigDict(title="search_bm25", extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_SEARCH_QUERY_LENGTH,
        description="A lexical paper-search query.",
    )


class SearchVectorToolInput(BaseModel):
    """Search for concepts whose wording may differ across relevant papers."""

    model_config = ConfigDict(title="search_vector", extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_SEARCH_QUERY_LENGTH,
        description="A natural-language description of the target concept.",
    )


class SearchExactTermsToolInput(BaseModel):
    """Require every supplied fixed term to occur literally in each matching paper."""

    model_config = ConfigDict(title="search_exact_terms", extra="forbid")

    terms: list[ExactToolTerm] = Field(
        min_length=1,
        max_length=MAX_EXACT_TERMS,
        description=(
            f"One to {MAX_EXACT_TERMS} literal terms using case-insensitive "
            "contiguous-substring AND semantics."
        ),
    )


class SearchMetadataToolInput(BaseModel):
    """Search using only the metadata filters already extracted by the router."""

    model_config = ConfigDict(title="search_metadata", extra="forbid")


MEDIUM_RETRIEVAL_TOOL_SCHEMAS = (
    SearchBM25ToolInput,
    SearchVectorToolInput,
    SearchExactTermsToolInput,
    SearchMetadataToolInput,
)

_TOOL_INPUT_MODELS = {
    "search_bm25": SearchBM25ToolInput,
    "search_vector": SearchVectorToolInput,
    "search_exact_terms": SearchExactTermsToolInput,
    "search_metadata": SearchMetadataToolInput,
}


class RetrievalPlanValidationError(ValueError):
    """Raised when a retrieval plan cannot be executed safely."""


def action_from_tool_call(name: str, arguments: dict[str, Any]) -> RetrievalAction:
    """Convert one validated planner-facing tool call to an internal action."""
    input_model = _TOOL_INPUT_MODELS.get(name)
    if input_model is None:
        raise RetrievalPlanValidationError(f"Unknown retrieval tool: {name}")
    parsed = input_model.model_validate(arguments)
    if name == "search_bm25":
        return BM25RetrievalAction(query=parsed.query)
    if name == "search_vector":
        return VectorRetrievalAction(query=parsed.query)
    if name == "search_exact_terms":
        return ExactTermsRetrievalAction(terms=parsed.terms)
    return MetadataRetrievalAction()


def intent_filters(intent: SearchIntent) -> dict:
    """Translate router-owned filters to the shared search-service contract."""
    return {
        "title": intent.title,
        "id_list": intent.paper_ids or None,
        "author": intent.authors or None,
        "source": intent.venues or None,
        "min_year": intent.min_year,
        "max_year": intent.max_year,
        "min_citation_counts": intent.min_citations,
    }


def has_intent_filters(intent: SearchIntent) -> bool:
    return any(value is not None and value != [] and value != "" for value in intent_filters(intent).values())


def build_low_retrieval_plan(query: str, intent: SearchIntent) -> RetrievalPlan:
    """Represent the existing low-effort policy as an executable plan."""
    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalPlanValidationError(
            "Please provide a research topic or at least one filter such as title, author, venue, year, or paper ID."
        )
    if intent.topic:
        actions: list[RetrievalAction] = [
            BM25RetrievalAction(query=normalized_query),
            VectorRetrievalAction(query=normalized_query),
        ]
    elif has_intent_filters(intent):
        actions = [MetadataRetrievalAction()]
    else:
        raise RetrievalPlanValidationError(
            "Please provide a research topic or at least one filter such as title, author, venue, year, or paper ID."
        )
    return RetrievalPlan(source="low", actions=actions, rerank_query=normalized_query)


def _normalized_action(action: RetrievalAction) -> RetrievalAction:
    if isinstance(action, BM25RetrievalAction):
        query = action.query.strip()
        if not query:
            raise RetrievalPlanValidationError("BM25 query cannot be empty.")
        return BM25RetrievalAction(query=query)
    if isinstance(action, VectorRetrievalAction):
        query = action.query.strip()
        if not query:
            raise RetrievalPlanValidationError("Vector query cannot be empty.")
        return VectorRetrievalAction(query=query)
    if isinstance(action, ExactTermsRetrievalAction):
        terms: list[str] = []
        seen: set[str] = set()
        for raw_term in action.terms:
            term = str(raw_term).strip()
            if not term:
                raise RetrievalPlanValidationError("Exact terms cannot be empty.")
            if "," in term:
                raise RetrievalPlanValidationError("Exact terms cannot contain commas.")
            if len(term) > MAX_EXACT_TERM_LENGTH:
                raise RetrievalPlanValidationError("An exact term exceeds the maximum length.")
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(term)
        return ExactTermsRetrievalAction(terms=terms)
    return MetadataRetrievalAction()


def retrieval_action_signature(action: RetrievalAction) -> tuple:
    """Return a normalized signature for duplicate detection and diagnostics."""
    normalized = _normalized_action(action)
    if isinstance(normalized, (BM25RetrievalAction, VectorRetrievalAction)):
        return normalized.tool, normalized.query.casefold()
    if isinstance(normalized, ExactTermsRetrievalAction):
        return normalized.tool, tuple(sorted(term.casefold() for term in normalized.terms))
    return (normalized.tool,)


def _action_query(action: RetrievalAction) -> str | None:
    if isinstance(action, (BM25RetrievalAction, VectorRetrievalAction)):
        return action.query
    return None


def is_primary_query_action(action: RetrievalAction, primary_query: str) -> bool:
    """True when a ranked action uses the plan's original (anchor) query."""
    query = _action_query(action)
    return query is not None and query.casefold() == primary_query.casefold()


def _fit_actions_to_budget(actions: list[RetrievalAction], *, primary_query: str) -> list[RetrievalAction]:
    """Keep primary BM25/vector arms; drop rewrite/extra arms when over budget."""
    primary_ranked = [
        action
        for action in actions
        if action.tool in {"bm25", "vector"} and is_primary_query_action(action, primary_query)
    ]
    rewrite_ranked = [
        action
        for action in actions
        if action.tool in {"bm25", "vector"} and not is_primary_query_action(action, primary_query)
    ]
    other = [action for action in actions if action.tool not in {"bm25", "vector"}]

    selected: list[RetrievalAction] = []
    counts: Counter[str] = Counter()

    def _try_add(action: RetrievalAction) -> bool:
        tool = action.tool
        if counts[tool] >= MAX_CALLS_BY_TOOL[tool]:
            return False
        if len(selected) >= MAX_RETRIEVAL_CALLS:
            return False
        selected.append(action)
        counts[tool] += 1
        return True

    for action in primary_ranked:
        if not _try_add(action):
            raise RetrievalPlanValidationError(
                "Primary query BM25/vector arms exceed the retrieval call budget."
            )
    for action in rewrite_ranked:
        _try_add(action)
    for action in other:
        _try_add(action)
    return selected


def validate_retrieval_plan(plan: RetrievalPlan, *, intent: SearchIntent) -> RetrievalPlan:
    """Normalize, de-duplicate, and enforce server-owned retrieval budgets."""
    normalized_actions: list[RetrievalAction] = []
    seen: set[tuple] = set()
    for raw_action in plan.actions:
        action = _normalized_action(raw_action)
        key = retrieval_action_signature(action)
        if key in seen:
            continue
        seen.add(key)
        normalized_actions.append(action)

    if any(action.tool != "metadata" for action in normalized_actions):
        normalized_actions = [action for action in normalized_actions if action.tool != "metadata"]
    elif not has_intent_filters(intent):
        raise RetrievalPlanValidationError("Metadata retrieval requires at least one metadata filter.")

    ranked_tools = {action.tool for action in normalized_actions if action.tool in {"bm25", "vector"}}
    if ranked_tools:
        fallback_query = plan.rerank_query.strip()
        if not fallback_query:
            raise RetrievalPlanValidationError("A topical retrieval plan requires a non-empty fallback query.")
        # Always anchor topical plans on the original query hybrid pair.
        for primary in (
            BM25RetrievalAction(query=fallback_query),
            VectorRetrievalAction(query=fallback_query),
        ):
            key = retrieval_action_signature(primary)
            if key not in seen:
                normalized_actions.append(primary)
                seen.add(key)
        normalized_actions = _fit_actions_to_budget(
            normalized_actions,
            primary_query=fallback_query,
        )

    if not normalized_actions:
        raise RetrievalPlanValidationError("The retrieval plan contains no executable actions.")
    if len(normalized_actions) > MAX_RETRIEVAL_CALLS:
        raise RetrievalPlanValidationError(f"Retrieval plan exceeds the {MAX_RETRIEVAL_CALLS}-call budget.")
    counts = Counter(action.tool for action in normalized_actions)
    for tool_name, count in counts.items():
        if count > MAX_CALLS_BY_TOOL[tool_name]:
            raise RetrievalPlanValidationError(f"Retrieval plan exceeds the {tool_name} call budget.")

    return RetrievalPlan(
        source=plan.source,
        actions=normalized_actions,
        rerank_query=plan.rerank_query.strip(),
    )


def build_search_request(action: RetrievalAction, *, intent: SearchIntent, limit: int) -> SearchRequest:
    """Adapt one validated action to the existing search service."""
    kwargs = intent_filters(intent)
    if isinstance(action, BM25RetrievalAction):
        return SearchRequest(search_query=action.query, search_mode="bm25", limit=limit, **kwargs)
    if isinstance(action, VectorRetrievalAction):
        return SearchRequest(search_query=action.query, search_mode="vector", limit=limit, **kwargs)
    if isinstance(action, ExactTermsRetrievalAction):
        expression = " AND ".join(json.dumps(term) for term in action.terms)
        return SearchRequest(search_query=expression, search_mode="bool", limit=limit, **kwargs)
    return SearchRequest(search_mode="bool", limit=limit, **kwargs)
