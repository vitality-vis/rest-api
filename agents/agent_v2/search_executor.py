"""Low-effort paper-search execution: retrieval, fusion, and reranking."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from model.paper import SearchRequest
from service.search import SearchUnavailableError, search

from .logging import SearchV2Trace
from .models import SearchIntent, SearchV2Paper, SearchV2Request, SearchV2Response
from .reranker import paper_id, rerank


RRF_K = 60
CANDIDATE_LIMIT = 50
FUSED_CANDIDATE_LIMIT = 100
# Server-side experiment switch. Keep false until benchmark evidence supports enabling it.
DEFAULT_ENABLE_CROSS_ENCODER = False


class SearchCriteriaRequiredError(ValueError):
    """Raised when a request has neither a research topic nor usable filters."""


def _filters(intent: SearchIntent) -> dict:
    return {
        "title": intent.title,
        "id_list": intent.paper_ids or None,
        "author": intent.authors or None,
        "source": intent.venues or None,
        "min_year": intent.min_year,
        "max_year": intent.max_year,
        "min_citation_counts": intent.min_citations,
    }


def _has_filters(intent: SearchIntent) -> bool:
    return any(value is not None and value != [] and value != "" for value in _filters(intent).values())


def _policy(intent: SearchIntent) -> str:
    if not intent.topic and not _has_filters(intent):
        raise SearchCriteriaRequiredError(
            "Please provide a research topic or at least one filter such as title, author, venue, year, or paper ID."
        )
    if not intent.topic:
        return "filter"
    return "hybrid"


def _hybrid(query: str, intent: SearchIntent) -> tuple[list[SearchV2Paper], dict]:
    kwargs = _filters(intent)
    requests = {
        "bm25": SearchRequest(search_query=query, search_mode="bm25", limit=CANDIDATE_LIMIT, **kwargs),
        "vector": SearchRequest(search_query=query, search_mode="vector", limit=CANDIDATE_LIMIT, **kwargs),
    }
    results: dict[str, object] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(search, value) for name, value in requests.items()}
        for source, future in futures.items():
            try:
                results[source] = future.result()
            except SearchUnavailableError as error:
                failures[source] = str(error)
    if not results:
        raise SearchUnavailableError("Both hybrid retrieval arms failed.")

    merged: dict[str, SearchV2Paper] = {}
    for source, result in results.items():
        for rank, paper in enumerate(result.papers, start=1):
            identifier = paper_id(paper)
            if not identifier:
                continue
            item = merged.get(identifier)
            if item is None:
                item = SearchV2Paper(paper=paper)
                merged[identifier] = item
            item.retrieval_sources.append(source)
            item.retrieval_ranks[source] = rank
            item.rrf_score = (item.rrf_score or 0) + 1.0 / (RRF_K + rank)

    candidates = sorted(merged.values(), key=lambda item: (-(item.rrf_score or 0), paper_id(item.paper)))[:FUSED_CANDIDATE_LIMIT]
    return candidates, {"retrieval_failures": failures, "retrieval_counts": {name: len(value.papers) for name, value in results.items()}}


def _filter(intent: SearchIntent) -> tuple[list[SearchV2Paper], dict]:
    result = search(SearchRequest(search_mode="exact", limit=CANDIDATE_LIMIT, **_filters(intent)))
    candidates = [SearchV2Paper(paper=paper, retrieval_sources=["filter"], retrieval_ranks={"filter": rank}) for rank, paper in enumerate(result.papers, start=1)]
    return candidates, {}


def run_search(
    request: SearchV2Request,
    *,
    intent: SearchIntent,
    enable_cross_encoder: bool | None = None,
    trace: SearchV2Trace | None = None,
) -> SearchV2Response:
    """Execute one already-routed paper search."""
    query = request.query.strip()
    if not query:
        raise SearchCriteriaRequiredError(
            "Please provide a research topic or at least one filter such as title, author, venue, year, or paper ID."
        )
    policy = _policy(intent)
    candidates, diagnostics = _filter(intent) if policy == "filter" else _hybrid(query, intent)
    use_cross_encoder = DEFAULT_ENABLE_CROSS_ENCODER if enable_cross_encoder is None else enable_cross_encoder
    rerank_status = "skipped"
    if use_cross_encoder and candidates:
        try:
            scores = {paper_id(paper): score for paper, score in rerank(query, [item.paper for item in candidates])}
            candidates.sort(key=lambda item: (-scores[paper_id(item.paper)], paper_id(item.paper)))
            for item in candidates:
                item.rerank_score = scores[paper_id(item.paper)]
            rerank_status = "complete"
        except Exception as error:
            rerank_status = "failed"
            diagnostics["rerank_error"] = str(error)
    diagnostics.update({"retrieved": len(candidates), "reranked": len(candidates) if rerank_status == "complete" else 0, "rerank_status": rerank_status})
    return SearchV2Response(query=query, effort=request.effort, intent=intent, policy=policy, papers=candidates[:request.result_limit], status="partial" if diagnostics.get("retrieval_failures") or rerank_status == "failed" else "complete", diagnostics=diagnostics)
