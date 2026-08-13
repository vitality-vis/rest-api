"""Shared retrieval-plan execution, fusion, and reranking for search v2."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from service.search import SearchUnavailableError, search

from .logging import SearchV2Trace
from .models import (
    RetrievalAction,
    RetrievalPlan,
    SearchIntent,
    SearchV2Paper,
    SearchV2Request,
    SearchV2Response,
)
from .reranker import paper_id, rerank
from .search_tools import (
    RetrievalPlanValidationError,
    build_low_retrieval_plan,
    build_search_request,
    validate_retrieval_plan,
)


RRF_K = 60
CANDIDATE_LIMIT = 50
FUSED_CANDIDATE_LIMIT = 100
# Server-side experiment switch. Keep false until benchmark evidence supports enabling it.
DEFAULT_ENABLE_CROSS_ENCODER = False
# DEFAULT_ENABLE_CROSS_ENCODER = True


class SearchCriteriaRequiredError(ValueError):
    """Raised when a request has neither a research topic nor usable filters."""


@dataclass(frozen=True)
class _ActionResult:
    source: str
    action: RetrievalAction
    papers: list[dict]

    @property
    def ranked(self) -> bool:
        return self.action.tool in {"bm25", "vector"}


def _source_names(plan: RetrievalPlan) -> list[str]:
    """Keep low provenance stable while making repeated tools unambiguous."""
    totals = Counter(action.tool for action in plan.actions)
    seen: Counter[str] = Counter()
    names: list[str] = []
    for action in plan.actions:
        seen[action.tool] += 1
        if totals[action.tool] == 1:
            names.append("filter" if action.tool == "metadata" else action.tool)
        else:
            names.append(f"{action.tool}:{seen[action.tool]}")
    return names


def _execute_actions(plan: RetrievalPlan, intent: SearchIntent) -> tuple[list[_ActionResult], dict]:
    source_names = _source_names(plan)
    requests = [build_search_request(action, intent=intent, limit=CANDIDATE_LIMIT) for action in plan.actions]
    results: list[_ActionResult] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(search, request) for request in requests]
        for source, action, future in zip(source_names, plan.actions, futures):
            try:
                result = future.result()
                results.append(_ActionResult(source=source, action=action, papers=result.papers))
            except SearchUnavailableError as error:
                failures[source] = str(error)
    if not results:
        raise SearchUnavailableError("All retrieval actions failed.")
    diagnostics = {
        "retrieval_failures": failures,
        "retrieval_counts": {result.source: len(result.papers) for result in results},
    }
    return results, diagnostics


def _get_or_create(merged: dict[str, SearchV2Paper], paper: dict) -> SearchV2Paper | None:
    identifier = paper_id(paper)
    if not identifier:
        return None
    item = merged.get(identifier)
    if item is None:
        item = SearchV2Paper(paper=paper)
        merged[identifier] = item
    return item


def _merge_results(results: list[_ActionResult]) -> list[SearchV2Paper]:
    ranked_results = [result for result in results if result.ranked]
    unranked_results = [result for result in results if not result.ranked]
    merged: dict[str, SearchV2Paper] = {}

    if ranked_results:
        successful_calls_by_family = Counter(result.action.tool for result in ranked_results)
        for result in ranked_results:
            family_normalizer = float(successful_calls_by_family[result.action.tool])
            for rank, paper in enumerate(result.papers, start=1):
                item = _get_or_create(merged, paper)
                if item is None:
                    continue
                item.retrieval_sources.append(result.source)
                item.retrieval_ranks[result.source] = rank
                item.rrf_score = (item.rrf_score or 0) + 1.0 / family_normalizer / (RRF_K + rank)

        for result in unranked_results:
            for position, paper in enumerate(result.papers, start=1):
                item = _get_or_create(merged, paper)
                if item is None:
                    continue
                item.retrieval_sources.append(result.source)
                item.retrieval_ranks[result.source] = position
                if result.action.tool == "exact_terms":
                    item.exact_match = True

        return sorted(
            merged.values(),
            key=lambda item: (-(item.rrf_score or 0), -int(item.exact_match), paper_id(item.paper)),
        )[:FUSED_CANDIDATE_LIMIT]

    for result in unranked_results:
        for position, paper in enumerate(result.papers, start=1):
            item = _get_or_create(merged, paper)
            if item is None:
                continue
            if result.source not in item.retrieval_sources:
                item.retrieval_sources.append(result.source)
            item.retrieval_ranks[result.source] = position
            if result.action.tool == "exact_terms":
                item.exact_match = True
    return list(merged.values())[:FUSED_CANDIDATE_LIMIT]


def execute_retrieval_plan(
    plan: RetrievalPlan,
    *,
    request: SearchV2Request,
    intent: SearchIntent,
    enable_cross_encoder: bool | None = None,
    trace: SearchV2Trace | None = None,
) -> SearchV2Response:
    """Validate and execute a low- or medium-generated retrieval plan."""
    validated_plan = validate_retrieval_plan(plan, intent=intent)
    try:
        action_results, diagnostics = _execute_actions(validated_plan, intent)
    except SearchUnavailableError as error:
        if trace is not None:
            trace.log_retrieval_execution(
                plan=validated_plan,
                retrieval_counts={},
                retrieval_failures={"all": str(error)},
                rerank_status="skipped",
                status="failed",
            )
        raise
    candidates = _merge_results(action_results)
    use_cross_encoder = DEFAULT_ENABLE_CROSS_ENCODER if enable_cross_encoder is None else enable_cross_encoder
    rerank_status = "skipped"
    if use_cross_encoder and candidates:
        try:
            scores = {
                paper_id(paper): score
                for paper, score in rerank(validated_plan.rerank_query, [item.paper for item in candidates])
            }
            candidates.sort(key=lambda item: (-scores[paper_id(item.paper)], paper_id(item.paper)))
            for item in candidates:
                item.rerank_score = scores[paper_id(item.paper)]
            rerank_status = "complete"
        except Exception as error:
            rerank_status = "failed"
            diagnostics["rerank_error"] = str(error)

    diagnostics.update(
        {
            "plan_source": validated_plan.source,
            "retrieved": len(candidates),
            "reranked": len(candidates) if rerank_status == "complete" else 0,
            "rerank_status": rerank_status,
        }
    )
    policy = "hybrid" if any(result.ranked for result in action_results) else "filter"
    status = "partial" if diagnostics.get("retrieval_failures") or rerank_status == "failed" else "complete"
    if trace is not None:
        trace.log_retrieval_execution(
            plan=validated_plan,
            retrieval_counts=diagnostics["retrieval_counts"],
            retrieval_failures=diagnostics["retrieval_failures"],
            rerank_status=rerank_status,
            status=status,
        )
    return SearchV2Response(
        query=request.query.strip(),
        effort=request.effort,
        intent=intent,
        policy=policy,
        papers=candidates[:request.result_limit],
        status=status,
        diagnostics=diagnostics,
    )


def run_search(
    request: SearchV2Request,
    *,
    intent: SearchIntent,
    plan: RetrievalPlan | None = None,
    medium_fallback_reason: str | None = None,
    enable_cross_encoder: bool | None = None,
    trace: SearchV2Trace | None = None,
) -> SearchV2Response:
    """Execute a supplied medium plan or the deterministic low plan."""
    requested_plan_source = (
        plan.source
        if plan is not None
        else "medium" if medium_fallback_reason else "low"
    )
    fallback_reason = medium_fallback_reason
    try:
        selected_plan = plan or build_low_retrieval_plan(request.query, intent)
    except RetrievalPlanValidationError as error:
        raise SearchCriteriaRequiredError(str(error)) from error

    if fallback_reason and trace is not None:
        trace.log_retrieval_fallback(
            requested_plan_source="medium",
            executed_plan_source="low",
            reason=fallback_reason,
        )

    try:
        response = execute_retrieval_plan(
            selected_plan,
            request=request,
            intent=intent,
            enable_cross_encoder=enable_cross_encoder,
            trace=trace,
        )
    except (RetrievalPlanValidationError, SearchUnavailableError) as error:
        if selected_plan.source != "medium":
            if isinstance(error, RetrievalPlanValidationError):
                raise SearchCriteriaRequiredError(str(error)) from error
            raise
        fallback_reason = (
            "medium_plan_validation_failed"
            if isinstance(error, RetrievalPlanValidationError)
            else "all_medium_actions_failed"
        )
        if trace is not None:
            trace.log_retrieval_fallback(
                requested_plan_source="medium",
                executed_plan_source="low",
                reason=fallback_reason,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        try:
            low_plan = build_low_retrieval_plan(request.query, intent)
            response = execute_retrieval_plan(
                low_plan,
                request=request,
                intent=intent,
                enable_cross_encoder=enable_cross_encoder,
                trace=trace,
            )
        except RetrievalPlanValidationError as low_error:
            raise SearchCriteriaRequiredError(str(low_error)) from low_error

    response.diagnostics["requested_plan_source"] = requested_plan_source
    response.diagnostics["executed_plan_source"] = response.diagnostics.get("plan_source")
    if fallback_reason:
        response.diagnostics["fallback_reason"] = fallback_reason
    return response
