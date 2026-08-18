"""Unit tests for search-v2 primary-query anchoring and fusion weights."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.agent_v2.models import (
    BM25RetrievalAction,
    ExactTermsRetrievalAction,
    RetrievalPlan,
    SearchIntent,
    VectorRetrievalAction,
)
from agents.agent_v2.search_executor import (
    PRIMARY_CANDIDATE_LIMIT,
    PRIMARY_ARM_WEIGHT,
    REWRITE_CANDIDATE_LIMIT,
    REWRITE_ARM_WEIGHT,
    RRF_K,
    _ActionResult,
    _candidate_limit_for_action,
    _merge_results,
    _rrf_arm_weight,
)
from agents.agent_v2.search_tools import validate_retrieval_plan
from collections import Counter


def _paper(paper_id: str) -> dict:
    return {"ID": paper_id, "Title": paper_id}


def test_validate_injects_primary_query_arms_and_prefers_them_under_budget():
    intent = SearchIntent(topic="privacy cameras")
    plan = RetrievalPlan(
        source="medium",
        rerank_query="privacy cameras",
        actions=[
            BM25RetrievalAction(query="rewrite one"),
            BM25RetrievalAction(query="rewrite two"),
            BM25RetrievalAction(query="rewrite three"),
            VectorRetrievalAction(query="vector rewrite"),
        ],
    )
    validated = validate_retrieval_plan(plan, intent=intent)
    queries_by_tool = {
        "bm25": [action.query for action in validated.actions if action.tool == "bm25"],
        "vector": [action.query for action in validated.actions if action.tool == "vector"],
    }
    assert "privacy cameras" in queries_by_tool["bm25"]
    assert "privacy cameras" in queries_by_tool["vector"]
    assert len(queries_by_tool["bm25"]) == 3  # primary kept; one rewrite dropped
    assert queries_by_tool["bm25"][0] == "privacy cameras"


def test_low_plan_unchanged_shape():
    intent = SearchIntent(topic="privacy cameras")
    plan = RetrievalPlan(
        source="low",
        rerank_query="privacy cameras",
        actions=[
            BM25RetrievalAction(query="privacy cameras"),
            VectorRetrievalAction(query="privacy cameras"),
        ],
    )
    validated = validate_retrieval_plan(plan, intent=intent)
    assert [(action.tool, action.query) for action in validated.actions] == [
        ("bm25", "privacy cameras"),
        ("vector", "privacy cameras"),
    ]


def test_shared_plan_budget_and_candidate_depth_remain_legacy_compatible():
    primary = BM25RetrievalAction(query="privacy cameras")
    rewrite = VectorRetrievalAction(query="camera privacy practices")
    assert PRIMARY_CANDIDATE_LIMIT == 50
    assert REWRITE_CANDIDATE_LIMIT == 50
    assert _candidate_limit_for_action(primary, primary_query="privacy cameras") == 50
    assert _candidate_limit_for_action(rewrite, primary_query="privacy cameras") == 50
    assert _candidate_limit_for_action(
        rewrite,
        primary_query="privacy cameras",
        ranked_candidate_limit=100,
    ) == 100

    with pytest.raises(ValidationError):
        RetrievalPlan(
            source="medium",
            rerank_query="privacy cameras",
            actions=[BM25RetrievalAction(query=f"rewrite {index}") for index in range(9)],
        )


def test_rrf_primary_not_diluted_by_rewrite_family():
    primary_query = "privacy cameras"
    rewrite_counts = Counter(bm25=2)
    primary = BM25RetrievalAction(query=primary_query)
    rewrite = BM25RetrievalAction(query="other wording")
    assert _rrf_arm_weight(
        primary, primary_query=primary_query, rewrite_family_counts=rewrite_counts
    ) == PRIMARY_ARM_WEIGHT
    assert _rrf_arm_weight(
        rewrite, primary_query=primary_query, rewrite_family_counts=rewrite_counts
    ) == REWRITE_ARM_WEIGHT / 2


def test_merge_prefers_primary_hit_over_rewrite_only_hit():
    primary_query = "privacy cameras"
    results = [
        _ActionResult(
            source="bm25:1",
            action=BM25RetrievalAction(query=primary_query),
            papers=[_paper("primary-hit")],
        ),
        _ActionResult(
            source="bm25:2",
            action=BM25RetrievalAction(query="rewrite"),
            papers=[_paper("rewrite-hit")],
        ),
        _ActionResult(
            source="vector",
            action=VectorRetrievalAction(query=primary_query),
            papers=[_paper("primary-hit")],
        ),
    ]
    merged = _merge_results(results, primary_query=primary_query)
    assert [paper_id(item.paper) for item in merged] == ["primary-hit", "rewrite-hit"]

    # Primary paper gets two full primary votes; rewrite gets one down-weighted vote.
    primary_score = merged[0].rrf_score
    rewrite_score = merged[1].rrf_score
    expected_primary = 2 * (PRIMARY_ARM_WEIGHT / (RRF_K + 1))
    expected_rewrite = REWRITE_ARM_WEIGHT / (RRF_K + 1)
    assert primary_score == expected_primary
    assert rewrite_score == expected_rewrite


def paper_id(paper: dict) -> str:
    return str(paper["ID"])


def test_exact_terms_still_tie_breaks_without_rrf_when_unranked_only():
    results = [
        _ActionResult(
            source="exact_terms",
            action=ExactTermsRetrievalAction(terms=["camera"]),
            papers=[_paper("a"), _paper("b")],
        )
    ]
    merged = _merge_results(results, primary_query="unused")
    assert [item.paper["ID"] for item in merged] == ["a", "b"]
    assert all(item.exact_match for item in merged)
