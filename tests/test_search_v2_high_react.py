"""Unit tests for bounded high-effort ReAct retrieval."""
from __future__ import annotations

import json
import asyncio
import threading
from types import SimpleNamespace

from agents.agent_v2.high_react import (
    HIGH_ARM_CANDIDATE_LIMIT,
    HIGH_CONTEXT_ITEMS_PER_GROUP,
    HighActionSpec,
    HighExclusion,
    HighSearchFacet,
    HighSearchGroup,
    HighTaskPlan,
    _ActionResult,
    _HighGraphRunner,
    _balanced_pending_ids,
    _bounded_memory,
    _coverage,
    _ensure_facet_group_coverage,
    _expand_group_actions,
    _final_title_rerank,
    _group_aware_results,
    _normalize_plan_group_ids,
    run_high_search,
)
from agents.agent_v2.search_executor import PRIMARY_CANDIDATE_LIMIT, REWRITE_CANDIDATE_LIMIT
from agents.agent_v2.models import (
    BM25RetrievalAction,
    HighTitleRerankConfig,
    SearchIntent,
    SearchV2Request,
    SearchV2Paper,
    SearchV2Response,
    RouteDecision,
    V2ChatRequest,
    VectorRetrievalAction,
)


def _paper(identifier: str, title: str) -> dict:
    return {"ID": identifier, "Title": title, "Abstract": f"Abstract for {title}"}


class _ScriptedLLM:
    def __init__(self):
        self.screen_calls = 0
        self.evaluate_calls = 0
        self.matching_policies = []

    def invoke(self, messages):
        prompt = str(messages[0].content)
        if "planning node" in prompt:
            payload = {
                "summary": "LLM support for literature review without citation recommendation",
                "matching_mode": "broad",
                "matching_rationale": "The request asks for exploratory discovery.",
                "groups": [{"id": "review", "topic": "LLM support for literature review"}],
                "exclusions": [{"text": "citation recommendation", "mode": "semantic"}],
                "actions": [
                    {"group_id": "review", "tool": "bm25", "query": "LLM literature review", "terms": []},
                    {"group_id": "review", "tool": "vector", "query": "language models supporting literature review workflows", "terms": []},
                ],
            }
        elif "relevance-scoring node" in prompt:
            self.screen_calls += 1
            request = json.loads(messages[1].content)
            self.matching_policies.append(request["matching_policy"])
            rows = []
            for paper in request["papers"]:
                excluded = "Citation recommender" in paper["text"]
                rows.append(
                    {
                        "index": paper["index"],
                        "relevant_groups": [] if excluded else ["review"],
                        "excluded": excluded,
                        "score": 0.0 if excluded else 0.9,
                        "reason": "excluded direction" if excluded else "relevant",
                    }
                )
            payload = {"assessments": rows}
        elif "evaluation node" in prompt:
            self.evaluate_calls += 1
            payload = {
                "decision": "search_more",
                "diagnosis": "Only one candidate remains after exclusion screening.",
                "actions": [
                    {
                        "group_id": "review",
                        "tool": "vector",
                        "query": "AI support for evidence synthesis and systematic review workflows",
                        "terms": [],
                    }
                ],
            }
        else:
            raise AssertionError(f"Unexpected prompt: {prompt[:80]}")
        return SimpleNamespace(content=json.dumps(payload))


def test_normalize_plan_group_ids_truncates_deduplicates_and_remaps_actions():
    first = "interactive_3d_unsteady_flow_perspectives_primary"
    second = "interactive_3d_unsteady_flow_perspectives_secondary"
    normalized = _normalize_plan_group_ids(
        {
            "groups": [{"id": first, "topic": "one"}, {"id": second, "topic": "two"}],
            "actions": [
                {"group_id": first, "tool": "bm25", "query": "one", "terms": []},
                {"group_id": second, "tool": "vector", "query": "two", "terms": []},
            ],
        }
    )

    group_ids = [group["id"] for group in normalized["groups"]]
    assert len(set(group_ids)) == 2
    assert all(len(group_id) <= 40 for group_id in group_ids)
    assert [action["group_id"] for action in normalized["actions"]] == group_ids


def test_expand_group_actions_builds_five_bm25_vector_pairs_per_group():
    task = HighTaskPlan(
        summary="paired rewrites",
        groups=[HighSearchGroup(id="g", topic="interactive volume visualization")],
        actions=[
            HighActionSpec(group_id="g", tool="bm25", query="volume visualization interaction"),
            HighActionSpec(
                group_id="g",
                tool="vector",
                query="Methods for interactively exploring volumetric data",
            ),
        ],
    )

    expanded = _expand_group_actions(task, task.actions)

    assert [action.tool for action in expanded] == ["bm25"] * 5 + ["vector"] * 5
    assert len({(action.tool, action.query) for action in expanded}) == 10
    assert expanded[0].query == "volume visualization interaction"
    assert expanded[5].query == "Methods for interactively exploring volumetric data"


def test_ranked_retrieval_arms_fetch_top_100():
    assert PRIMARY_CANDIDATE_LIMIT == 50
    assert REWRITE_CANDIDATE_LIMIT == 50
    assert HIGH_ARM_CANDIDATE_LIMIT == 100


def test_high_search_loops_after_exclusion_screening(monkeypatch):
    llm = _ScriptedLLM()
    initial = [
        _paper("excluded", "Citation recommender for scholarly search"),
        _paper("kept-1", "Language models for literature reviews"),
    ]
    follow_up = [
        _paper("kept-2", "AI-assisted evidence synthesis"),
        _paper("kept-3", "Automating systematic review workflows"),
    ]

    def fake_execute(plan, intent, *, source_prefix=None, ranked_candidate_limit=None):
        assert ranked_candidate_limit == HIGH_ARM_CANDIDATE_LIMIT
        is_follow_up = any("evidence synthesis" in getattr(action, "query", "") for action in plan.actions)
        papers = follow_up if is_follow_up else initial
        results = [
            _ActionResult(source=f"{source_prefix}.arm", action=action, papers=papers)
            for action in plan.actions
        ]
        return results, {"retrieval_failures": {}}

    monkeypatch.setattr("agents.agent_v2.high_react._execute_actions", fake_execute)
    response = run_high_search(
        SearchV2Request(query="LLM literature review", effort="high", result_limit=3),
        user_request="Find LLM literature review papers but exclude citation recommendation",
        intent=SearchIntent(topic="LLM literature review"),
        llm=llm,
    )

    assert response.papers[0].paper["ID"] == "kept-1"
    assert {item.paper["ID"] for item in response.papers} == {"kept-1", "kept-2", "kept-3"}
    assert response.diagnostics["rounds"] == 2
    assert response.diagnostics["tool_calls_used"] == 20
    assert response.diagnostics["stop_reason"] == "sufficient_scored_coverage"
    assert response.diagnostics["matching_mode"] == "broad"
    assert response.diagnostics["llm_score_policy"] == "soft_ranking_signal"
    assert response.diagnostics["ranking_policy"] == "explicit_rrf_group_llm_direct_fusion"
    assert {item["mode"] for item in llm.matching_policies} == {"broad"}
    assert llm.evaluate_calls == 1


def test_high_action_validation_deduplicates_and_rejects_negative_queries():
    runner = _HighGraphRunner(llm=object(), trace=None)
    task = HighTaskPlan(
        summary="test",
        groups=[HighSearchGroup(id="g", topic="visual analytics")],
        exclusions=[HighExclusion(text="COVID dashboards")],
        actions=[HighActionSpec(group_id="g", tool="bm25", query="visual analytics")],
    )
    state = {
        "task": task,
        "pending_actions": [
            HighActionSpec(group_id="g", tool="bm25", query="visual analytics"),
            HighActionSpec(group_id="g", tool="vector", query="visual analytics NOT COVID"),
            HighActionSpec(group_id="g", tool="vector", query="healthcare rather than COVID"),
            HighActionSpec(group_id="missing", tool="vector", query="healthcare analytics"),
        ],
        "attempted_signatures": ["('bm25', 'visual analytics')"],
        "tool_counts": {},
        "tool_calls_used": 1,
    }
    assert runner._validated_pending_actions(state) == []


def test_high_budget_allows_final_bm25_and_vector_calls():
    runner = _HighGraphRunner(llm=object(), trace=None)
    task = HighTaskPlan(
        summary="use full positive-query budget",
        groups=[HighSearchGroup(id="g", topic="topic")],
        actions=[HighActionSpec(group_id="g", tool="bm25", query="initial")],
    )
    state = {
        "task": task,
        "pending_actions": [
            HighActionSpec(group_id="g", tool="bm25", query="eighth lexical query"),
            HighActionSpec(group_id="g", tool="vector", query="eighth semantic query"),
        ],
        "attempted_signatures": [],
        "tool_counts": {"bm25": 7, "vector": 7},
        "tool_calls_used": 14,
    }
    selected = runner._validated_pending_actions(state)
    assert [action.tool for action in selected] == ["bm25", "vector"]


def test_group_aware_results_interleave_independent_directions():
    g1_papers = [_paper(f"g1-{index}", f"GNN paper {index}") for index in range(3)]
    g2_papers = [_paper(f"g2-{index}", f"VR paper {index}") for index in range(2)]
    results = [
        _ActionResult("g1", BM25RetrievalAction(query="GNN molecules"), g1_papers),
        _ActionResult("g2", VectorRetrievalAction(query="VR architecture"), g2_papers),
    ]
    assessments = {
        paper["ID"]: {"relevant_groups": ["gnn"], "excluded": False, "score": 0.9}
        for paper in g1_papers
    }
    assessments.update(
        {
            paper["ID"]: {"relevant_groups": ["vr"], "excluded": False, "score": 0.9}
            for paper in g2_papers
        }
    )
    state = {
        "task": HighTaskPlan(
            summary="two directions",
            groups=[
                HighSearchGroup(id="gnn", topic="GNN molecules"),
                HighSearchGroup(id="vr", topic="VR architecture"),
            ],
            actions=[HighActionSpec(group_id="gnn", tool="bm25", query="GNN molecules")],
        ),
        "action_results": results,
        "assessments": assessments,
    }
    selected = _group_aware_results(state, query="two directions", limit=4)
    assert [item.paper["ID"] for item in selected] == ["g1-0", "g2-0", "g1-1", "g2-1"]


def test_direct_topic_signal_can_promote_direct_match_without_filtering_tail():
    component = _paper("component", "Gaussian splatting scene editing")
    direct = _paper("direct", "Natural language interaction for volume visualization")
    state = {
        "task": HighTaskPlan(
            summary="direct topic ranking",
            groups=[HighSearchGroup(id="g", topic="volume visualization interaction")],
            actions=[HighActionSpec(group_id="g", tool="bm25", query="volume visualization")],
        ),
        "action_results": [
            _ActionResult(
                "bm25",
                BM25RetrievalAction(query="volume visualization"),
                [component, direct],
            )
        ],
        "candidate_group_hints": {"component": ["g"], "direct": ["g"]},
        "assessments": {},
    }

    selected = _group_aware_results(
        state,
        query="natural language interaction for volume visualization",
        limit=2,
    )

    assert [item.paper["ID"] for item in selected] == ["direct", "component"]
    assert set(selected[0].ranking_signals) == {
        "rrf",
        "llm",
        "direct_topic",
        "group_coverage",
        "final",
    }


def test_final_title_rerank_is_soft_and_keeps_every_candidate(monkeypatch):
    papers = [
        SearchV2Paper(paper=_paper("broad", "Broad component"), rerank_score=0.9),
        SearchV2Paper(paper=_paper("direct", "Direct topic"), rerank_score=0.6),
    ]
    monkeypatch.setattr(
        "agents.agent_v2.high_react.score_batch",
        lambda query, batch, **kwargs: {"broad": 0.0, "direct": 1.0},
    )

    reranked, diagnostics = _final_title_rerank(
        papers,
        query="direct topic",
        llm=object(),
        config=HighTitleRerankConfig(enabled=True, weight=0.5),
    )

    assert [item.paper["ID"] for item in reranked] == ["direct", "broad"]
    assert {item.paper["ID"] for item in reranked} == {"broad", "direct"}
    assert diagnostics["policy"] == "soft_signal_no_filter"
    assert reranked[0].ranking_signals["title"] == 1.0


def test_llm_title_scorer_supports_raw_paper_uid_records():
    from agents.agent_v2.llm_reranker import score_batch

    class Scorer:
        def invoke(self, messages):
            return SimpleNamespace(content='{"scores":[{"index":1,"score":0.8}]}')

    assert score_batch(
        "direct topic",
        [{"paper_uid": "paper-1", "title": "Direct topic"}],
        screening_fields="title",
        llm=Scorer(),
    ) == {"paper-1": 0.8}


def test_facet_validator_splits_broad_single_group_and_seeds_every_group():
    task = HighTaskPlan(
        summary="multi-facet broad query",
        matching_mode="broad",
        facets=[
            HighSearchFacet(id="context", topic="contextual expert knowledge"),
            HighSearchFacet(id="quality", topic="data quality and uncertainty"),
            HighSearchFacet(id="sensemaking", topic="collaborative sensemaking"),
            HighSearchFacet(id="decisions", topic="analysis decision making"),
        ],
        groups=[
            HighSearchGroup(
                id="all",
                topic="expert knowledge in data analysis",
                facet_ids=["context", "quality", "sensemaking", "decisions"],
            )
        ],
        actions=[HighActionSpec(group_id="all", tool="bm25", query="expert knowledge data analysis")],
    )

    validated = _ensure_facet_group_coverage(task)
    expanded = _expand_group_actions(validated, validated.actions)

    assert len(validated.groups) == 2
    assert {facet.id for facet in validated.facets} == {
        facet_id for group in validated.groups for facet_id in group.facet_ids
    }
    assert len(expanded) == 20
    assert all(sum(action.group_id == group.id for action in expanded) == 10 for group in validated.groups)


class _RetryPlannerLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(content='{"summary":"truncated"')
        return SimpleNamespace(
            content=json.dumps(
                {
                    "summary": "repaired plan",
                    "matching_mode": "broad",
                    "matching_rationale": "exploratory request",
                    "facets": [{"id": "interaction", "topic": "interactive visualization"}],
                    "groups": [
                        {
                            "id": "interaction",
                            "topic": "interactive visualization",
                            "facet_ids": ["interaction"],
                        }
                    ],
                    "exclusions": [],
                    "actions": [
                        {
                            "group_id": "interaction",
                            "tool": "bm25",
                            "query": "interactive visualization",
                            "terms": [],
                        }
                    ],
                }
            )
        )


def test_planner_retries_once_after_malformed_json():
    llm = _RetryPlannerLLM()
    update = _HighGraphRunner(llm=llm, trace=None).plan(
        {
            "user_request": "Find interactive visualization research",
            "retrieval_query": "interactive visualization",
            "intent": SearchIntent(topic="interactive visualization"),
            "action_results": [],
            "attempted_signatures": [],
            "assessments": {},
            "replan_count": 0,
            "round_number": 0,
            "tool_calls_used": 0,
        }
    )

    assert llm.calls == 2
    assert len(update["task"].actions) == 10
    assert update["planner_attempts"] == 2
    assert update["planner_repaired"] is True


def test_planner_uses_strict_json_schema_when_provider_supports_it():
    calls = {}

    class Runnable:
        def invoke(self, messages):
            return HighTaskPlan(
                summary="strict plan",
                facets=[HighSearchFacet(id="topic", topic="visual analytics")],
                groups=[HighSearchGroup(id="topic", topic="visual analytics", facet_ids=["topic"])],
                actions=[HighActionSpec(group_id="topic", tool="bm25", query="visual analytics")],
            )

    class StructuredLLM:
        def with_structured_output(self, schema, **kwargs):
            calls["schema"] = schema
            calls.update(kwargs)
            return Runnable()

    update = _HighGraphRunner(llm=StructuredLLM(), trace=None).plan(
        {
            "user_request": "Find visual analytics research",
            "retrieval_query": "visual analytics",
            "intent": SearchIntent(topic="visual analytics"),
            "action_results": [],
            "attempted_signatures": [],
            "assessments": {},
            "replan_count": 0,
            "round_number": 0,
            "tool_calls_used": 0,
        }
    )

    assert calls["schema"]["title"] == "HighTaskPlan"
    assert calls["schema"]["additionalProperties"] is False
    assert calls["method"] == "json_schema"
    assert calls["include_raw"] is False
    assert calls["strict"] is True
    assert len(update["task"].actions) == 10


def test_low_scored_and_unscreened_candidates_remain_in_final_results():
    papers = [_paper(f"p-{index:03d}", f"Paper {index}") for index in range(3)]
    state = {
        "task": HighTaskPlan(
            summary="soft scores",
            groups=[HighSearchGroup(id="g", topic="topic")],
            actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
        ),
        "action_results": [_ActionResult("bm25", BM25RetrievalAction(query="topic"), papers)],
        "candidate_group_hints": {paper["ID"]: ["g"] for paper in papers},
        "assessments": {
            papers[0]["ID"]: {
                "relevant_groups": [],
                "excluded": False,
                "score": 0.1,
            },
            papers[2]["ID"]: {
                "relevant_groups": ["g"],
                "excluded": False,
                "score": 0.9,
            },
        },
    }
    selected = _group_aware_results(state, query="topic", limit=3)
    assert {item.paper["ID"] for item in selected} == {paper["ID"] for paper in papers}
    assert selected[0].paper["ID"] == papers[2]["ID"]


def test_explicitly_excluded_candidate_is_still_hard_filtered():
    papers = [_paper("kept", "Relevant paper"), _paper("excluded", "Excluded paper")]
    state = {
        "task": HighTaskPlan(
            summary="explicit exclusion",
            groups=[HighSearchGroup(id="g", topic="topic")],
            exclusions=[HighExclusion(text="excluded direction")],
            actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
        ),
        "action_results": [_ActionResult("bm25", BM25RetrievalAction(query="topic"), papers)],
        "candidate_group_hints": {paper["ID"]: ["g"] for paper in papers},
        "assessments": {
            "kept": {"relevant_groups": ["g"], "excluded": False, "score": 0.4},
            "excluded": {"relevant_groups": [], "excluded": True, "score": 0.9},
        },
    }
    selected = _group_aware_results(state, query="topic", limit=2)
    assert [item.paper["ID"] for item in selected] == ["kept"]


def test_retrieve_keeps_every_result_from_top_100_arm(monkeypatch):
    papers = [_paper(f"p-{index:03d}", f"Paper {index}") for index in range(100)]

    def fake_execute(plan, intent, *, source_prefix=None, ranked_candidate_limit=None):
        assert ranked_candidate_limit == HIGH_ARM_CANDIDATE_LIMIT
        return [
            _ActionResult(
                source=f"{source_prefix}.bm25",
                action=plan.actions[0],
                papers=papers,
            )
        ], {"retrieval_failures": {}}

    monkeypatch.setattr("agents.agent_v2.high_react._execute_actions", fake_execute)
    task = HighTaskPlan(
        summary="no per-arm truncation",
        groups=[HighSearchGroup(id="g", topic="topic")],
        actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
    )
    update = _HighGraphRunner(llm=object(), trace=None).retrieve(
        {
            "task": task,
            "intent": SearchIntent(topic="topic"),
            "pending_actions": task.actions,
            "action_results": [],
            "candidate_group_hints": {},
            "assessments": {},
            "attempted_signatures": [],
            "tool_counts": {},
            "tool_calls_used": 0,
            "round_number": 0,
            "retrieval_failures": {},
            "rescreen_all": False,
        }
    )
    assert len(update["action_results"][0].papers) == 100
    assert len(update["candidate_group_hints"]) == 100


def test_retrieve_executes_groups_concurrently(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_execute(plan, intent, *, source_prefix=None, ranked_candidate_limit=None):
        assert ranked_candidate_limit == HIGH_ARM_CANDIDATE_LIMIT
        barrier.wait(timeout=1)
        return [
            _ActionResult(
                source=f"{source_prefix}.bm25",
                action=plan.actions[0],
                papers=[_paper(source_prefix, source_prefix)],
            )
        ], {"retrieval_failures": {}}

    monkeypatch.setattr("agents.agent_v2.high_react._execute_actions", fake_execute)
    task = HighTaskPlan(
        summary="parallel groups",
        groups=[HighSearchGroup(id="g1", topic="one"), HighSearchGroup(id="g2", topic="two")],
        actions=[
            HighActionSpec(group_id="g1", tool="bm25", query="one"),
            HighActionSpec(group_id="g2", tool="bm25", query="two"),
        ],
    )
    update = _HighGraphRunner(llm=object(), trace=None).retrieve(
        {
            "task": task,
            "intent": SearchIntent(topic="parallel"),
            "pending_actions": task.actions,
            "action_results": [],
            "candidate_group_hints": {},
            "assessments": {},
            "attempted_signatures": [],
            "tool_counts": {},
            "tool_calls_used": 0,
            "round_number": 0,
            "retrieval_failures": {},
            "rescreen_all": False,
        }
    )

    assert len(update["action_results"]) == 2


def test_final_fusion_keeps_candidates_beyond_old_global_240_limit():
    papers = [_paper(f"p-{index:03d}", f"Paper {index}") for index in range(300)]
    state = {
        "task": HighTaskPlan(
            summary="no global candidate truncation",
            groups=[HighSearchGroup(id="g", topic="topic")],
            actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
        ),
        "action_results": [_ActionResult("bm25", BM25RetrievalAction(query="topic"), papers)],
        "candidate_group_hints": {paper["ID"]: ["g"] for paper in papers},
        "assessments": {},
    }
    selected = _group_aware_results(state, query="topic", limit=300)
    assert len(selected) == 300
    assert selected[-1].paper["ID"] == "p-299"


def test_context_memory_is_bounded_per_group():
    papers = [_paper(f"p-{index}", f"Paper {index}") for index in range(20)]
    state = {
        "task": HighTaskPlan(
            summary="bounded context",
            groups=[HighSearchGroup(id="g", topic="topic")],
            actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
        ),
        "result_limit": 100,
        "action_results": [_ActionResult("bm25", BM25RetrievalAction(query="topic"), papers)],
        "assessments": {
            paper["ID"]: {"relevant_groups": ["g"], "excluded": False, "score": 0.8}
            for paper in papers
        },
    }
    memory = _bounded_memory(state)
    assert len(memory["accepted_examples"]["g"]) == HIGH_CONTEXT_ITEMS_PER_GROUP


def test_matching_mode_changes_coverage_threshold_not_final_inclusion():
    paper = _paper("partial", "Useful partial match")
    base = {
        "result_limit": 10,
        "action_results": [_ActionResult("bm25", BM25RetrievalAction(query="topic"), [paper])],
        "candidate_group_hints": {"partial": ["g"]},
        "assessments": {
            "partial": {"relevant_groups": ["g"], "excluded": False, "score": 0.5}
        },
    }
    broad_task = HighTaskPlan(
        summary="broad discovery",
        matching_mode="broad",
        groups=[HighSearchGroup(id="g", topic="topic")],
        actions=[HighActionSpec(group_id="g", tool="bm25", query="topic")],
    )
    strict_task = broad_task.model_copy(update={"matching_mode": "strict"})

    assert _coverage({**base, "task": broad_task}) == {"g": 1}
    assert _coverage({**base, "task": strict_task}) == {"g": 0}
    assert [
        item.paper["ID"]
        for item in _group_aware_results({**base, "task": strict_task}, query="topic", limit=10)
    ] == ["partial"]


def test_screening_candidates_are_balanced_across_groups():
    candidates = {
        **{f"g1-{index}": _paper(f"g1-{index}", "GNN") for index in range(10)},
        **{f"g2-{index}": _paper(f"g2-{index}", "VR") for index in range(3)},
    }
    hints = {
        identifier: ["g1" if identifier.startswith("g1") else "g2"]
        for identifier in candidates
    }
    selected = _balanced_pending_ids(
        candidates,
        hints=hints,
        group_ids=["g1", "g2"],
        already_screened=set(),
        limit=6,
    )
    assert selected == ["g1-0", "g2-0", "g1-1", "g2-1", "g1-2", "g2-2"]


def test_chat_runner_dispatches_high_effort_to_react_graph(monkeypatch):
    from agents.agent_v2 import runner

    called = {}

    def fake_high(request, *, user_request, intent, llm, trace, title_rerank):
        called["effort"] = request.effort
        called["user_request"] = user_request
        return SearchV2Response(
            query=request.query,
            effort="high",
            intent=intent,
            policy="hybrid",
            papers=[SearchV2Paper(paper=_paper("high-paper", "High result"))],
        )

    monkeypatch.setattr(
        runner,
        "route",
        lambda request, trace: RouteDecision(
            route="search",
            response_mode="papers",
            search_intent=SearchIntent(topic="high effort retrieval"),
        ),
    )
    monkeypatch.setattr(runner, "get_llm", lambda model=None: object())
    monkeypatch.setattr("agents.agent_v2.high_react.run_high_search", fake_high)

    async def collect():
        return "".join(
            [
                chunk
                async for chunk in runner.run(
                    V2ChatRequest(text="Find high effort papers", effort="high")
                )
            ]
        )

    output = asyncio.run(collect())
    assert called == {"effort": "high", "user_request": "Find high effort papers"}
    assert "high-paper" in output


def test_chat_runner_default_effort_stays_on_low_search(monkeypatch):
    from agents.agent_v2 import runner

    called = {}

    monkeypatch.setattr(
        runner,
        "route",
        lambda request, trace: RouteDecision(
            route="search",
            response_mode="papers",
            search_intent=SearchIntent(topic="default low retrieval"),
        ),
    )

    def fake_search(request, **kwargs):
        called["effort"] = request.effort
        return SearchV2Response(
            query=request.query,
            effort=request.effort,
            intent=kwargs["intent"],
            policy="hybrid",
            papers=[SearchV2Paper(paper=_paper("low-paper", "Low result"))],
        )

    monkeypatch.setattr(runner, "run_search", fake_search)

    async def collect():
        return "".join([chunk async for chunk in runner.run(V2ChatRequest(text="Find papers"))])

    output = asyncio.run(collect())
    assert called == {"effort": "low"}
    assert "low-paper" in output
