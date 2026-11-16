"""Codex chat runner tests with an in-process fake app-server client."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.chat.events import AgentAction, PapersResult, TextDelta
from app.chat.run_control import RunAborted, RunControl
from app.codex.runner import (
    DEVELOPER_INSTRUCTIONS,
    _conversation_locks,
    _papers_result_for_citation_backfill,
    citation_ids_from_text,
    run,
)
from app.codex.threads import _THREAD_SCHEMA_VERSION, get_thread_id, put_thread_id
from app.codex.tools import DYNAMIC_TOOLS, execute_tool


@pytest.fixture(autouse=True)
def in_memory_thread_bindings(monkeypatch):
    bindings: dict[tuple[str, str], str] = {}

    def get(user_id, chat_id):
        return bindings.get((str(user_id), str(chat_id)))

    def put(user_id, chat_id, thread_id):
        key = (str(user_id), str(chat_id))
        if key in bindings:
            return False
        bindings[key] = thread_id
        return True

    def drop(user_id, chat_id, expected_thread_id=None):
        key = (str(user_id), str(chat_id))
        current = bindings.get(key)
        if current is None or (expected_thread_id and current != expected_thread_id):
            return False
        del bindings[key]
        return True

    for module in ("app.codex.threads", "app.codex.runner"):
        monkeypatch.setattr(f"{module}.get_thread_id", get)
        monkeypatch.setattr(f"{module}.put_thread_id", put)
        monkeypatch.setattr(f"{module}.drop_thread_id", drop)
    # This test module imported these helpers directly.
    monkeypatch.setitem(globals(), "get_thread_id", get)
    monkeypatch.setitem(globals(), "put_thread_id", put)
    yield bindings


class FakeCodex:
    def __init__(self):
        self.workspace_path = "/tmp/vitality-codex"
        self.requests: list[tuple[str, dict]] = []
        self.handler = None
        self.listener = None

    async def request(self, method, params, timeout=None):
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thr-1"}}
        if method == "thread/resume":
            assert params["threadId"] == "thr-1"
            return {"thread": {"id": "thr-1"}}
        if method == "turn/start":
            await self.handler(
                {
                    "threadId": "thr-1",
                    "callId": "call-1",
                    "tool": "get_selected_paper_evidence",
                    "arguments": {"detail_level": "auto"},
                }
            )
            self.listener(
                {
                    "method": "item/agentMessage/delta",
                    "params": {"threadId": "thr-1", "delta": "Grounded answer"},
                }
            )
            self.listener(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-1", "status": "completed"}},
                }
            )
            return {"turn": {"id": "turn-1", "status": "inProgress"}}
        raise AssertionError(method)

    def listen(self, thread_id, callback):
        assert thread_id == "thr-1"
        self.listener = callback

    def handle_tool(self, thread_id, handler):
        assert thread_id == "thr-1"
        self.handler = handler

    def clear_thread(self, thread_id):
        assert thread_id == "thr-1"


def test_research_instructions_favor_natural_curated_answers():
    instructions = DEVELOPER_INSTRUCTIONS

    assert "Match the user's language" in instructions
    assert "Lead with the useful answer" in instructions
    assert "Do not narrate routine searches" in instructions
    assert "natural prose by default" in instructions
    assert "Do not force a paper-by-paper template" in instructions
    assert "prioritize the strongest recommendations" in instructions
    assert "weak, mixed, indirect" in instructions
    assert "Ask a follow-up only when" in instructions
    assert "private working evidence" in instructions
    assert "UI only shows papers after `present_papers`" in instructions
    assert "Reformulate, broaden, or narrow queries" in instructions
    assert "Write the final answer yourself" in instructions
    summarize_skill = (
        Path(__file__).parents[1]
        / "app/codex/agent/skills/summarize-selected-papers/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "adaptively" in summarize_skill
    assert "one section per paper" in summarize_skill
    assert "Do not pad with methods/findings/limitations templates" in summarize_skill
    assert "search_openalex" in instructions
    assert "ordered best-first" in instructions
    assert "after each useful search" in instructions
    assert "later calls append" in instructions or "append/dedupe" in instructions
    assert "must also appear in the accumulated presented list" in instructions
    assert "Do not present every hit by default" not in instructions
    assert "Cap is 20 IDs per call" not in instructions
    assert "claim they were added to the library" in instructions
    assert "VitaLITy corpus coverage" in instructions
    assert "`UIST`" in instructions
    assert "`CSCW`" in instructions
    assert "[[ID:paper_id]]" in instructions


def test_search_tool_describes_results_as_private_working_evidence():
    search_tool = next(tool for tool in DYNAMIC_TOOLS if tool["name"] == "search_papers")
    description = search_tool["description"]

    assert "private working evidence" in description
    assert "not shown as papers" in description
    assert "use present_papers" in description
    assert "displays the returned papers" not in description


def test_codex_thread_schema_invalidates_old_research_instructions():
    assert _THREAD_SCHEMA_VERSION == "research-v16"


def test_turn_text_bounds_large_selection_ids():
    from app.codex.runner import _turn_text

    request = SimpleNamespace(
        text="Summarize these papers",
        selected_paper_ids=[f"private-paper-{index}" for index in range(500)],
        user_id="user-1",
        project_id="project-1",
    )
    rendered = _turn_text(request, history=[])

    assert "Current paper selection: 500 unique papers" in rendered
    assert "private-paper-0" not in rendered
    assert "inspect_selected_papers" in rendered
    assert "Preloaded selected-paper evidence" not in rendered


def test_turn_text_preloads_small_selection_evidence(monkeypatch):
    from app.codex.runner import _turn_text

    def fake_evidence(**kwargs):
        assert kwargs["request"].detail_level == "abstracts"
        return {
            "selected_count": 2,
            "returned_count": 2,
            "papers": [
                {"paper_id": "paper-1", "title": "One", "abstract": "A"},
                {"paper_id": "paper-2", "title": "Two", "abstract": "B"},
            ],
        }

    monkeypatch.setattr(
        "service.selected_paper_tool.get_selected_paper_evidence", fake_evidence
    )
    request = SimpleNamespace(
        text="帮我总结一下这两篇论文。",
        selected_paper_ids=["paper-1", "paper-2"],
        selected_paper_groups={"library": ["paper-1", "paper-2"]},
        user_id="user-1",
        project_id="project-1",
    )
    rendered = _turn_text(request, history=[])

    assert "Preloaded selected-paper evidence" in rendered
    assert "answer directly" in rendered
    assert "paper-1" in rendered
    assert '"abstract": "A"' in rendered
    assert "do not call inspect_selected_papers" in rendered


def test_codex_runner_returns_selected_paper_evidence(monkeypatch):
    seen: dict[str, object] = {}

    def fake_evidence(**kwargs):
        seen.update(kwargs)
        return {
            "selected_count": 1,
            "returned_count": 1,
            "detailed_count": 1,
            "papers": [{"paper_id": "paper-1", "title": "Paper", "abstract": "Evidence"}],
        }

    monkeypatch.setattr(
        "service.selected_paper_tool.get_selected_paper_evidence", fake_evidence
    )
    request = SimpleNamespace(
        text="What method is used?",
        history=[{"role": "user", "content": "Earlier question"}],
        selected_paper_ids=["paper-1"],
        user_id="user-1",
        project_id="project-1",
        effort="low",
        control=None,
    )

    fake = FakeCodex()
    events = asyncio.run(collect(run(request, client=fake)))

    assert seen["user_id"] == "user-1"
    assert seen["project_id"] == "project-1"
    assert seen["selected_paper_ids"] == ["paper-1"]
    thread_params = next(params for method, params in fake.requests if method == "thread/start")
    turn_params = next(params for method, params in fake.requests if method == "turn/start")
    assert thread_params["ephemeral"] is True
    assert thread_params["sandbox"] == "read-only"
    assert turn_params["sandboxPolicy"] == {"type": "readOnly", "networkAccess": True}
    assert "Earlier question" in turn_params["input"][0]["text"]
    assert any(isinstance(event, TextDelta) and "Grounded answer" in event.text for event in events)
    assert any(
        isinstance(event, AgentAction) and event.action == "get.selected_paper_evidence"
        for event in events
    )


def test_conversation_reuses_one_codex_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "service.selected_paper_tool.get_selected_paper_evidence",
        lambda **_kwargs: {"papers": [], "returned_count": 0, "detailed_count": 0},
    )
    request = SimpleNamespace(
        text="What about the follow-up?",
        history=[{"role": "user", "content": "Earlier question"}],
        selected_paper_ids=[],
        user_id="user-1",
        chat_id="conv-1",
        effort="low",
        control=None,
    )
    fake = FakeCodex()

    async def both():
        await collect(run(request, client=fake))
        await collect(run(request, client=fake))

    asyncio.run(both())

    starts = [params for method, params in fake.requests if method == "thread/start"]
    resumes = [params for method, params in fake.requests if method == "thread/resume"]
    turns = [params for method, params in fake.requests if method == "turn/start"]
    assert len(starts) == 1
    assert starts[0]["ephemeral"] is False
    assert resumes == [{"threadId": "thr-1"}]
    assert "Earlier question" in turns[0]["input"][0]["text"]
    assert "Earlier question" not in turns[1]["input"][0]["text"]
    assert "What about the follow-up?" in turns[1]["input"][0]["text"]


def test_turn_start_failure_does_not_publish_thread_mapping(tmp_path, monkeypatch):

    class FailingCodex(FakeCodex):
        async def request(self, method, params, timeout=None):
            self.requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thr-new"}}
            if method == "turn/start":
                raise RuntimeError("turn rejected")
            raise AssertionError(method)

        def listen(self, thread_id, callback):
            self.listener = callback

        def handle_tool(self, thread_id, handler):
            self.handler = handler

        def clear_thread(self, thread_id):
            pass

    request = SimpleNamespace(
        text="Hello", history=[], selected_paper_ids=[], user_id="user-failure",
        chat_id="conv-failure", effort="low", control=None, has_codex_thread=False,
    )

    with pytest.raises(RuntimeError, match="turn rejected"):
        asyncio.run(collect(run(request, client=FailingCodex())))

    assert get_thread_id("user-failure", "conv-failure") is None


def test_failed_completion_does_not_publish_thread_mapping(tmp_path, monkeypatch):

    class FailedCompletionCodex(FakeCodex):
        async def request(self, method, params, timeout=None):
            self.requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thr-failed"}}
            if method == "turn/start":
                self.listener({
                    "method": "turn/completed",
                    "params": {"threadId": "thr-failed", "turn": {
                        "id": "turn-failed", "status": "failed"}},
                })
                return {"turn": {"id": "turn-failed"}}
            raise AssertionError(method)

        def listen(self, thread_id, callback):
            self.listener = callback

        def handle_tool(self, thread_id, handler):
            self.handler = handler

        def clear_thread(self, thread_id):
            pass

    request = SimpleNamespace(
        text="Hello", history=[], selected_paper_ids=[], user_id="user-failed",
        chat_id="conv-failed", effort="low", control=None, has_codex_thread=False,
    )

    with pytest.raises(RuntimeError, match="could not complete"):
        asyncio.run(collect(run(request, client=FailedCompletionCodex())))

    assert get_thread_id("user-failed", "conv-failed") is None


def test_failed_resumed_turn_drops_existing_mapping(tmp_path, monkeypatch):
    put_thread_id("user-resumed", "conv-resumed", "thr-resumed")

    class FailedResumedCodex(FakeCodex):
        async def request(self, method, params, timeout=None):
            self.requests.append((method, params))
            if method == "thread/resume":
                assert params == {"threadId": "thr-resumed"}
                return {"thread": {"id": "thr-resumed"}}
            if method == "turn/start":
                self.listener({
                    "method": "turn/completed",
                    "params": {"threadId": "thr-resumed", "turn": {
                        "id": "turn-resumed", "status": "failed"}},
                })
                return {"turn": {"id": "turn-resumed"}}
            raise AssertionError(method)

        def listen(self, thread_id, callback):
            assert thread_id == "thr-resumed"
            self.listener = callback

        def handle_tool(self, thread_id, handler):
            assert thread_id == "thr-resumed"
            self.handler = handler

        def clear_thread(self, thread_id):
            assert thread_id == "thr-resumed"

    request = SimpleNamespace(
        text="Follow up", history=[], selected_paper_ids=[], user_id="user-resumed",
        chat_id="conv-resumed", effort="low", control=None, has_codex_thread=True,
    )

    with pytest.raises(RuntimeError, match="could not complete"):
        asyncio.run(collect(run(request, client=FailedResumedCodex())))

    assert get_thread_id("user-resumed", "conv-resumed") is None


def test_generator_close_does_not_publish_thread_mapping(tmp_path, monkeypatch):

    class StreamingCodex(FakeCodex):
        async def request(self, method, params, timeout=None):
            self.requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thr-streaming"}}
            if method == "turn/start":
                self.listener({
                    "method": "item/agentMessage/delta",
                    "params": {"threadId": "thr-streaming", "delta": "partial"},
                })
                return {"turn": {"id": "turn-streaming"}}
            raise AssertionError(method)

        def listen(self, thread_id, callback):
            self.listener = callback

        def handle_tool(self, thread_id, handler):
            self.handler = handler

        def clear_thread(self, thread_id):
            pass

    request = SimpleNamespace(
        text="Hello", history=[], selected_paper_ids=[], user_id="user-close",
        chat_id="conv-close", effort="low", control=None, has_codex_thread=False,
    )

    async def consume_one_then_close():
        stream = run(request, client=StreamingCodex())
        assert isinstance(await anext(stream), TextDelta)
        await stream.aclose()

    asyncio.run(consume_one_then_close())

    assert get_thread_id("user-close", "conv-close") is None
    assert _conversation_locks == {}


def test_waiting_turn_aborts_without_waiting_for_holder(tmp_path, monkeypatch):

    class HoldingCodex:
        workspace_path = "/tmp/vitality-codex"

        def __init__(self):
            self.holder_started = threading.Event()
            self.release_holder = threading.Event()
            self.listener = None

        async def request(self, method, params, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr-holder"}}
            if method == "turn/start":
                self.holder_started.set()
                await asyncio.to_thread(self.release_holder.wait, 5)
                self.listener({
                    "method": "turn/completed",
                    "params": {"threadId": "thr-holder", "turn": {
                        "id": "turn-holder", "status": "completed"}},
                })
                return {"turn": {"id": "turn-holder"}}
            raise AssertionError(method)

        def listen(self, _thread_id, callback):
            self.listener = callback

        def handle_tool(self, _thread_id, _handler):
            pass

        def clear_thread(self, _thread_id):
            pass

    fake = HoldingCodex()
    holder_request = SimpleNamespace(
        text="holder", history=[], selected_paper_ids=[], user_id="user-abort",
        chat_id="conv-abort", effort="low", control=None, has_codex_thread=False,
    )
    waiter_control = RunControl(queue_wait_timeout_s=10, agent_timeout_s=10)
    waiter_request = SimpleNamespace(
        text="waiter", history=[], selected_paper_ids=[], user_id="user-abort",
        chat_id="conv-abort", effort="low", control=waiter_control,
        has_codex_thread=False,
    )
    errors: list[BaseException] = []

    def invoke(request):
        try:
            asyncio.run(collect(run(request, client=fake)))
        except BaseException as error:  # noqa: BLE001 - asserted below
            errors.append(error)

    holder = threading.Thread(target=invoke, args=(holder_request,))
    waiter = threading.Thread(target=invoke, args=(waiter_request,))
    holder.start()
    assert fake.holder_started.wait(2)
    waiter.start()
    time.sleep(0.1)
    waiter_control.cancel()
    waiter.join(2)

    assert not waiter.is_alive()
    assert holder.is_alive()
    assert any(isinstance(error, RunAborted) for error in errors)

    fake.release_holder.set()
    holder.join(5)
    assert not holder.is_alive()
    assert _conversation_locks == {}


def test_same_conversation_turns_serialize_across_event_loop_threads(
    tmp_path, monkeypatch
):

    class BlockingCodex:
        workspace_path = "/tmp/vitality-codex"

        def __init__(self):
            self.guard = threading.Lock()
            self.requests: list[tuple[str, dict]] = []
            self.listeners = {}
            self.handlers = {}
            self.first_turn_started = threading.Event()
            self.release_first = threading.Event()
            self.second_turn_started = threading.Event()
            self.turn_count = 0

        async def request(self, method, params, timeout=None):
            with self.guard:
                self.requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thr-shared"}}
            if method == "thread/resume":
                return {"thread": {"id": "thr-shared"}}
            if method == "turn/start":
                with self.guard:
                    self.turn_count += 1
                    turn_number = self.turn_count
                if turn_number == 1:
                    self.first_turn_started.set()
                    await asyncio.to_thread(self.release_first.wait, 5)
                else:
                    self.second_turn_started.set()
                self.listeners["thr-shared"]({
                    "method": "turn/completed",
                    "params": {"threadId": "thr-shared", "turn": {
                        "id": f"turn-{turn_number}", "status": "completed"}},
                })
                return {"turn": {"id": f"turn-{turn_number}"}}
            raise AssertionError(method)

        def listen(self, thread_id, callback):
            self.listeners[thread_id] = callback

        def handle_tool(self, thread_id, handler):
            self.handlers[thread_id] = handler

        def clear_thread(self, thread_id):
            self.listeners.pop(thread_id, None)
            self.handlers.pop(thread_id, None)

    fake = BlockingCodex()
    request = SimpleNamespace(
        text="Hello", history=[], selected_paper_ids=[], user_id="user-concurrent",
        chat_id="conv-concurrent", effort="low", control=None, has_codex_thread=False,
    )
    errors: list[BaseException] = []

    def invoke():
        try:
            asyncio.run(collect(run(request, client=fake)))
        except BaseException as error:  # noqa: BLE001 - asserted in main thread
            errors.append(error)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert fake.first_turn_started.wait(2)
    second.start()
    time.sleep(0.1)
    assert not fake.second_turn_started.is_set()
    fake.release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert fake.second_turn_started.is_set()
    assert [method for method, _ in fake.requests].count("thread/start") == 1
    assert [method for method, _ in fake.requests].count("thread/resume") == 1
    assert _conversation_locks == {}


def test_search_tool_emits_ranked_papers(monkeypatch):
    class Result:
        papers = [{"ID": "paper-1", "Title": "One", "Year": 2024, "Source": "CHI"}]

    monkeypatch.setattr("service.search.search", lambda request: Result())
    emitted: list[object] = []

    result = execute_tool(
        user_id=None,
        selected_paper_ids=[],
        effort="low",
        call_id="search-1",
        tool="search_papers",
        arguments={"query": "literature review", "mode": "bm25", "limit": 5},
        emit=emitted.append,
    )

    actions = [event for event in emitted if isinstance(event, AgentAction)]
    assert not any(isinstance(event, PapersResult) for event in emitted)
    assert actions[0].status == "started"
    assert actions[0].data == {
        "query": "literature review",
        "mode": "bm25",
    }
    assert actions[-1].status == "completed"
    assert actions[-1].data == {"resultCount": 1}
    assert result["success"] is True

    presented: list[object] = []
    candidates = {
        "paper-1": {"paper_id": "paper-1"},
        "paper-2": {"paper_id": "paper-2"},
    }
    presented_ids: list[str] = []
    present_result = execute_tool(
        user_id=None,
        selected_paper_ids=[],
        effort="low",
        call_id="present-1",
        tool="present_papers",
        arguments={"paper_ids": ["paper-1"]},
        emit=presented.append,
        candidates=candidates,
        presented_paper_ids=presented_ids,
    )
    second_result = execute_tool(
        user_id=None,
        selected_paper_ids=[],
        effort="low",
        call_id="present-2",
        tool="present_papers",
        arguments={"paper_ids": ["paper-2"]},
        emit=presented.append,
        candidates=candidates,
        presented_paper_ids=presented_ids,
    )
    papers = [event for event in presented if isinstance(event, PapersResult)]
    assert papers[0].ids == ["paper-1"]
    assert papers[1].ids == ["paper-1", "paper-2"]
    assert papers[1].policy == "agent_curated"
    assert present_result["success"] is True
    assert second_result["success"] is True
    present_tool = next(tool for tool in DYNAMIC_TOOLS if tool["name"] == "present_papers")
    assert present_tool["inputSchema"]["properties"]["paper_ids"]["maxItems"] == 100
    assert "accumulate" in present_tool["description"]


def test_citation_backfill_appends_cited_candidates_missing_from_present():
    assert citation_ids_from_text(
        "See [[ID:a]] and [[ID:b]] then [[ID:a]] again."
    ) == ["a", "b"]

    presented: list[str] = ["a"]
    candidates = {
        "a": {"paper_id": "a"},
        "b": {"paper_id": "b"},
        "c": {"paper_id": "c"},
    }
    result = _papers_result_for_citation_backfill(
        "I recommend [[ID:a]], [[ID:b]], and unknown [[ID:z]].",
        candidates=candidates,
        presented_paper_ids=presented,
        effort="medium",
    )
    assert result is not None
    assert result.ids == ["a", "b"]
    assert presented == ["a", "b"]
    assert result.effort == "medium"

    assert (
        _papers_result_for_citation_backfill(
            "Only [[ID:a]] again.",
            candidates=candidates,
            presented_paper_ids=presented,
            effort="low",
        )
        is None
    )


def test_citation_backfill_presents_cited_papers_when_agent_skipped_present():
    presented: list[str] = []
    candidates = {"p1": {"paper_id": "p1"}, "p2": {"paper_id": "p2"}}
    result = _papers_result_for_citation_backfill(
        "Top picks: [[ID:p2]] then [[ID:p1]].",
        candidates=candidates,
        presented_paper_ids=presented,
        effort="low",
    )
    assert result is not None
    assert result.ids == ["p2", "p1"]
    assert presented == ["p2", "p1"]


def test_search_tool_overfetches_then_excludes_and_truncates(monkeypatch):
    class Result:
        papers = [
            {"ID": "selected-1", "Title": "Selected"},
            *[
                {"ID": f"related-{index}", "Title": f"Related {index}"}
                for index in range(5)
            ],
        ]

    seen: dict[str, object] = {}

    def fake_search(request):
        seen["limit"] = request.limit
        return Result()

    monkeypatch.setattr("service.search.search", fake_search)
    candidates: dict[str, dict[str, object]] = {}
    result = execute_tool(
        user_id="user-1",
        selected_paper_ids=["selected-1"],
        effort="low",
        call_id="search-overfetch",
        tool="search_papers",
        arguments={"query": "related work", "limit": 2},
        emit=lambda _event: None,
        candidates=candidates,
    )

    payload = json.loads(result["contentItems"][0]["text"])
    assert seen["limit"] == 6
    assert [paper["paper_id"] for paper in payload["papers"]] == [
        "related-0",
        "related-1",
    ]
    assert set(candidates) == {"related-0", "related-1"}


def test_find_similar_tool_samples_seeds_and_excludes_selection(monkeypatch):
    class Result:
        papers = [
            {"ID": "seed-1", "Title": "Selected"},
            {"ID": "related-1", "Title": "Related"},
        ]

    seen: dict[str, object] = {}

    def fake_similar(seeds, limit):
        seen["seed_ids"] = seeds
        seen["limit"] = limit
        return Result()

    monkeypatch.setattr("app.codex.tools._run_similar_search", fake_similar)
    monkeypatch.setattr(
        "app.codex.tools._resolve_selected_for_tool", lambda **_kwargs: []
    )
    monkeypatch.setattr("app.codex.tools._is_search_unavailable", lambda _error: False)
    monkeypatch.setattr(
        "service.selected_paper_tool.select_representative_paper_ids",
        lambda ids, **_kwargs: (ids[:20], "deterministic_map_coverage"),
    )
    candidates: dict[str, dict[str, object]] = {}
    result = execute_tool(
        user_id="user-1",
        selected_paper_ids=[f"seed-{index}" for index in range(25)],
        selected_paper_groups={"clicked": [f"seed-{index}" for index in range(25)]},
        effort="low",
        call_id="similar-1",
        tool="find_similar_papers",
        arguments={"limit": 20},
        emit=lambda _event: None,
        candidates=candidates,
    )

    payload = json.loads(result["contentItems"][0]["text"])
    assert len(seen["seed_ids"]) == 20
    assert seen["limit"] == 60
    assert payload["used_seed_count"] == 20
    assert [paper["paper_id"] for paper in payload["papers"]] == ["related-1"]
    assert set(candidates) == {"related-1"}


def test_find_similar_tool_requires_trusted_selection(monkeypatch):
    monkeypatch.setattr(
        "app.codex.tools._run_similar_search",
        lambda *_args: pytest.fail("search must not run"),
    )
    result = execute_tool(
        user_id="user-1",
        selected_paper_ids=[],
        effort="low",
        call_id="similar-empty",
        tool="find_similar_papers",
        arguments={},
        emit=lambda _event: None,
    )
    assert result["success"] is False
    assert "Select or pin" in result["contentItems"][0]["text"]


def test_selected_citations_add_corpus_and_openalex_candidates(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.codex.tools._resolve_selected_for_tool",
        lambda **_kwargs: [
            SimpleNamespace(paper_id="seed-1", metadata={"doi": "10.test/seed"})
        ],
    )
    monkeypatch.setattr(
        "app.codex.tools._fetch_citations",
        lambda *_args, **_kwargs: {
            "references": {
                "papers": [
                    {"paper_id": "related-1", "in_corpus": True, "title": "Corpus"},
                    {
                        "paper_id": "openalex:W99",
                        "in_corpus": False,
                        "title": "External",
                        "doi": "10.ext/1",
                        "openalex_id": "W99",
                    },
                ]
            },
            "cited_by": {"papers": []},
        },
    )
    candidates: dict[str, dict[str, object]] = {}
    result = execute_tool(
        user_id="user-1",
        selected_paper_ids=["seed-1"],
        effort="low",
        call_id="citations-1",
        tool="get_selected_paper_citations",
        arguments={"paper_ids": ["seed-1"], "direction": "references"},
        emit=lambda _event: None,
        candidates=candidates,
    )
    payload = json.loads(result["contentItems"][0]["text"])
    assert result["success"] is True
    assert len(payload["sources"][0]["references"]) == 2
    assert set(candidates) == {"related-1", "openalex:W99"}


def test_search_openalex_prefers_corpus_ids_and_keeps_relevance_order(monkeypatch):
    monkeypatch.setattr(
        "service.openalex_papers.search_openalex_papers",
        lambda **_kwargs: {
            "query": "steerable chatbots",
            "venue": "UIST",
            "venue_source_id": "S1",
            "papers": [
                {
                    "paper_id": "corpus-1",
                    "in_corpus": True,
                    "title": "In corpus",
                    "openalex_id": "W1",
                },
                {
                    "paper_id": "openalex:W2",
                    "in_corpus": False,
                    "title": "External only",
                    "openalex_id": "W2",
                },
            ],
        },
    )
    candidates: dict[str, dict[str, object]] = {}
    result = execute_tool(
        user_id="user-1",
        selected_paper_ids=[],
        effort="low",
        call_id="openalex-1",
        tool="search_openalex",
        arguments={"query": "steerable chatbots", "venue": "UIST", "year_min": 2026},
        emit=lambda _event: None,
        candidates=candidates,
    )
    payload = json.loads(result["contentItems"][0]["text"])
    assert result["success"] is True
    assert [paper["paper_id"] for paper in payload["papers"]] == [
        "corpus-1",
        "openalex:W2",
    ]
    assert set(candidates) == {"corpus-1", "openalex:W2"}
    assert candidates["corpus-1"]["in_corpus"] is True


def test_reasoning_lifecycle_emits_activity_without_private_content():
    from app.codex.runner import _event_from_notification

    events = [
        _event_from_notification({
            "method": method,
            "params": {
                "threadId": "thr-1",
                "turnId": "turn-1",
                "item": {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": ["Private reasoning summary"],
                    "content": ["Private reasoning content"],
                },
            },
        })
        for method in ("item/started", "item/completed")
    ]
    assert events == [
        AgentAction(action_id="thinking-reasoning-1", action="thinking", status=status)
        for status in ("started", "completed")
    ]
    for method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
        assert _event_from_notification({
            "method": method,
            "params": {"delta": "Private reasoning content"},
        }) is None


def test_web_search_activity_shows_the_query_without_results():
    from app.codex.runner import _event_from_notification

    completed = _event_from_notification({
        "method": "item/completed",
        "params": {
            "item": {
                "type": "webSearch",
                "id": "search-1",
                "query": "survey papers",
                "results": [{"url": "https://example.com", "title": "hidden"}],
            }
        },
    })
    assert completed == AgentAction(
        action_id="web-search-search-1",
        action="web.search",
        status="completed",
        data={"query": "survey papers"},
    )
    started = _event_from_notification({
        "method": "item/started",
        "params": {
            "item": {
                "type": "webSearch",
                "id": "search-2",
                "action": {"type": "search", "query": "openalex doi"},
            }
        },
    })
    assert started == AgentAction(
        action_id="web-search-search-2",
        action="web.search",
        status="started",
        data={"query": "openalex doi"},
    )
    assert _event_from_notification({
        "method": "item/started",
        "params": {"item": {"type": "webSearch"}},
    }) is None


def test_non_reasoning_and_malformed_items_do_not_emit_thinking():
    from app.codex.runner import _event_from_notification

    for item in (None, {}, {"type": "reasoning"}, {"type": "agentMessage", "id": "a"}):
        assert _event_from_notification({
            "method": "item/started", "params": {"item": item},
        }) is None


async def collect(events):
    return [event async for event in events]
