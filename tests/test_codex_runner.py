"""Codex chat runner tests with an in-process fake app-server client."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from app.chat.events import AgentAction, PapersResult, TextDelta
from app.chat.run_control import RunAborted, RunControl
from app.codex.runner import DEVELOPER_INSTRUCTIONS, _conversation_locks, run
from app.codex.threads import _THREAD_SCHEMA_VERSION, get_thread_id, put_thread_id
from app.codex.tools import DYNAMIC_TOOLS, execute_tool
from service.selected_paper_tool import SelectedPaperAnswer


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
                    "tool": "answer_selected_papers",
                    "arguments": {"question": "What method is used?"},
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
    assert "only shows papers after you call present_papers" in instructions
    assert "Reformulate, broaden, or narrow queries" in instructions
    assert "preserve every [[ID:...]] citation exactly" in instructions


def test_search_tool_describes_results_as_private_working_evidence():
    search_tool = next(tool for tool in DYNAMIC_TOOLS if tool["name"] == "search_papers")
    description = search_tool["description"]

    assert "private working evidence" in description
    assert "not shown as papers" in description
    assert "use present_papers" in description
    assert "displays the returned papers" not in description


def test_codex_thread_schema_invalidates_old_research_instructions():
    assert _THREAD_SCHEMA_VERSION == "research-v3"


def test_codex_runner_relays_selected_paper_answer(monkeypatch):
    seen: dict[str, object] = {}

    def fake_answer(**kwargs):
        seen.update(kwargs)
        return SelectedPaperAnswer(
            answer="It uses interviews [[ID:paper-1]].",
            evidence="full_text",
            paper_ids=["paper-1"],
            full_text_paper_ids=["paper-1"],
        )

    monkeypatch.setattr(
        "service.selected_paper_tool.answer_selected_papers", fake_answer
    )
    request = SimpleNamespace(
        text="What method is used?",
        history=[{"role": "user", "content": "Earlier question"}],
        selected_paper_ids=["paper-1"],
        user_id="user-1",
        effort="low",
        control=None,
    )

    fake = FakeCodex()
    events = asyncio.run(collect(run(request, client=fake)))

    assert seen["user_id"] == "user-1"
    assert seen["paper_ids"] == ["paper-1"]
    thread_params = next(params for method, params in fake.requests if method == "thread/start")
    turn_params = next(params for method, params in fake.requests if method == "turn/start")
    assert thread_params["ephemeral"] is True
    assert thread_params["sandbox"] == "read-only"
    assert turn_params["sandboxPolicy"] == {"type": "readOnly"}
    assert "Earlier question" in turn_params["input"][0]["text"]
    assert any(isinstance(event, TextDelta) and "Grounded answer" in event.text for event in events)
    assert any(
        isinstance(event, AgentAction) and event.action == "answer.selected_papers"
        for event in events
    )


def test_conversation_reuses_one_codex_thread(tmp_path, monkeypatch):
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(
        "service.selected_paper_tool.answer_selected_papers",
        lambda **_kwargs: SelectedPaperAnswer(
            answer="ok",
            evidence="metadata",
            paper_ids=[],
            full_text_paper_ids=[],
        ),
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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))

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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))

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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))
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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))

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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))

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
    monkeypatch.setattr("app.codex.threads.config.CODEX_HOME", str(tmp_path))

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


def test_non_reasoning_and_malformed_items_do_not_emit_thinking():
    from app.codex.runner import _event_from_notification

    for item in (None, {}, {"type": "reasoning"}, {"type": "agentMessage", "id": "a"}):
        assert _event_from_notification({
            "method": "item/started", "params": {"item": item},
        }) is None


async def collect(events):
    return [event async for event in events]
