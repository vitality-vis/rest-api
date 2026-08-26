"""Live check: /getPapers stays responsive while /chat/v2 is running.

Logic:
  1. Start /chat/v2 (medium paper search, streamed).
  2. Wait a short delay so the server has begun the chat work.
  3. Call /getPapers and measure how long it takes.
  4. Also record when chat gets headers / first body byte / finishes.
  5. Fail if /getPapers is slow or errors relative to an idle baseline.

Run against an already-running API server:
    make test-live TESTS=tests/test_api_concurrency.py

Optional env:
    CHAT_START_DELAY_S=0.3       # wait after starting chat before /getPapers
    GET_PAPERS_TIMEOUT_S=15      # max wait for /getPapers during chat
    CHAT_V2_INFLIGHT=1           # concurrent chat streams (default 1)
"""

from __future__ import annotations

import os
import threading
import time
from uuid import uuid4

import pytest
import requests


pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Return the explicitly selected running API server, or skip this live check."""
    base_url = os.getenv("API_BASE_URL")
    if not base_url:
        pytest.skip("API_BASE_URL is not set; no running API server was selected")
    return base_url.rstrip("/")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _get_papers(api_base_url: str, *, timeout_s: float) -> float:
    started = time.monotonic()
    response = requests.post(
        f"{api_base_url}/getPapers",
        json={"offset": 0, "limit": 20},
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert response.headers.get("Content-Type", "").startswith("application/json")
    data = response.json()
    assert isinstance(data.get("papers"), list)
    assert len(data["papers"]) > 0
    return elapsed


def test_get_papers_responds_while_chat_v2_in_flight(api_base_url: str):
    """Start chat, briefly wait, then time /getPapers against the idle baseline."""
    inflight = max(1, _env_int("CHAT_V2_INFLIGHT", 1))
    chat_start_delay_s = max(0.0, _env_float("CHAT_START_DELAY_S", 0.3))
    get_papers_timeout_s = max(1.0, _env_float("GET_PAPERS_TIMEOUT_S", 15))

    baseline_s = _get_papers(api_base_url, timeout_s=30)
    print(f"\n[concurrency] baseline /getPapers={baseline_s:.2f}s")

    t0 = time.monotonic()
    timings_lock = threading.Lock()
    chat_timings: list[dict[str, float | None]] = []
    chat_errors: list[BaseException] = []

    def run_chat(index: int) -> None:
        timing: dict[str, float | None] = {
            "headers_s": None,
            "first_byte_s": None,
            "done_s": None,
        }
        payload = {
            "client_request_id": str(uuid4()),
            "chat_id": f"api-concurrency-chat-{index}-{uuid4()}",
            "text": (
                "Find papers about using large language models to support "
                "literature reviews. List the relevant papers."
            ),
            "effort": "medium",
        }
        try:
            with requests.post(
                f"{api_base_url}/chat/v2",
                json=payload,
                stream=True,
                timeout=(10, 300),
            ) as response:
                timing["headers_s"] = time.monotonic() - t0
                if response.status_code != 200:
                    raise AssertionError(
                        f"/chat/v2 status {response.status_code}: {response.text[:500]}"
                    )
                for chunk in response.iter_content(chunk_size=64):
                    if not chunk:
                        continue
                    if timing["first_byte_s"] is None:
                        timing["first_byte_s"] = time.monotonic() - t0
                timing["done_s"] = time.monotonic() - t0
        except BaseException as error:  # noqa: BLE001 - collect for main thread
            with timings_lock:
                chat_errors.append(error)
            timing["done_s"] = time.monotonic() - t0
        finally:
            with timings_lock:
                chat_timings.append(timing)

    threads = [
        threading.Thread(target=run_chat, args=(i,), daemon=True)
        for i in range(inflight)
    ]
    for thread in threads:
        thread.start()

    time.sleep(chat_start_delay_s)

    get_papers_started_s = time.monotonic() - t0
    get_papers_elapsed: float | None = None
    get_papers_error: BaseException | None = None
    try:
        get_papers_elapsed = _get_papers(
            api_base_url, timeout_s=get_papers_timeout_s
        )
    except BaseException as error:  # noqa: BLE001 - report after chat joins
        get_papers_error = error
        get_papers_elapsed = time.monotonic() - t0 - get_papers_started_s

    for thread in threads:
        thread.join(timeout=300)

    with timings_lock:
        chats = list(chat_timings)
        errors = list(chat_errors)

    print(
        f"[concurrency] chat_start_delay={chat_start_delay_s:.2f}s, "
        f"getPapers_started_at={get_papers_started_s:.2f}s, "
        f"getPapers_elapsed={_fmt(get_papers_elapsed)}s "
        f"(baseline={baseline_s:.2f}s, timeout={get_papers_timeout_s:.0f}s)"
    )
    for index, timing in enumerate(chats):
        print(
            f"[concurrency] chat[{index}] "
            f"headers={_fmt(timing['headers_s'])}s "
            f"first_byte={_fmt(timing['first_byte_s'])}s "
            f"done={_fmt(timing['done_s'])}s"
        )

    if errors:
        pytest.fail(f"/chat/v2 failed: {errors[0]}")
    if get_papers_error is not None:
        pytest.fail(
            f"/getPapers failed while chat in flight "
            f"(started_at={get_papers_started_s:.2f}s, "
            f"elapsed={_fmt(get_papers_elapsed)}s, baseline={baseline_s:.2f}s): "
            f"{get_papers_error}"
        )

    assert get_papers_elapsed is not None
    # Idle getPapers is usually <1s. A multi-second jump that lines up with
    # chat duration is head-of-line blocking (see chat headers≈done timing).
    slow_limit_s = max(2.0, baseline_s * 5)
    if get_papers_elapsed > slow_limit_s:
        pytest.fail(
            f"/getPapers took {get_papers_elapsed:.2f}s during chat "
            f"(baseline {baseline_s:.2f}s, limit {slow_limit_s:.2f}s); "
            f"likely blocked by /chat/v2"
        )
