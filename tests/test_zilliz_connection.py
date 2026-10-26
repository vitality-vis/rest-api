"""Focused tests for thread-safe Zilliz connection lifecycle management."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

# Keep this focused unit test independent of application configuration loading.
_original_config = sys.modules.get("config")
sys.modules["config"] = SimpleNamespace(
    ZILLIZ_URI="https://example.test",
    ZILLIZ_TOKEN="test-token",
    ZILLIZ_REQUEST_TIMEOUT_SECONDS=30,
)
from repositories.zilliz import connection

if _original_config is None:
    sys.modules.pop("config")
else:
    sys.modules["config"] = _original_config


@pytest.fixture(autouse=True)
def reset_connection_state(monkeypatch):
    connection._reset_state_for_tests()
    monkeypatch.setattr(connection.config, "ZILLIZ_URI", "https://example.test")
    monkeypatch.setattr(connection.config, "ZILLIZ_TOKEN", "test-token")
    yield
    connection._reset_state_for_tests()


def test_concurrent_calls_construct_check_and_load_once(monkeypatch):
    counts = {"construct": 0, "has": 0, "load": 0}
    counts_lock = threading.Lock()

    class FakeClient:
        def __init__(self, **_kwargs):
            with counts_lock:
                counts["construct"] += 1
            time.sleep(0.01)

        def has_collection(self, _collection_name):
            with counts_lock:
                counts["has"] += 1
            time.sleep(0.01)
            return True

        def load_collection(self, _collection_name):
            with counts_lock:
                counts["load"] += 1
            time.sleep(0.01)

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeClient))
    barrier = threading.Barrier(12)

    def ensure_loaded():
        barrier.wait()
        return connection.ensure_collection_loaded("papers")

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(lambda _index: ensure_loaded(), range(12)))

    assert results == [True] * 12
    assert counts == {"construct": 1, "has": 1, "load": 1}


def test_loaded_collection_skips_subsequent_has_collection(monkeypatch):
    has_calls = 0

    def has_collection(_name):
        nonlocal has_calls
        has_calls += 1
        return True

    client = SimpleNamespace(
        has_collection=has_collection,
        load_collection=lambda _name: None,
    )
    monkeypatch.setattr(connection, "get_client", lambda: client)

    assert connection.ensure_collection_loaded("papers") is True
    assert connection.ensure_collection_loaded("papers") is True
    assert connection.ensure_collection_loaded("papers") is True
    assert has_calls == 1


def test_failed_client_initialization_is_retryable(monkeypatch):
    attempts = 0

    class FakeClient:
        def __init__(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary failure")

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeClient))

    assert connection.get_client() is None
    assert isinstance(connection.get_client(), FakeClient)
    assert attempts == 2


def test_failed_load_is_retryable(monkeypatch):
    load_attempts = 0

    class FakeClient:
        def has_collection(self, _name):
            return True

        def load_collection(self, _name):
            nonlocal load_attempts
            load_attempts += 1
            if load_attempts == 1:
                raise RuntimeError("temporary failure")

    client = FakeClient()
    monkeypatch.setattr(connection, "get_client", lambda: client)

    assert connection.ensure_collection_loaded("papers") is False
    assert connection.ensure_collection_loaded("papers") is True
    assert load_attempts == 2
