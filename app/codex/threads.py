"""Map a Vitality conversation to one persisted Codex thread."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import config

_lock = threading.Lock()
# Bump when thread-scoped instructions or dynamic tool schemas change.
_THREAD_SCHEMA_VERSION = "research-v3"


def _path() -> Path:
    return Path(config.CODEX_HOME) / "vitality-threads.json"


def _key(user_id: str | None, chat_id: str) -> str:
    return f"{_THREAD_SCHEMA_VERSION}:{user_id or '-'}:{chat_id}"


def _load() -> dict[str, str]:
    path = _path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    threads = payload.get("threads") if isinstance(payload, dict) else None
    if not isinstance(threads, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in threads.items()
        if isinstance(key, str) and isinstance(value, str) and value
    }


def _save(threads: dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"threads": threads}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_thread_id(user_id: str | None, chat_id: str) -> str | None:
    """Return the Codex thread already bound to this conversation, if any."""
    if not chat_id:
        return None
    with _lock:
        return _load().get(_key(user_id, chat_id))


def put_thread_id(user_id: str | None, chat_id: str, thread_id: str) -> None:
    """Remember which Codex thread belongs to this conversation."""
    if not chat_id or not thread_id:
        return
    with _lock:
        threads = _load()
        threads[_key(user_id, chat_id)] = thread_id
        _save(threads)


def drop_thread_id(user_id: str | None, chat_id: str) -> None:
    """Forget a mapping after Codex can no longer resume that thread."""
    if not chat_id:
        return
    with _lock:
        threads = _load()
        if threads.pop(_key(user_id, chat_id), None) is not None:
            _save(threads)
