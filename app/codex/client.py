"""Long-lived stdio client for one local Codex app-server process."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import config

logger = logging.getLogger(__name__)

NotificationCallback = Callable[[dict[str, Any]], None]
ToolHandler = Callable[[dict[str, Any]], Any]


class CodexClientError(RuntimeError):
    """The Codex process or protocol could not complete a request."""


@dataclass
class _PendingRequest:
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop


@dataclass
class _ThreadListener:
    callback: NotificationCallback
    loop: asyncio.AbstractEventLoop


@dataclass
class _ThreadToolHandler:
    handler: ToolHandler
    loop: asyncio.AbstractEventLoop


class CodexClient:
    """Multiplex JSON-RPC requests, notifications, and server tool calls.

    The app-server subprocess is process-wide, but each Chat worker owns its
    own asyncio loop. Futures and callbacks are therefore bound to the loop
    that registered them, never to a single shared loop.
    """

    def __init__(self, command: list[str] | None = None):
        self.command = command or [config.CODEX_BIN, "app-server"]
        self.process: subprocess.Popen[str] | None = None
        self.workspace: tempfile.TemporaryDirectory[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, _PendingRequest] = {}
        self._listeners: dict[str, _ThreadListener] = {}
        self._tool_handlers: dict[str, _ThreadToolHandler] = {}
        self._buffered: dict[str, list[dict[str, Any]]] = defaultdict(list)

    @property
    def workspace_path(self) -> str:
        if self.workspace is None:
            raise CodexClientError("Codex client is not started.")
        return self.workspace.name

    async def start(self) -> None:
        """Start app-server and complete the initialize handshake."""
        if self.process is not None and self.process.poll() is None:
            return
        self.workspace = tempfile.TemporaryDirectory(prefix="vitality-codex-")
        os.makedirs(config.CODEX_HOME, exist_ok=True)
        from app.codex.azure_config import subprocess_env, write_project_codex_config

        write_project_codex_config(config.CODEX_HOME)
        env = subprocess_env(os.environ.copy())
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(
            target=self._read_stdout, name="codex-app-server", daemon=True
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, name="codex-app-server-stderr", daemon=True
        )
        self._reader.start()
        self._stderr_reader.start()
        try:
            await self.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "vitality",
                        "title": "VitaLITy",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                timeout=20,
            )
        except Exception:
            await self.close()
            raise
        self.notify("initialized", {})

    async def close(self) -> None:
        """Stop the process and fail any outstanding request."""
        process = self.process
        self.process = None
        with self._state_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            if item.loop.is_closed():
                continue
            item.loop.call_soon_threadsafe(
                _resolve_future, item.future, None, CodexClientError("Codex client closed.")
            )
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, 2)
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None

    def listen(self, thread_id: str, callback: NotificationCallback) -> None:
        """Receive notifications for one thread, including those already buffered."""
        loop = asyncio.get_running_loop()
        with self._state_lock:
            buffered = self._buffered.pop(thread_id, [])
            self._listeners[thread_id] = _ThreadListener(callback=callback, loop=loop)
        for message in buffered:
            loop.call_soon(callback, message)

    def handle_tool(self, thread_id: str, handler: ToolHandler) -> None:
        """Register the coroutine that executes dynamic tool calls for a thread."""
        loop = asyncio.get_running_loop()
        with self._state_lock:
            self._tool_handlers[thread_id] = _ThreadToolHandler(handler=handler, loop=loop)

    def clear_thread(self, thread_id: str) -> None:
        with self._state_lock:
            self._listeners.pop(thread_id, None)
            self._tool_handlers.pop(thread_id, None)
            self._buffered.pop(thread_id, None)

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        """Send one request and wait for its matching response."""
        if self.process is None or self.process.poll() is not None:
            raise CodexClientError("Codex client is not started.")
        loop = asyncio.get_running_loop()
        with self._state_lock:
            self._next_id += 1
            request_id = self._next_id
            future: asyncio.Future[Any] = loop.create_future()
            self._pending[request_id] = _PendingRequest(future=future, loop=loop)
        self._send({"id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise CodexClientError("Codex client is not started.")
        encoded = json.dumps(message, ensure_ascii=False)
        with self._write_lock:
            process.stdin.write(encoded + "\n")
            process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Codex app-server sent a non-JSON line")
                continue
            if not isinstance(message, dict):
                continue
            self._receive(message)
        self._fail_pending(CodexClientError("Codex app-server exited."))

    def _read_stderr(self) -> None:
        """Drain stderr so a chatty app-server cannot block on a full pipe."""
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.rstrip()
            if text:
                logger.info("codex app-server: %s", text)

    def _receive(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            self._schedule_server_request(message)
            return
        if "id" in message:
            with self._state_lock:
                pending = self._pending.get(message["id"])
            if pending is None:
                return
            error = message.get("error")
            if error is not None:
                detail = error.get("message") if isinstance(error, dict) else str(error)
                exception = CodexClientError(detail or "Codex request failed.")
                pending.loop.call_soon_threadsafe(
                    _resolve_future, pending.future, None, exception
                )
            else:
                pending.loop.call_soon_threadsafe(
                    _resolve_future, pending.future, message.get("result"), None
                )
            return
        if "method" in message:
            self._dispatch_notification(message)

    def _schedule_server_request(self, message: dict[str, Any]) -> None:
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        thread_id = str(params.get("threadId") or "")
        with self._state_lock:
            tool = self._tool_handlers.get(thread_id)
            listener = self._listeners.get(thread_id)
        loop = (
            tool.loop
            if tool is not None
            else listener.loop
            if listener is not None
            else None
        )
        if loop is None or loop.is_closed():
            self._send(
                {
                    "id": message["id"],
                    "result": {
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": "This tool is no longer available.",
                            }
                        ],
                        "success": False,
                    },
                }
            )
            return
        asyncio.run_coroutine_threadsafe(self._handle_server_request(message), loop)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        try:
            if method == "item/tool/call":
                result = await self._call_tool(params)
            elif method.endswith("requestApproval") or method.endswith("RequestApproval"):
                result = {"decision": "decline"}
            else:
                self._send(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": f"Unsupported server request: {method}",
                        },
                    }
                )
                return
        except Exception as error:
            logger.warning("Codex tool call failed: %s", error, exc_info=True)
            result = {
                "contentItems": [
                    {"type": "inputText", "text": "The tool could not be completed."}
                ],
                "success": False,
            }
        self._send({"id": message["id"], "result": result})

    async def _call_tool(self, params: dict[str, Any]) -> Any:
        thread_id = str(params.get("threadId") or "")
        with self._state_lock:
            registered = self._tool_handlers.get(thread_id)
        if registered is None:
            return {
                "contentItems": [
                    {"type": "inputText", "text": "This tool is no longer available."}
                ],
                "success": False,
            }
        result = registered.handler(params)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def _dispatch_notification(self, message: dict[str, Any]) -> None:
        thread_id = _thread_id(message)
        with self._state_lock:
            listener = self._listeners.get(thread_id) if thread_id else None
            if listener is None and thread_id:
                self._buffered[thread_id].append(message)
                return
        if listener is not None:
            listener.loop.call_soon_threadsafe(listener.callback, message)

    def _fail_pending(self, error: Exception) -> None:
        with self._state_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.loop.call_soon_threadsafe(_resolve_future, item.future, None, error)


def _resolve_future(
    future: asyncio.Future[Any], value: Any, error: Exception | None
) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)


def _thread_id(message: dict[str, Any]) -> str | None:
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    thread_id = params.get("threadId")
    if isinstance(thread_id, str):
        return thread_id
    thread = params.get("thread")
    if isinstance(thread, dict) and isinstance(thread.get("id"), str):
        return thread["id"]
    turn = params.get("turn")
    if isinstance(turn, dict) and isinstance(turn.get("threadId"), str):
        return turn["threadId"]
    return None


_shared_client: CodexClient | None = None
_shared_lock = threading.Lock()
_start_lock = threading.Lock()


async def warm_shared_client() -> None:
    """Start app-server during process startup so the first chat skips spawn."""
    await get_shared_client()


async def get_shared_client() -> CodexClient:
    """Start the process-wide app-server client on first use."""
    global _shared_client
    with _shared_lock:
        client = _shared_client
        if client is not None and client.process is not None and client.process.poll() is None:
            return client
    # Serialize process startup across Chat worker threads.
    await asyncio.to_thread(_start_lock.acquire)
    try:
        with _shared_lock:
            client = _shared_client
            if (
                client is not None
                and client.process is not None
                and client.process.poll() is None
            ):
                return client
            client = CodexClient()
            _shared_client = client
        try:
            await client.start()
        except Exception:
            with _shared_lock:
                if _shared_client is client:
                    _shared_client = None
            raise
        return client
    finally:
        _start_lock.release()


async def close_shared_client() -> None:
    """Stop the shared client during application shutdown."""
    global _shared_client
    with _shared_lock:
        client = _shared_client
        _shared_client = None
    if client is not None:
        await client.close()
