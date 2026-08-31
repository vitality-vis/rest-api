"""Provenance logging wrapper for public MCP tools."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.mcpserver.exceptions import ToolError

from app.provenance.emit import log_mcp_tool_event, summarize_mcp_result

F = TypeVar("F", bound=Callable[..., Any])


def mcp_tool_logged(tool_name: str) -> Callable[[F], F]:
    """Decorator that records one provenance envelope per MCP tool call."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except ToolError as error:
                _log_tool_call(
                    tool_name,
                    kwargs,
                    status="error",
                    started=started,
                    error_message=str(error),
                )
                raise
            except Exception as error:
                _log_tool_call(
                    tool_name,
                    kwargs,
                    status="error",
                    started=started,
                    error_message=str(error),
                )
                raise

            _log_tool_call(
                tool_name,
                kwargs,
                status="ok",
                started=started,
                result=result,
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _log_tool_call(
    tool_name: str,
    args: dict[str, Any],
    *,
    status: str,
    started: float,
    result: Any | None = None,
    error_message: str | None = None,
) -> None:
    latency_ms = max(0, int((time.perf_counter() - started) * 1000))
    log_mcp_tool_event(
        tool=tool_name,
        args=args,
        status="ok" if status == "ok" else "error",
        latency_ms=latency_ms,
        result_summary=summarize_mcp_result(result) if result is not None else None,
        error_message=error_message,
    )
