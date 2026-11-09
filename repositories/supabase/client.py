"""Small server-only HTTP client helpers for Supabase."""

from __future__ import annotations

from dataclasses import dataclass
from threading import local
from typing import Any

import requests

import config


class SupabaseConfigurationError(RuntimeError):
    """Raised when server-side Supabase credentials are unavailable."""


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    service_role_key: str


def get_supabase_settings() -> SupabaseSettings:
    """Return required server-only Supabase connection settings."""
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        raise SupabaseConfigurationError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured"
        )
    return SupabaseSettings(
        url=config.SUPABASE_URL,
        service_role_key=config.SUPABASE_SERVICE_ROLE_KEY,
    )


def service_role_headers(settings: SupabaseSettings) -> dict[str, str]:
    """Build request headers for trusted server-to-Supabase calls."""
    return {
        "apikey": settings.service_role_key,
        "Authorization": f"Bearer {settings.service_role_key}",
        "Content-Type": "application/json",
    }


_http_state = local()


def get_http_session() -> requests.Session:
    """Return a thread-local Session so HTTPS connections are safely reused."""
    session = getattr(_http_state, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _http_state.session = session
    return session


def pooled_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send an HTTP request through the current thread's reusable pool."""
    return get_http_session().request(method, url, **kwargs)
