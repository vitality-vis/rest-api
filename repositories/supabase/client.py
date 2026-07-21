"""Small server-only HTTP client helpers for Supabase."""

from __future__ import annotations

from dataclasses import dataclass

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
