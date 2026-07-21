"""Verification of Supabase user access tokens."""

from __future__ import annotations

from uuid import UUID

import requests

from repositories.supabase.client import (
    SupabaseConfigurationError,
    get_supabase_settings,
)


AUTH_REQUEST_TIMEOUT_SECONDS = 10


class SupabaseAuthenticationError(RuntimeError):
    """Raised when an access token is malformed, invalid, or expired."""


def verify_access_token(access_token: str) -> str:
    """Verify an access token with Supabase Auth and return its user UUID."""
    settings = get_supabase_settings()
    try:
        response = requests.get(
            f"{settings.url}/auth/v1/user",
            headers={
                "apikey": settings.service_role_key,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=AUTH_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise SupabaseAuthenticationError("Could not verify access token") from error

    if response.status_code != 200:
        raise SupabaseAuthenticationError("Invalid or expired access token")

    try:
        user_id = response.json()["id"]
        return str(UUID(str(user_id)))
    except (KeyError, TypeError, ValueError) as error:
        raise SupabaseAuthenticationError("Supabase returned an invalid user identity") from error


__all__ = [
    "SupabaseAuthenticationError",
    "SupabaseConfigurationError",
    "verify_access_token",
]
