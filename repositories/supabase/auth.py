"""Local verification of Supabase access tokens using the project's JWKS."""

from __future__ import annotations

from threading import Lock
from uuid import UUID

import jwt

from repositories.supabase.client import (
    SupabaseConfigurationError,
    get_supabase_settings,
)

JWKS_CACHE_SECONDS = 600
JWKS_REQUEST_TIMEOUT_SECONDS = 10
_ALLOWED_ALGORITHMS = ("ES256", "RS256")
_jwks_client: jwt.PyJWKClient | None = None
_jwks_client_url: str | None = None
_jwks_lock = Lock()


class SupabaseAuthenticationError(RuntimeError):
    """Raised when an access token is malformed, invalid, or expired."""


def _issuer(url: str) -> str:
    return f"{url.rstrip('/')}/auth/v1"


def _get_jwks_client(url: str) -> jwt.PyJWKClient:
    """Return a process-wide JWKS client with bounded key-set caching.

    PyJWKClient refreshes the JWKS automatically when a token references an
    unknown ``kid``, which supports Supabase signing-key rotation.
    """
    global _jwks_client, _jwks_client_url
    jwks_url = f"{_issuer(url)}/.well-known/jwks.json"
    if _jwks_client is not None and _jwks_client_url == jwks_url:
        return _jwks_client
    with _jwks_lock:
        if _jwks_client is None or _jwks_client_url != jwks_url:
            _jwks_client = jwt.PyJWKClient(
                jwks_url,
                cache_keys=True,
                cache_jwk_set=True,
                lifespan=JWKS_CACHE_SECONDS,
                timeout=JWKS_REQUEST_TIMEOUT_SECONDS,
            )
            _jwks_client_url = jwks_url
    return _jwks_client


def verify_access_token(access_token: str) -> str:
    """Verify a Supabase JWT locally and return its UUID subject.

    The first verification may fetch the project's public JWKS. Subsequent
    requests verify locally until the bounded JWKS cache expires.
    """
    settings = get_supabase_settings()
    token = access_token.strip()
    if not token:
        raise SupabaseAuthenticationError("Missing access token")
    try:
        signing_key = _get_jwks_client(settings.url).get_signing_key_from_jwt(token)
        algorithm = signing_key.algorithm_name
        if algorithm not in _ALLOWED_ALGORITHMS:
            raise SupabaseAuthenticationError("Unsupported access-token algorithm")
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[algorithm],
            audience="authenticated",
            issuer=_issuer(settings.url),
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
        return str(UUID(str(claims["sub"])))
    except SupabaseAuthenticationError:
        raise
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as error:
        raise SupabaseAuthenticationError("Invalid or expired access token") from error


__all__ = [
    "SupabaseAuthenticationError",
    "SupabaseConfigurationError",
    "verify_access_token",
]
