"""Local Supabase JWT verification and claim validation."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from repositories.supabase.auth import SupabaseAuthenticationError, verify_access_token
from repositories.supabase.client import SupabaseSettings

ISSUER = "https://example.supabase.co/auth/v1"
USER_ID = "11111111-1111-4111-8111-111111111111"
PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())


class SigningKey:
    algorithm_name = "ES256"
    key = PRIVATE_KEY.public_key()


class JwksClient:
    def get_signing_key_from_jwt(self, _token):
        return SigningKey()


def token(**overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "sub": USER_ID,
        "iss": ISSUER,
        "aud": "authenticated",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, PRIVATE_KEY, algorithm="ES256", headers={"kid": "key-1"})


def verify(value):
    with (
        patch(
            "repositories.supabase.auth.get_supabase_settings",
            return_value=SupabaseSettings(
                url="https://example.supabase.co", service_role_key="secret"
            ),
        ),
        patch("repositories.supabase.auth._get_jwks_client", return_value=JwksClient()),
    ):
        return verify_access_token(value)


def test_verifies_valid_supabase_jwt_locally():
    assert verify(token()) == USER_ID


@pytest.mark.parametrize(
    "value",
    [
        token(iss="https://other.supabase.co/auth/v1"),
        token(aud="other"),
        token(exp=datetime.now(timezone.utc) - timedelta(seconds=1)),
        token(sub="not-a-uuid"),
    ],
)
def test_rejects_invalid_identity_claims(value):
    with pytest.raises(SupabaseAuthenticationError):
        verify(value)


def test_rejects_malformed_token():
    with pytest.raises(SupabaseAuthenticationError):
        verify("not-a-jwt")
