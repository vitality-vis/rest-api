"""Shared Supabase HTTP transport behavior."""
from unittest.mock import Mock, patch

from repositories.supabase import client


def test_thread_local_session_is_reused():
    client._http_state = client.local()
    session = Mock()
    with patch("repositories.supabase.client.requests.Session", return_value=session) as factory:
        first = client.get_http_session()
        second = client.get_http_session()
    assert first is second is session
    factory.assert_called_once()
    assert session.mount.call_count == 2
