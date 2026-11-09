"""Token-bucket behavior for Notes completion requests."""
from service.note_completion_rate_limit import NoteCompletionRateLimiter


def test_rate_limiter_refills_after_interval():
    limiter = NoteCompletionRateLimiter(interval_seconds=3.0, capacity=2)

    assert limiter.check("user:project", now=10.0) == (True, 0)
    assert limiter.check("user:project", now=10.0) == (True, 0)
    assert limiter.check("user:project", now=10.0) == (False, 3)
    assert limiter.check("user:project", now=13.0) == (True, 0)


def test_rate_limiter_keeps_keys_independent():
    limiter = NoteCompletionRateLimiter(interval_seconds=3.0, capacity=1)

    assert limiter.check("user-a:project", now=10.0) == (True, 0)
    assert limiter.check("user-a:project", now=10.0) == (False, 3)
    assert limiter.check("user-b:project", now=10.0) == (True, 0)
