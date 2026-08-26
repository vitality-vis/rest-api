"""Composition root helpers for profile-aware app construction."""

from __future__ import annotations

from app.profiles import AppProfile, ApplicationBundle, resolve_profile


def create_application(profile: str | AppProfile | None = None) -> ApplicationBundle:
    """Create the Flask (and optional Socket.IO) bundle for the selected profile.

    ``profile`` may be an ``AppProfile``, a raw string, or ``None`` to read
    ``VITALITY_APP_PROFILE`` (default ``full``).
    """
    if isinstance(profile, AppProfile):
        selected = profile
    else:
        selected = resolve_profile(profile)

    if selected is AppProfile.PAPERS:
        from app.profiles.papers import create_papers_bundle

        return create_papers_bundle()

    from app.profiles.full import create_full_bundle

    return create_full_bundle()
