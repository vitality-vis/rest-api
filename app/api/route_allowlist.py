"""Explicit papers/full blueprint allowlists for profile-aware registration.

HTTP paths are unchanged; this module only controls which blueprints are imported and registered.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any

# (import module, blueprint attribute)
PAPERS_BLUEPRINTS: tuple[tuple[str, str], ...] = (
    ("app.api.public.health", "health_bp"),
    ("app.api.public.papers", "papers_bp"),
    ("app.api.public.lookup", "lookup_bp"),
    ("app.api.public.corpus", "corpus_bp"),
)

FULL_ONLY_BLUEPRINTS: tuple[tuple[str, str], ...] = (
    ("app.api.user.config", "app_config_bp"),
    ("app.api.user.export", "export_bp"),
    ("app.api.user.papers", "user_papers_bp"),
    ("app.api.user.library", "library_bp"),
    ("app.api.user.notes", "notes_bp"),
    ("app.api.chat", "chat_bp"),
    ("agents.agent_v1_legacy.summary_routes", "legacy_summary_bp"),
)

FULL_BLUEPRINTS: tuple[tuple[str, str], ...] = PAPERS_BLUEPRINTS + FULL_ONLY_BLUEPRINTS


def _load_blueprint(module_name: str, attr_name: str) -> Any:
    module = import_module(module_name)
    return getattr(module, attr_name)


def iter_blueprint_specs(specs: Iterable[tuple[str, str]]) -> Iterable[Any]:
    for module_name, attr_name in specs:
        yield _load_blueprint(module_name, attr_name)


def load_papers_blueprints() -> list[Any]:
    """Import and return only public paper blueprints."""
    return list(iter_blueprint_specs(PAPERS_BLUEPRINTS))


def load_full_blueprints() -> list[Any]:
    """Import and return papers + full-only blueprints."""
    return list(iter_blueprint_specs(FULL_BLUEPRINTS))
