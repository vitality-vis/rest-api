"""Application profile selection and capability discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import config


class AppProfile(str, Enum):
    PAPERS = "papers"
    FULL = "full"


DEFAULT_APP_PROFILE = AppProfile.FULL
PROFILE_ENV_VAR = "VITALITY_APP_PROFILE"


@dataclass(frozen=True)
class Capabilities:
    paper_search: bool
    bm25_search: bool
    vector_search: bool
    chat: bool
    user_library: bool
    socket_io: bool

    def as_dict(self) -> dict[str, bool]:
        """Wire format for /health (camelCase keys)."""
        return {
            "paperSearch": self.paper_search,
            "bm25Search": self.bm25_search,
            "vectorSearch": self.vector_search,
            "chat": self.chat,
            "userLibrary": self.user_library,
            "socketIo": self.socket_io,
        }


def resolve_profile(raw: str | None = None) -> AppProfile:
    """Parse ``VITALITY_APP_PROFILE`` (default: full)."""
    value = (raw if raw is not None else os.environ.get(PROFILE_ENV_VAR, "")).strip().lower()
    if not value:
        return DEFAULT_APP_PROFILE
    try:
        return AppProfile(value)
    except ValueError as error:
        allowed = ", ".join(profile.value for profile in AppProfile)
        raise ValueError(
            f"Invalid {PROFILE_ENV_VAR}={value!r}. Expected one of: {allowed}"
        ) from error


def discover_capabilities(
    profile: AppProfile,
    *,
    zilliz_ready: bool,
    socket_io_enabled: bool,
) -> Capabilities:
    """Derive a capability snapshot without performing external I/O."""
    vector_ready = zilliz_ready and config.is_azure_embedding_configured()
    is_full = profile is AppProfile.FULL
    return Capabilities(
        paper_search=zilliz_ready,
        bm25_search=zilliz_ready,
        vector_search=vector_ready,
        chat=is_full and config.is_azure_chat_configured(),
        user_library=is_full and config.is_supabase_configured(),
        socket_io=bool(socket_io_enabled),
    )


@dataclass
class ApplicationBundle:
    profile: AppProfile
    flask_app: Any
    socketio: Any | None
    capabilities: Capabilities
    logger: Any
