"""Process lifespan: cache init, session reset, capability refresh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.profiles import AppProfile, discover_capabilities
from app.wsgi import apply_profile_config
from service.static_cache import cached_data

if TYPE_CHECKING:
    from app.profiles import ApplicationBundle


def startup_bundle(bundle: ApplicationBundle) -> None:
    """Initialize cache, refresh capabilities, and reset full-profile sessions."""
    logger = bundle.logger
    if bundle.profile is AppProfile.FULL:
        from agents.agent_v1_legacy.runner import reset_all_sessions

        reset_all_sessions()
        print("[startup] Cleared all chat sessions (docs + memory).")

    cached_data.init()
    capabilities = discover_capabilities(
        bundle.profile,
        zilliz_ready=cached_data.zilliz_ready,
        socket_io_enabled=bundle.socketio is not None,
    )
    apply_profile_config(
        bundle.flask_app,
        profile=bundle.profile,
        capabilities=capabilities,
        socket_io_enabled=bundle.socketio is not None,
    )
    bundle.capabilities = capabilities

    if bundle.profile is AppProfile.FULL:
        from app.chat.execution import create_agent_runtime

        # Idempotent: repeated startup must not orphan a live ThreadPoolExecutor.
        if bundle.agent_runtime is None:
            bundle.agent_runtime = create_agent_runtime()
        # Flask /health reads this provider; avoid a second ASGI-only metrics URL.
        bundle.flask_app.config["VITALITY_AGENT_RUNTIME_SNAPSHOT"] = (
            bundle.agent_runtime.snapshot
        )
    else:
        bundle.flask_app.config["VITALITY_AGENT_RUNTIME_SNAPSHOT"] = None

    if bundle.profile is AppProfile.PAPERS:
        logger.info(
            "Papers profile ready (paperSearch=%s vectorSearch=%s)",
            capabilities.paper_search,
            capabilities.vector_search,
        )
    else:
        logger.info(
            "Full profile ready (paperSearch=%s chat=%s userLibrary=%s vectorSearch=%s)",
            capabilities.paper_search,
            capabilities.chat,
            capabilities.user_library,
            capabilities.vector_search,
        )


async def shutdown_bundle(bundle: ApplicationBundle) -> None:
    """Stop Agent and Socket.IO resources, then log shutdown."""
    if bundle.agent_runtime is not None:
        await bundle.agent_runtime.shutdown()
        bundle.agent_runtime = None
        bundle.flask_app.config["VITALITY_AGENT_RUNTIME_SNAPSHOT"] = None
    if bundle.socketio is not None:
        await bundle.socketio.shutdown()
    bundle.logger.info(
        "Shutting down VitaLITy ASGI app profile=%s", bundle.profile.value
    )
