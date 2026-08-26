"""Local process startup helpers."""

from __future__ import annotations

import os
from typing import Any

import uvicorn


def run_local(
    app: Any,
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    debug: bool = False,
) -> None:
    """Run the ASGI app with Uvicorn.

    Pass the app **object**, not an import string, so ``python main.py`` does not
    re-import ``main`` and double-run application construction / lifespan setup.
    """
    uvicorn.run(
        app,
        host=host,
        port=int(os.environ.get("PORT", 3000) if port is None else port),
        log_level="debug" if debug else "info",
    )
