"""Process startup ordering for optional third-party integrations."""

import importlib

from logger_config import setup_logger


def initialize_runtime(name: str = "vitality2", enable_gcp: bool = True):
    """Configure runtime dependencies before application logging.

    PyMilvus configures its own logging during import. Import it before
    CloudLoggingHandler is attached so PyMilvus cannot flush that handler via
    ``logging.config.dictConfig``.
    """
    if enable_gcp:
        try:
            importlib.import_module("pymilvus")
        except ImportError:
            # Logging can still run locally in entry points that do not use
            # Milvus; callers that do use it will report the missing dependency.
            pass

    return setup_logger(name=name, enable_gcp=enable_gcp)
