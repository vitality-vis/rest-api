"""
Centralized logging configuration for VitaLITy2
Sends logs to both terminal (console) and Google Cloud Platform
"""

import copy
import logging
import sys
from collections.abc import Mapping
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None
_gcp_handler_attached = False


class _ProvenanceConsoleFormatter(logging.Formatter):
    """Keep provenance payloads out of terminal output without changing Cloud payloads."""

    def format(self, record: logging.LogRecord) -> str:
        if not getattr(record, "provenance_event", False):
            return super().format(record)

        payload = record.msg if isinstance(record.msg, Mapping) else None
        if not isinstance(payload, Mapping):
            return super().format(record)
        console_record = copy.copy(record)
        console_record.msg = payload.get("message", "Socket Event")
        console_record.args = ()
        return super().format(console_record)


def _has_cloud_logging_handler(logger: logging.Logger) -> bool:
    try:
        from google.cloud.logging.handlers import CloudLoggingHandler
    except ImportError:
        return False
    return any(isinstance(handler, CloudLoggingHandler) for handler in logger.handlers)


def _attach_gcp_handler(logger: logging.Logger, name: str) -> None:
    """Attach Cloud Logging when credentials and dependencies are available."""
    global _gcp_handler_attached

    if _gcp_handler_attached or _has_cloud_logging_handler(logger):
        _gcp_handler_attached = True
        return

    try:
        import google.cloud.logging  # noqa: F401
        from google.cloud.logging.handlers import CloudLoggingHandler

        # Credentials are automatically detected from:
        # 1. GOOGLE_APPLICATION_CREDENTIALS environment variable
        # 2. Application Default Credentials (ADC)
        # 3. Metadata service (when running on GCP)
        # Prefer HTTP transport for Cloud Logging under Uvicorn. gRPC has been
        # flaky in some long-running process layouts; REST remains stable.
        client = google.cloud.logging.Client(_use_grpc=False)
        cloud_handler = CloudLoggingHandler(client, name=name)
        cloud_handler.setLevel(logging.INFO)
        logger.addHandler(cloud_handler)
        _gcp_handler_attached = True
        logger.info("✅ Google Cloud Logging initialized successfully")
    except ImportError:
        logger.warning(
            "⚠️ google-cloud-logging not installed. "
            "Install with: pip install google-cloud-logging"
        )
    except Exception as error:
        logger.warning(
            "⚠️ Could not initialize Google Cloud Logging: %s\n"
            "Logs will only appear in terminal. "
            "To enable GCP logging, set GOOGLE_APPLICATION_CREDENTIALS environment variable.",
            error,
        )


def setup_logger(name: str = "vitality2", enable_gcp: bool = True) -> logging.Logger:
    """
    Set up a logger that outputs to both console and Google Cloud Logging.

    Args:
        name: Logger name (default: "vitality2")
        enable_gcp: Whether to enable Google Cloud Logging (default: True)

    Returns:
        Configured logger instance
    """
    global _logger

    # Import-time callers may create a console-only logger first. Upgrade it
    # when the application entry point requests GCP logging.
    if _logger is not None:
        if enable_gcp:
            _attach_gcp_handler(_logger, name)
        return _logger

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent propagation to avoid interference with Flask/SocketIO loggers
    logger.propagate = False

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler (for terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ProvenanceConsoleFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(console_handler)

    if enable_gcp:
        _attach_gcp_handler(logger, name)

    # Store global logger
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.
    If not yet initialized, creates a new one.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        # Services may be imported outside an application entry point. Keep
        # that fallback local-only so third-party import-time logging cannot
        # affect a CloudLoggingHandler; production entry points use
        # service.bootstrap.initialize_runtime instead.
        return setup_logger(enable_gcp=False)
    return _logger


def log_structured(event_name: str, data: dict, level: str = "INFO"):
    """
    Log structured data (useful for Socket.io events).
    This creates a structured log entry that's easily queryable in GCP.

    Args:
        event_name: Name of the event
        data: Dictionary containing event data
        level: Log level (INFO, WARNING, ERROR, etc.)
    """
    logger = get_logger()

    # Create structured log entry
    log_data = {
        "event_name": event_name,
        **data
    }

    # Log at appropriate level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"Event: {event_name}", extra={"json_fields": log_data})
