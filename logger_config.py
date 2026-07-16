"""
Centralized logging configuration for VitaLITy2
Sends logs to both terminal (console) and Google Cloud Platform
"""

import os
import logging
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


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

    # Return existing logger if already configured
    if _logger is not None:
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
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add Google Cloud Logging handler if enabled
    if enable_gcp:
        try:
            import google.cloud.logging
            from google.cloud.logging.handlers import CloudLoggingHandler

            # Initialize Google Cloud Logging client
            # Credentials are automatically detected from:
            # 1. GOOGLE_APPLICATION_CREDENTIALS environment variable
            # 2. Application Default Credentials (ADC)
            # 3. Metadata service (when running on GCP)
            client = google.cloud.logging.Client()

            # Create Cloud Logging handler
            # Note: Shutdown warnings in short-lived scripts are harmless
            # Long-running apps (like Flask) don't have this issue
            cloud_handler = CloudLoggingHandler(client, name=name)
            cloud_handler.setLevel(logging.INFO)
            logger.addHandler(cloud_handler)

            logger.info("✅ Google Cloud Logging initialized successfully")

        except ImportError:
            logger.warning(
                "⚠️ google-cloud-logging not installed. "
                "Install with: pip install google-cloud-logging"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Could not initialize Google Cloud Logging: {e}\n"
                f"Logs will only appear in terminal. "
                f"To enable GCP logging, set GOOGLE_APPLICATION_CREDENTIALS environment variable."
            )

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
        return setup_logger()
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
