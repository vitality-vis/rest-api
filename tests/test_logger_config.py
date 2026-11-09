"""Tests for logger bootstrap ordering."""

from __future__ import annotations

import logging

import logger_config


def _reset_logger_state() -> None:
    if logger_config._logger is not None:
        logger_config._logger.handlers.clear()
    logger_config._logger = None
    logger_config._gcp_handler_attached = False
    logging.getLogger("vitality2").handlers.clear()


def test_setup_logger_upgrades_console_only_logger(monkeypatch):
    _reset_logger_state()
    try:
        calls: list[str] = []

        def fake_attach(logger: logging.Logger, name: str) -> None:
            calls.append(name)
            logger.info("✅ Google Cloud Logging initialized successfully")

        monkeypatch.setattr(logger_config, "_attach_gcp_handler", fake_attach)

        logger_config.setup_logger(enable_gcp=False)
        logger_config.setup_logger(enable_gcp=True)

        assert calls == ["vitality2"]
    finally:
        _reset_logger_state()


def test_setup_logger_suppresses_routine_http_client_logs():
    _reset_logger_state()
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_levels = (httpx_logger.level, httpcore_logger.level)

    try:
        httpx_logger.setLevel(logging.INFO)
        httpcore_logger.setLevel(logging.DEBUG)

        logger_config.setup_logger(enable_gcp=False)

        assert httpx_logger.level == logging.WARNING
        assert httpcore_logger.level == logging.WARNING
    finally:
        httpx_logger.setLevel(original_levels[0])
        httpcore_logger.setLevel(original_levels[1])
        _reset_logger_state()
