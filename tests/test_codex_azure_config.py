"""Azure-backed Codex home config generation."""

from __future__ import annotations

from pathlib import Path

from app.codex.azure_config import (
    azure_base_url,
    subprocess_env,
    write_project_codex_config,
)


def test_write_project_codex_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.codex.azure_config.config.AZURE_OPENAI_ENDPOINT",
        "https://example.openai.azure.com/",
    )
    monkeypatch.setattr(
        "app.codex.azure_config.config.AZURE_OPENAI_API_KEY", "secret-key"
    )
    monkeypatch.setattr(
        "app.codex.azure_config.config.AZURE_OPENAI_API_VERSION",
        "2025-04-01-preview",
    )
    monkeypatch.setattr(
        "app.codex.azure_config.config.AZURE_OPENAI_AVAILABLE_MODELS",
        {"gpt-test": "gpt-test-deploy"},
    )
    monkeypatch.setattr(
        "app.codex.azure_config.config.AZURE_OPENAI_DEFAULT_MODEL", "gpt-test"
    )
    monkeypatch.setattr(
        "app.codex.azure_config.config.is_azure_chat_configured", lambda: True
    )
    monkeypatch.setattr("app.codex.azure_config.config.CODEX_MODEL", "")

    path = write_project_codex_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")

    assert 'model = "gpt-test-deploy"' in text
    assert 'model_provider = "azure"' in text
    assert 'env_key = "AZURE_OPENAI_API_KEY"' in text
    assert "secret-key" not in text
    assert azure_base_url() == "https://example.openai.azure.com/openai"

    env = subprocess_env({})
    assert env["AZURE_OPENAI_API_KEY"] == "secret-key"
    assert env["OPENAI_API_KEY"] == "secret-key"
