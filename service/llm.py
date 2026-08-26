"""Shared chat-model construction for agents and application services."""
from __future__ import annotations

import os

import config  # Loads the project environment before the LLM is configured.
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI


class _NoStopAzureChatOpenAI(AzureChatOpenAI):
    """Azure chat model that ignores stop sequences unsupported by GPT-5."""

    def _generate(self, messages, stop=None, **kwargs):
        return super()._generate(messages, stop=None, **kwargs)

    def generate(self, messages, stop=None, **kwargs):
        return super().generate(messages, stop=None, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        return super().generate_prompt(prompts, stop=None, **kwargs)


def get_llm(*, model: str | None = None, streaming: bool = False) -> BaseChatModel:
    """Create the chat model used by the application.

    ``model`` is a logical key from ``AZURE_OPENAI_AVAILABLE_MODELS``. When
    omitted, ``AZURE_OPENAI_DEFAULT_MODEL`` is used. Callers depend on this
    provider-neutral entry point; provider-specific configuration stays here.
    """
    deployment = config.resolve_chat_deployment(model)
    return _NoStopAzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=deployment,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=1,  # GPT-5 only supports temperature=1
        streaming=streaming,
        timeout=config.LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )
