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


def get_llm(*, streaming: bool = False) -> BaseChatModel:
    """Create the chat model used by the application.

    Callers depend on this provider-neutral entry point. Provider-specific
    configuration remains private to this module until another provider is
    actually needed.
    """
    return _NoStopAzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=1,  # GPT-5 only supports temperature=1
        streaming=streaming,
    )
