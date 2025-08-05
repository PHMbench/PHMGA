"""Utilities for language model instantiation."""

from __future__ import annotations

import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import FakeListChatModel

from ..configuration import Configuration


_FAKE_LLM: FakeListChatModel | None = None


def get_llm(
    configurable: Optional[Configuration] = None,
    *,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:
    """Return a chat model instance for agent use.

    Parameters
    ----------
    configurable : Optional[Configuration]
        Configuration object providing ``query_generator_model``. If ``None``, a
        new :class:`Configuration` will be created with environment variables.
    temperature : float, optional
        Sampling temperature for the model. Defaults to ``1.0``.
    max_retries : int, optional
        Maximum number of API retries. Defaults to ``2``.

    Returns
    -------
    ChatGoogleGenerativeAI
        Instantiated LLM ready for calls.
    """
    if configurable is None:
        configurable = Configuration() # .from_runnable_config(None)
    fake_llm =  False # configurable.fake_llm
    if fake_llm:
        global _FAKE_LLM
        # Use a shared mock model for testing
        if _FAKE_LLM is None:
            responses = [
                '[{"op_name": "mean", "params": {"parent": "ch1"}}]',
                '{"decision": "finish", "reason": "analysis complete"}',
                '{"plan": []}',
            ]
            _FAKE_LLM = FakeListChatModel(responses=responses)
        return _FAKE_LLM
    # Use real model with API key
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model=configurable.query_generator_model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=api_key,
        )