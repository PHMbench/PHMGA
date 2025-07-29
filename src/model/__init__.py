"""Utilities for language model instantiation."""

from __future__ import annotations

import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from ..configuration import Configuration


def get_llm(configurable: Optional[Configuration] = None,
            *,
            temperature: float = 1.0,
            max_retries: int = 2) -> ChatGoogleGenerativeAI:
    """Return a default ChatGoogleGenerativeAI instance.

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
        configurable = Configuration.from_runnable_config(None)

    return ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
