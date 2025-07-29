"""Factory for language model instances used by agents."""

from __future__ import annotations

import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .configuration import Configuration


def get_default_llm(
    config: Optional[Configuration] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model for agent use.

    Parameters
    ----------
    config : Configuration, optional
        Optional configuration from which to read default model name.
    model_name : str, optional
        Explicit model name to instantiate. If omitted, ``config`` or
        environment variables provide ``query_generator_model``.
    **kwargs : Any
        Additional parameters forwarded to ``ChatGoogleGenerativeAI``.

    Returns
    -------
    ChatGoogleGenerativeAI
        Configured chat model instance ready for invocation.
    """
    conf = config or Configuration.from_runnable_config(None)
    name = model_name or conf.query_generator_model
    return ChatGoogleGenerativeAI(
        model=name,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
        **kwargs,
    )
