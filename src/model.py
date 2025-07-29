from __future__ import annotations

"""Factory for language model instances used across agents."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from .configuration import Configuration


def get_llm(config: Configuration | None = None) -> ChatGoogleGenerativeAI:
    """Return the default LLM for agents.

    Parameters
    ----------
    config : Configuration | None, optional
        Optional configuration object. If not provided, ``Configuration.from_runnable_config``
        will be called with ``None`` to fetch defaults from environment variables.

    Returns
    -------
    ChatGoogleGenerativeAI
        Instantiated language model.
    """
    configurable = config or Configuration()
    return ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

