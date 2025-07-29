from __future__ import annotations

"""Utility for creating language model instances used by the agents."""

import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .configuration import Configuration


def get_llm(config: Optional[Configuration] = None) -> ChatGoogleGenerativeAI:
    """Return a Google Generative AI chat model.

    Parameters
    ----------
    config : Configuration, optional
        Configuration providing the model name. If ``None``, a default
        configuration is created which reads values from environment
        variables.

    Returns
    -------
    ChatGoogleGenerativeAI
        Instantiated chat model ready for use by agents.
    """
    cfg = config or Configuration()
    return ChatGoogleGenerativeAI(
        model=cfg.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
