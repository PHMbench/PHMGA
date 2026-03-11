from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from src.config.llm import OpenRouterLLMConfig, validate_provider_env


class Configuration(OpenRouterLLMConfig):
    """Backward-compatible alias over the OpenRouter-only runtime config."""

    number_of_initial_queries: int = 3
    max_research_loops: int = 2

    @classmethod
    def from_runnable_config(
        cls,
        config: Optional[RunnableConfig] = None,
    ) -> "Configuration":
        base = OpenRouterLLMConfig.from_runnable_config(config)
        configurable = config["configurable"] if config and "configurable" in config else {}
        payload = base.model_dump()
        payload["number_of_initial_queries"] = int(configurable.get("number_of_initial_queries") or 3)
        payload["max_research_loops"] = int(configurable.get("max_research_loops") or 2)
        return cls.model_validate(payload)

    @classmethod
    def validate_provider_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        return validate_provider_env(env=env, strict=strict)
