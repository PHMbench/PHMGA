import os
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    llm_provider: str = Field(
        default="gemini",
        metadata={
            "description": "LLM provider: gemini | openai_compatible | deepseek | glm. "
            "Can also be set via env LLM_PROVIDER."
        },
    )

    phm_model: str = Field(
        # default="gemini-2.5-pro", 2.0-flash
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the phm agent's."
        },
    )

    query_generator_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    fake_llm: bool = Field(
        default=False,
        metadata={
            "description": "Use a fake LLM for testing purposes. If set to True, the model will not make real API calls."
        },
    )

    # Optional OpenAI-compatible settings (DeepSeek / GLM / OpenAI-compatible gateway).
    # Prefer environment variables; do not hardcode secrets in YAML.
    openai_api_key: Optional[str] = Field(
        default=None,
        metadata={"description": "API key for OpenAI-compatible providers (env: OPENAI_API_KEY)."},
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        metadata={"description": "Base URL for OpenAI-compatible providers (env: OPENAI_BASE_URL)."},
    )
    deepseek_api_key: Optional[str] = Field(
        default=None,
        metadata={"description": "DeepSeek API key (env: DEEPSEEK_API_KEY)."},
    )
    deepseek_api_base: Optional[str] = Field(
        default=None,
        metadata={"description": "DeepSeek base URL (env: DEEPSEEK_API_BASE)."},
    )
    glm_api_key: Optional[str] = Field(
        default=None,
        metadata={"description": "GLM/Zhipu API key (env: GLM_API_KEY)."},
    )
    glm_api_base: Optional[str] = Field(
        default=None,
        metadata={"description": "GLM/Zhipu base URL (env: GLM_API_BASE)."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)

    @classmethod
    def validate_provider_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        env_map = dict(os.environ)
        if env:
            env_map.update(env)

        provider = (env_map.get("LLM_PROVIDER") or "gemini").strip().lower()
        model = (env_map.get("QUERY_GENERATOR_MODEL") or env_map.get("PHM_MODEL") or "").strip()
        model_lc = model.lower()

        errors: list[str] = []
        warnings: list[str] = []

        if provider == "glm":
            if model_lc.startswith("gemini"):
                errors.append("LLM_PROVIDER=glm but model looks like a Gemini model.")
            if not env_map.get("GLM_API_BASE"):
                errors.append("Missing GLM_API_BASE.")
            if not env_map.get("GLM_API_KEY"):
                errors.append("Missing GLM_API_KEY.")
        elif provider == "gemini":
            if model_lc.startswith("glm") or "deepseek" in model_lc:
                errors.append("LLM_PROVIDER=gemini but model looks OpenAI-compatible.")
            if not env_map.get("GEMINI_API_KEY"):
                warnings.append("GEMINI_API_KEY is not set.")
        elif provider in {"openai", "openai_compatible", "deepseek"}:
            if not (env_map.get("OPENAI_API_KEY") or env_map.get("DEEPSEEK_API_KEY") or env_map.get("GLM_API_KEY")):
                errors.append("OpenAI-compatible provider selected but no API key found.")
            if not (env_map.get("OPENAI_BASE_URL") or env_map.get("OPENAI_API_BASE") or env_map.get("DEEPSEEK_API_BASE") or env_map.get("GLM_API_BASE")):
                errors.append("OpenAI-compatible provider selected but no BASE URL found.")
        elif provider == "auto":
            warnings.append("LLM_PROVIDER=auto may route unexpectedly when multiple BASE URLs are set.")
        else:
            errors.append(f"Unsupported LLM_PROVIDER={provider!r}.")

        report = {
            "provider": provider,
            "model": model,
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
        if strict and errors:
            raise ValueError("; ".join(errors))
        return report
