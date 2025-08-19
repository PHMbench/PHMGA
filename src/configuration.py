import os
import warnings
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """
    Legacy configuration class for backward compatibility.

    This class is deprecated. Use the UnifiedStateManager from
    src.states.phm_states for new code.
    """

    phm_model: str = Field(
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

    def __init__(self, **data):
        super().__init__(**data)
        warnings.warn(
            "Configuration class is deprecated. Use UnifiedStateManager from "
            "src.states.phm_states.get_unified_state() instead. "
            "This will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
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

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig.

        This method now integrates with the UnifiedStateManager for
        consistent configuration management.
        """
        warnings.warn(
            "Configuration.from_runnable_config is deprecated. Use "
            "get_unified_state().get_llm_config() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Try to get values from unified state first
        try:
            from src.states.phm_states import get_unified_state
            unified_state = get_unified_state()

            # Map unified state values to legacy configuration
            values = {
                "phm_model": unified_state.get('llm.model', 'gemini-2.5-pro'),
                "query_generator_model": unified_state.get('llm.query_generator_model', 'gemini-2.5-pro'),
                "reflection_model": unified_state.get('llm.reflection_model', 'gemini-2.5-pro'),
                "answer_model": unified_state.get('llm.answer_model', 'gemini-2.5-pro'),
            }

            # Override with explicit config values
            for name in cls.model_fields.keys():
                config_value = configurable.get(name)
                if config_value is not None:
                    values[name] = config_value

            return cls(**values)

        except ImportError:
            # Fallback to legacy behavior if unified state not available
            raw_values: dict[str, Any] = {
                name: os.environ.get(name.upper(), configurable.get(name))
                for name in cls.model_fields.keys()
            }

            # Filter out None values
            values = {k: v for k, v in raw_values.items() if v is not None}

            return cls(**values)
