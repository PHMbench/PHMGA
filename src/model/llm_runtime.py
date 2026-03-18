"""LangChain-compatible adapters over the canonical PHM LLM client layer."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

from src.configuration import Configuration
from src.llm import LLMClient, get_llm as get_llm_client


def _prompt_to_text(prompt_input: Any) -> str:
    if isinstance(prompt_input, str):
        return prompt_input
    if isinstance(prompt_input, PromptValue):
        return prompt_input.to_string()
    if isinstance(prompt_input, BaseMessage):
        return str(prompt_input.content)
    to_string = getattr(prompt_input, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(prompt_input)


class LangChainLLMAdapter:
    """Expose task-specific Runnable adapters while reusing the canonical client."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @classmethod
    def from_config(cls, configuration: Configuration) -> "LangChainLLMAdapter":
        return cls(get_llm_client(configuration.to_runtime_dict()))

    def bind_task(self, task: str, **kwargs: Any) -> RunnableLambda:
        def _invoke(prompt_input: Any) -> AIMessage:
            prompt_text = _prompt_to_text(prompt_input)
            if task == "plan":
                result = self.client.generate_step_plan(prompt=prompt_text, **kwargs)
                return AIMessage(content=result.model_dump_json())
            if task == "param":
                result = self.client.resolve_missing_params(prompt=prompt_text, **kwargs)
                import json as json_mod

                return AIMessage(content=json_mod.dumps(result, ensure_ascii=False))
            if task == "reflect":
                result = self.client.reflect_workflow(prompt=prompt_text, **kwargs)
                return AIMessage(content=result.model_dump_json())
            if task == "report":
                result = self.client.render_report(prompt=prompt_text, **kwargs)
                return AIMessage(content=result)
            raise ValueError(f"Unsupported LangChain task adapter: {task}")

        return RunnableLambda(_invoke)


def get_llm(configuration: Configuration) -> LangChainLLMAdapter:
    """Return the LangChain-facing adapter used by rebuilt agents."""

    return LangChainLLMAdapter.from_config(configuration)
