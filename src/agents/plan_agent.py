from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, RootModel  # 导入 RootModel

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.plan_prompt import PLANNER_PROMPT
from phm_core import PHMState
from src.tools.signal_processing_schemas import OP_REGISTRY


# 1. 定义期望的输出结构
class Step(BaseModel):
    """A single step in the processing plan."""

    op_name: str = Field(
        ...,
        description="The name of the operator to use, must be one of the provided tools.",
    )
    params: Dict[str, Any] = Field(
        ..., description="The parameters for the operator, including node_id and parent."
    )


class Plan(BaseModel):
    """The detailed processing plan."""

    plan: List[Step] = Field(
        ..., description="A list of processing steps to execute in sequence."
    )


def plan_agent(state: PHMState) -> dict:
    """Call LLM to generate a detailed processing plan using structured output."""

    llm = get_llm(Configuration.from_runnable_config(None))
    tools_schemas = json.dumps(
        [cls.model_json_schema() for cls in OP_REGISTRY.values()], ensure_ascii=False
    )
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    if hasattr(llm, "responses"):
        chain = prompt | llm
        try:
            resp = chain.invoke(
                {
                    "instruction": state.user_instruction,
                    "channels": state.dag_state.channels,
                    "tools": tools_schemas,
                }
            )
            plan_json = json.loads(resp.content)
            if isinstance(plan_json, dict) and "plan" in plan_json:
                detailed_plan = plan_json["plan"]
            else:
                detailed_plan = plan_json
        except Exception as e:
            state.error_logs.append(f"Planner structured output error: {e}")
            detailed_plan = []
        return {"detailed_plan": detailed_plan}

    structured_llm = llm.with_structured_output(Plan)

    chain = prompt | structured_llm

    try:
        resp = chain.invoke(
            {
                "instruction": state.user_instruction,
                "channels": state.dag_state.channels,
                "tools": tools_schemas,
            }
        )
        # 3. 响应 'resp' 现在是 Plan 对象 (一个 RootModel)。
        # 我们可以直接迭代它，因为它有了 __iter__ 方法。
        detailed_plan = [step.model_dump() for step in resp]
    except Exception as e:
        # 捕获 LLM 调用或 Pydantic 验证中可能出现的错误
        state.error_logs.append(f"Planner structured output error: {e}")
        detailed_plan = []

    return {"detailed_plan": detailed_plan}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from langchain_community.chat_models import FakeListChatModel
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import PHMState, DAGState, InputData
    from src.tools import __init__ as init_tools

    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(
        responses=[
            '{"plan": ['
            '{"op_name": "mean", "params": {"parent": "ch1"}},'
            '{"op_name": "mean", "params": {"parent": "ch2"}}'
            ']}'
        ]
    )

    instruction = "轴承故障诊断"
    ch1 = InputData(node_id="ch1", data={"signal": np.ones((1, 4, 1))}, parents=[], shape=(1, 4, 1))
    ch2 = InputData(node_id="ch2", data={"signal": np.ones((1, 4, 1))}, parents=[], shape=(1, 4, 1))
    dag = DAGState(user_instruction=instruction, channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2}, leaves=["ch1", "ch2"])
    state = PHMState(user_instruction=instruction, reference_signal=ch1, test_signal=ch2, dag_state=dag)
    print({"before": state.model_dump(exclude={"reference_signal", "test_signal"})})
    out = plan_agent(state)
    print({"after": out})

