from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, RootModel  # 导入 RootModel

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.plan_prompt import PLANNER_PROMPT
from src.states.phm_states import PHMState
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

    llm = get_llm(Configuration())
    # 现在，structured_llm 期望 LLM 返回一个步骤的 JSON 数组
    structured_llm = llm.with_structured_output(Plan)

    tools_schemas = json.dumps(
        [cls.model_json_schema() for cls in OP_REGISTRY.values()], ensure_ascii=False
    )
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    chain = prompt | structured_llm

    try:
        resp = chain.invoke(
            {
                "instruction": state.user_instruction,
                "reference_root": state.dag_state.reference_root,
                "test_root": state.dag_state.test_root,
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
    # 当直接运行此脚本进行测试时，
    # 我们需要使用绝对导入，因为相对导入的上下文已经丢失。
    # launch.json 中的 PYTHONPATH 设置使得从 'src' 开始的绝对导入成为可能。
    import numpy as np
    from src.states.phm_states import DAGState, InputData
    
    # --- 关键修复：导入工具包以填充 OP_REGISTRY ---
    # 这会执行 src/tools/__init__.py 文件，从而注册所有工具。
    from src.tools import __init__ as init_tools
    from src.tools import OP_REGISTRY
    
    # 打印已注册的工具以进行验证
    print(f"--- Registered Tools ({len(OP_REGISTRY)} total) ---")
    print(list(OP_REGISTRY.keys()))
    print("-------------------------------------\n")


    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(
        node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,)
    )
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="demo instruction",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
    )
    print("Initial state:", state.model_dump(exclude={'reference_signal', 'test_signal'})) # 简化输出
    result = plan_agent(state)
    print("\n--- Planner Output ---")
    print(result)
    print("----------------------")

