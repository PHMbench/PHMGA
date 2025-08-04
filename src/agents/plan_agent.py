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

    parent: str = Field(
        ...,
        description="The ID of the parent node to which this operation should be applied."
    )
    op_name: str = Field(
        ...,
        description="The name of the operator to use, must be one of the provided tools.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The parameters for the operator."
    )


class Plan(BaseModel):
    """The detailed processing plan."""

    plan: List[Step] = Field(
        ..., description="A list of processing steps to execute in sequence."
    )


def plan_agent(state: PHMState) -> dict:
    """Call LLM to generate a detailed processing plan using structured output."""

    llm = get_llm(Configuration.from_runnable_config(None))
    
    # 创建一个简化的工具描述列表，而不是完整的 JSON Schema
    simplified_tools = [
        f"- {op.op_name}: {op.description}" for op in OP_REGISTRY.values()
    ]
    tools_description = "\n".join(simplified_tools)

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    # fake test
    # if hasattr(llm, "responses"):
    #     chain = prompt | llm
    #     try:
    #         resp = chain.invoke(
    #             {
    #                 "instruction": state.user_instruction,
    #                 "dag_json": state.dag_state.model_dump_json(indent=2),
    #                 "tools": tools_description,
    #             }
    #         )
    #         plan_json = json.loads(resp.content)
    #         plan_obj = Plan.model_validate(plan_json)
    #         detailed_plan = [step.model_dump() for step in plan_obj.plan]
    #     except Exception as e:
    #         detailed_plan = []
    #         state.error_logs.append(f"Planner structured output error: {e}")
        
    #     return {"detailed_plan": detailed_plan}

    # 不再使用 .with_structured_output()，而是手动解析
    chain = prompt | llm

    try:
        # 创建 dag_state 的轻量级副本以进行序列化
        dag_state_light = state.dag_state.model_copy(deep=True)
        for node in dag_state_light.nodes.values():
            # 移除包含 numpy 数组的字段
            if hasattr(node, 'data'):
                node.data = {}
            if hasattr(node, 'results'):
                node.results = {}
            # if hasattr(node, 'processed_data'):
            #     node.processed_data = None
            del node.data
            del node.results
            # del node.processed_data

        resp = chain.invoke(
            {
                "instruction": state.user_instruction,
                "dag_json": dag_state_light.model_dump_json(indent=2), # 传递轻量级的 DAG
                "tools": tools_description,
            }
        )
        
        # 1. 从 AIMessage.content 中提取 JSON 字符串
        json_str = resp.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].strip()
        if "```" in json_str:
            json_str = json_str.split("```")[0].strip()

        # 2. 使用 json.loads() 解析字符串
        plan_dict = json.loads(json_str)

        # 3. 手动预处理（例如，处理空的 params）
        for step_data in plan_dict.get("plan", []):
            if "params" in step_data and step_data["params"] == '':
                step_data["params"] = {}
        
        # 4. 使用 Plan.model_validate() 验证和转换
        plan_obj = Plan.model_validate(plan_dict)
        detailed_plan = [step.model_dump() for step in plan_obj.plan]

    except Exception as e:
        # 捕获 LLM 调用、解析或验证中可能出现的错误
        detailed_plan = []
        error_logs = state.error_logs + [f"Planner error: {e}"]
        state.error_logs = error_logs

    return {"detailed_plan": detailed_plan}


def run_test_with_fake_llm(state: PHMState):
    """使用 FakeLLM 测试 plan_agent。"""
    print("\n--- Testing with Fake LLM ---")
    
    initial_leaves = state.dag_state.leaves
    
    # 使用 FakeLLM 来模拟一个可预测的输出
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=[
            json.dumps({
                "plan": [
                    {"parent": leaf, "op_name": "fft", "params": {}} for leaf in initial_leaves
                ]
            })
        ]
    )

    result = plan_agent(state)
    
    print("\n--- Fake LLM Plan Agent Output ---")
    print(json.dumps(result, indent=2))
    print("----------------------------------\n")

    # --- 输出验证 ---
    assert "detailed_plan" in result
    plan = result["detailed_plan"]
    assert isinstance(plan, list)
    assert len(plan) == len(initial_leaves)
    for i, step in enumerate(plan):
        assert "parent" in step
        assert "op_name" in step
        assert "params" in step
        assert step["parent"] == initial_leaves[i]
    
    print("✅ Fake LLM Plan Agent test passed!")

def run_test_with_real_llm(state: PHMState):
    """使用真实的 LLM 测试 plan_agent。"""
    print("\n--- Testing with Real LLM ---")
    
    initial_leaves = state.dag_state.leaves

    # 禁用 FakeLLM
    os.environ["FAKE_LLM"] = "false" 
    from src import model
    import importlib
    importlib.reload(model)

    # 调用 plan_agent
    real_result = plan_agent(state)

    print("\n--- Real LLM Plan Agent Output ---")
    print(json.dumps(real_result, indent=2))
    print("----------------------------------\n")

    # --- 对真实 LLM 的输出进行验证 ---
    assert "detailed_plan" in real_result
    real_plan = real_result["detailed_plan"]
    assert isinstance(real_plan, list)
    assert len(real_plan) > 0 
    for step in real_plan:
        assert "parent" in step
        assert "op_name" in step
        assert "params" in step
        assert step["parent"] in initial_leaves

    print("✅ Real LLM Plan Agent test passed!")


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from langchain_community.chat_models import FakeListChatModel
    from dotenv import load_dotenv
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import PHMState, DAGState, InputData
    from src.tools import __init__ as init_tools

    # 加载环境变量 (例如 GOOGLE_API_KEY)
    load_dotenv()

    # --- 模拟一个更真实的初始状态 ---
    instruction = "Analyze the bearing signals from multiple channels for potential faults."
    
    initial_nodes = {}
    initial_leaves = []
    channels = ["ch1", "ch2", "ch3"]
    for channel_name in channels:
        node = InputData(
            node_id=channel_name,
            results={
                "ref": np.random.randn(1, 1024, 1),
                "tst": np.random.randn(1, 1024, 1) * 1.5
            },
            parents=[],
            shape=(1, 1024, 1),
            meta={"channel": channel_name}
        )
        initial_nodes[channel_name] = node
        initial_leaves.append(channel_name)

    dag = DAGState(
        user_instruction=instruction, 
        channels=channels, 
        nodes=initial_nodes, 
        leaves=initial_leaves
    )
    
    state = PHMState(
        user_instruction=instruction, 
        reference_signal=initial_nodes["ch1"], 
        test_signal=initial_nodes["ch1"], 
        dag_state=dag
    )
    
    print("--- Initial State for Plan Agent ---")
    print(f"Instruction: {state.user_instruction}")
    print(f"Leaf Nodes (Channels): {state.dag_state.leaves}")
    print("------------------------------------")

    # 依次运行测试
    # run_test_with_fake_llm(state)
    run_test_with_real_llm(state)

