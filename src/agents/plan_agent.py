from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, RootModel  # 导入 RootModel

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.plan_prompt import PLANNER_PROMPT
from src.states.phm_states import PHMState
from src.tools.signal_processing_schemas import OP_REGISTRY, get_operator
from src.utils import get_dag_depth

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
    
    # --- MODIFIED: Generate a concise, human-readable tool description ---
    tool_descriptions = []
    for op in OP_REGISTRY.values():
        schema = op.model_json_schema()
        description = schema.get('description', 'No description available.')
        
        params_info = []
        # Exclude common/internal fields from the parameter list
        excluded_params = {'node_id', 'parent', 'kind', 'in_shape', 'out_shape', 'params'}
        for name, prop in schema.get('properties', {}).items():
            if name not in excluded_params:
                param_desc = prop.get('description', 'No description.')
                params_info.append(f"- {name}: {param_desc}")
        
        params_str = "\n".join(params_info) if params_info else "        params: {}"
        
        tool_descriptions.append(
            f"- op_name: {schema.get('title', op.op_name)}\n"
            f"  description: {description}\n"
            # f"  params:\n{params_str}" # TODO 
        )
    tools_description = "\n---\n".join(tool_descriptions)

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    # 不再使用 .with_structured_output()，而是手动解析
    chain = prompt | llm

    try:
        # --- MODIFIED: Create a lightweight topology-only representation of the DAG ---
        dag_topology = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parents": node.parents,
                    "stage": node.stage,
                    "method": getattr(node, 'method', None),
                    "shape": node.shape
                }
                for node in state.dag_state.nodes.values()
            ],
            # "leaves": state.dag_state.leaves, # Optional: include leaves if needed
        }
        dag_json = json.dumps(dag_topology, indent=2)
        
        reflection = state.reflection_history

        resp = chain.invoke(
            {
                "instruction": state.user_instruction,
                "dag_json": dag_json, # Pass the lightweight topology
                "tools": tools_description,
                "reflection": json.dumps(reflection, indent=2),
                "min_depth": state.min_depth,
                "min_width": state.min_width,
                "max_depth": state.max_depth,
                "current_depth": get_dag_depth(state.dag_state),

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

        # --- Inject sampling frequency if required ---
        fs = getattr(state, "fs", None)
        if fs is None:
            fs = getattr(state.reference_signal, "meta", {}).get("fs")

        if fs is not None:
            for step in detailed_plan:
                try:
                    op_cls = get_operator(step["op_name"])
                except KeyError:
                    continue
                if "fs" in op_cls.model_fields and "fs" not in step["params"]:
                    step["params"]["fs"] = fs

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
    from src.states.phm_states import PHMState, DAGState, InputData
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

