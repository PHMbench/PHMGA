from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.plan_prompt import PLANNER_PROMPT
from src.states.phm_states import PHMState
from src.tools.signal_processing_schemas import OP_REGISTRY, AggregateOp, MultiVariableOp, get_operator
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


def _tool_descriptions() -> str:
    tool_descriptions = []
    for op in OP_REGISTRY.values():
        schema = op.model_json_schema()
        description = schema.get("description", "No description available.")
        tool_descriptions.append(
            f"- op_name: {schema.get('title', op.op_name)}\n"
            f"  description: {description}\n"
        )
    return "\n---\n".join(tool_descriptions)


def _is_feature_leaf(state: PHMState, node_id: str) -> bool:
    node = state.dag_state.nodes[node_id]
    if node.stage == "input":
        return False
    op_name = str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", ""))
    op_cls = get_operator(op_name)
    if issubclass(op_cls, AggregateOp):
        return True
    if issubclass(op_cls, MultiVariableOp):
        parent_ids = node.parents if isinstance(node.parents, list) else [node.parents]
        return all(_is_feature_leaf(state, parent_id) for parent_id in parent_ids)
    return False


def _offline_plan(state: PHMState) -> List[Dict[str, Any]]:
    leaves = list(state.dag_state.leaves)
    if not leaves:
        return []

    if all(state.dag_state.nodes[leaf].stage == "input" for leaf in leaves):
        return [{"parent": leaf, "op_name": "fft", "params": {}} for leaf in leaves]

    if all(_is_feature_leaf(state, leaf) for leaf in leaves):
        if len(leaves) >= 2 and get_dag_depth(state.dag_state) < state.min_depth:
            return [{"parent": ",".join(leaves), "op_name": "concatenate", "params": {}}]
        return []

    feature_ops = ["mean", "std", "kurtosis"]
    plan: List[Dict[str, Any]] = []
    for leaf in leaves:
        if state.dag_state.nodes[leaf].stage == "input":
            plan.append({"parent": leaf, "op_name": "fft", "params": {}})
            continue
        if _is_feature_leaf(state, leaf):
            continue
        for op_name in feature_ops:
            plan.append({"parent": leaf, "op_name": op_name, "params": {}})
    return plan


def plan_agent(state: PHMState) -> dict:
    """Generate the next single-layer plan."""

    try:
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
        runtime_config = state.runtime_config or {"llm": Configuration.from_runnable_config(None).model_dump()}
        llm = get_llm(runtime_config)
        if getattr(llm, "mode", "") == "offline_stub":
            detailed_plan = _offline_plan(state)
        else:
            prompt = PLANNER_PROMPT.format(
                instruction=state.user_instruction,
                dag_json=dag_json,
                tools=_tool_descriptions(),
                reflection=json.dumps(reflection, indent=2),
                min_depth=state.min_depth,
                min_width=state.min_width,
                max_depth=state.max_depth,
                current_depth=get_dag_depth(state.dag_state),
            )
            repair_prompt = (
                "Return only a JSON object with a single top-level key `plan`.\n"
                "Each item in `plan` must include `parent`, `op_name`, and `params`.\n\n"
                + prompt
            )
            plan_dict = llm.generate_json(prompt, repair_prompt=repair_prompt)
            for step_data in plan_dict.get("plan", []):
                if "params" in step_data and step_data["params"] == "":
                    step_data["params"] = {}
            plan_obj = Plan.model_validate(plan_dict)
            detailed_plan = [step.model_dump() for step in plan_obj.plan]

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
