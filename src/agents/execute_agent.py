from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from src.states.phm_states import PHMState, InputData, ProcessedData
from src.model import get_llm
from src.tools.signal_processing_schemas import (
    MultiVariableOp,
    get_operator,
)


DATA_DIR = os.environ.get("PHM_DATA_DIR", "/home/lq/LQcode/2_project/PHMBench/PHMGA/save")
MAX_STEPS = 20


def _resolve_params(llm, params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve templated parameters using an LLM if needed."""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
            prompt = v.strip("{}")
            try:
                resp = llm.invoke(prompt)
                resolved[k] = float(resp.content)
            except Exception:
                resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def _get_results(node: InputData | ProcessedData) -> Dict[str, Any]:
    if isinstance(node, InputData):
        return node.results
    if isinstance(node, ProcessedData):
        return node.results
    return {}


def _extract_array(data: Any) -> npt.NDArray | None:
    """Return the first ndarray from a mapping, or the object itself."""
    if data is None:
        return None
    if isinstance(data, dict):
        if not data:
            return None
        first = next(iter(data.values()))
        return first
    return data


def execute_agent(state: PHMState) -> Dict[str, Any]:
    executed_steps = 0
    llm = get_llm(None)
    
    # Get the base save directory from environment, fallback to a default
    base_save_dir = os.environ.get("PHM_SAVE_DIR", "/home/lq/LQcode/2_project/PHMBench/PHMGA/save")
    # Construct a case-specific directory
    case_save_dir = os.path.join(base_save_dir, state.case_name, "nodes")

    # 采用不可变模式：创建当前节点和叶子的副本
    new_nodes = state.dag_state.nodes.copy()
    new_leaves = state.dag_state.leaves.copy()

    for idx, step in enumerate(state.detailed_plan[:MAX_STEPS], start=1):
        op_name = step.get("op_name")
        params = step.get("params", {})
        parent_field = step.get("parent")
        if isinstance(parent_field, str):
            parent_ids = [p.strip() for p in parent_field.split(",") if p.strip()]
        elif isinstance(parent_field, list):
            parent_ids = parent_field
        else:
            parent_ids = []

        if not parent_ids or any(pid not in new_nodes for pid in parent_ids):
            state.dag_state.error_log.append(f"Missing or invalid parent: {parent_field} in step {step}")
            continue

        try:
            params = _resolve_params(llm, params)
            op_cls = get_operator(op_name)
            op_parent = parent_ids if len(parent_ids) > 1 else parent_ids[0]
            op = op_cls(**params, parent=op_parent)

            if isinstance(op, MultiVariableOp):
                input_keys = [part.split(":")[0].strip() for part in op.input_spec.split(",")]
                ref_dict: Dict[str, npt.NDArray] = {}
                tst_dict: Dict[str, npt.NDArray] = {}
                for key, pid in zip(input_keys, parent_ids):
                    p_node = new_nodes[pid]
                    p_results = _get_results(p_node)
                    r = _extract_array(p_results.get("ref"))
                    t = _extract_array(p_results.get("tst"))
                    if r is not None:
                        ref_dict[key] = r
                    if t is not None:
                        tst_dict[key] = t
                ref_in = ref_dict if ref_dict else None
                tst_in = tst_dict if tst_dict else None
                channel = ",".join([new_nodes[pid].meta.get("channel", "unknown") for pid in parent_ids])
                parent_id_for_id = "__".join(parent_ids)
            else:
                pid = parent_ids[0]
                parent_node = new_nodes[pid]
                parent_results = _get_results(parent_node)
                ref_in = _extract_array(parent_results.get("ref"))
                tst_in = _extract_array(parent_results.get("tst"))
                channel = parent_node.meta.get("channel", "unknown")
                parent_id_for_id = pid

            out_ref = op.execute(ref_in) if ref_in is not None else None
            out_tst = op.execute(tst_in) if tst_in is not None else None

            op_abbr = op_name
            new_id = f"{op_abbr}_{idx:02d}_{parent_id_for_id}"
            kind = "both"
            if out_ref is not None and out_tst is None:
                kind = "ref"
            elif out_ref is None and out_tst is not None:
                kind = "tst"

            save_dir = os.path.join(case_save_dir, new_id)
            os.makedirs(save_dir, exist_ok=True)
            saved_meta = {}
            if out_ref is not None:
                if isinstance(out_ref, dict):
                    path = os.path.join(save_dir, "ref.npz")
                    np.savez(path, **out_ref)
                else:
                    path = os.path.join(save_dir, "ref.npy")
                    np.save(path, out_ref)
                saved_meta["ref_path"] = path

            if out_tst is not None:
                if isinstance(out_tst, dict):
                    path = os.path.join(save_dir, "tst.npz")
                    np.savez(path, **out_tst)
                else:
                    path = os.path.join(save_dir, "tst.npy")
                    np.save(path, out_tst)
                saved_meta["tst_path"] = path

            output_for_shape = out_tst if out_tst is not None else out_ref
            if isinstance(output_for_shape, dict):
                # Get shape from the first element in the dictionary
                shape = next(iter(output_for_shape.values())).shape if output_for_shape else (0,)
            elif output_for_shape is not None:
                shape = np.array(output_for_shape).shape
            else:
                shape = (0,)
            node = ProcessedData(
                node_id=new_id,
                parents=parent_ids,
                source_signal_id=parent_ids[0],
                method=op_name,
                results={"ref": out_ref, "tst": out_tst},
                meta={
                    "tool": op_name,
                    "params": params,
                    "parent": parent_field,
                    "channel": channel,
                    "kind": kind,
                    "method": op_name,
                    "saved": saved_meta,
                },
                shape=shape,
            )
            new_nodes[node.node_id] = node
            for pid in parent_ids:
                if pid in new_leaves:
                    new_leaves.remove(pid)
            new_leaves.append(node.node_id)

            executed_steps += 1
        except Exception as exc:
            state.dag_state.error_log.append(f"{parent_field}: {exc}")
            break

    # 基于更新后的副本创建新的 DAGState 对象
    new_dag_state = state.dag_state.model_copy(update={"nodes": new_nodes, "leaves": new_leaves})

    # 使用新的 DAGState 创建临时的 tracker 来生成图像
    temp_tracker = state.tracker()
    temp_tracker.update(new_dag_state)
    
    # Save the graph image to a case-specific directory
    case_graph_dir = os.path.join(base_save_dir, state.case_name, "graphs")
    os.makedirs(case_graph_dir, exist_ok=True)
    png_path = os.path.join(case_graph_dir, f"dag_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    temp_tracker.write_png(png_path)
    new_dag_state.graph_path = png_path + ".png"

    return {"dag_state": new_dag_state}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from langchain_community.chat_models import FakeListChatModel

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import PHMState, DAGState, InputData

    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    # --- 模拟一个与 plan_agent 输出一致的初始状态 ---
    instruction = "Analyze the bearing signals from multiple channels for potential faults."

    ref_sig = {'id1': np.random.randn(1, 1024, 1),
                'id2': np.random.randn(1, 1024, 1) * 1.5,
                'id3': np.random.randn(1, 1024, 1) * 2.0}
    tst_sig = {'id4': np.random.randn(1, 1024, 1) * 1.2,
                'id5': np.random.randn(1, 1024, 1) * 1.8,
                'id6': np.random.randn(1, 1024, 1) * 2.5}

    initial_nodes = {}
    initial_leaves = []
    channels = ["ch1", "ch2", "ch3"]
    for channel_name in channels:
        node = InputData(
            node_id=channel_name,
            results={
                # "ref": np.random.randn(1, 1024, 1),
                # "tst": np.random.randn(1, 1024, 1) * 1.5
                'ref':ref_sig,
                'tst':tst_sig

            },
            parents=[],
            shape=(1, 1024, 1),
            meta={"channel": channel_name},
            metadata={"source": "simulated"}
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
        dag_state=dag,
        detailed_plan=[
            {"parent": "ch1", "op_name": "fft", "params": {}},
            {"parent": "ch2", "op_name": "fft", "params": {}},
            {"parent": "ch3", "op_name": "fft", "params": {}},
        ]
    )
    
    print("--- Initial DAG State ---")
    print(f"Nodes: {list(state.dag_state.nodes.keys())}")
    print(f"Leaves: {state.dag_state.leaves}")
    print("-------------------------\n")

    # --- 执行 execute_agent 并验证结果 ---
    result = execute_agent(state)
    
    print("\n--- Updated DAG State ---")
    updated_dag = result["dag_state"]
    print(f"Nodes: {list(updated_dag.nodes.keys())}")
    print(f"Leaves: {updated_dag.leaves}")
    print("-------------------------\n")

    # --- 验证 ---
    assert len(updated_dag.nodes) == len(initial_nodes) + len(state.detailed_plan)
    assert "fft_01_ch1" in updated_dag.nodes
    assert "fft_02_ch2" in updated_dag.nodes
    assert "fft_03_ch3" in updated_dag.nodes
    assert updated_dag.leaves == ["fft_01_ch1", "fft_02_ch2", "fft_03_ch3"]
    
    print("✅ Execute Agent test passed!")

