from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any
import json

import numpy as np

from src.states.phm_states import PHMState, InputData, ProcessedData
from src.model import get_llm
from src.tools.signal_processing_schemas import get_operator
from src.tools.multi_schemas import MultiVariableOp


DATA_DIR = os.environ.get("PHM_DATA_DIR", "/home/lq/LQcode/2_project/PHMBench/PHMGA/save")
MAX_STEPS = 20


def _resolve_params(llm, op_cls, params: Dict[str, Any], state: PHMState) -> Dict[str, Any]:
    """
    Resolves operator parameters. If a required parameter is missing, it uses an LLM to generate a sensible default.
    """
    resolved_params = params.copy()
    model_fields = op_cls.model_fields

    # Try to get fs from state for context, if available
    fs = getattr(state, "fs", "unknown")

    for field_name, field in model_fields.items():
        # Skip fields that are already provided or are internal/not required for execution
        if field_name in resolved_params or field_name in ["op_name", "parent", "description", "input_spec", "output_spec"]:
            continue

        # If a field is required (i.e., has no default value) and is missing, generate it.
        if field.is_required():
            prompt = f"""
You are an expert signal processing engineer. Your task is to provide a sensible default parameter for a signal processing operation.

Operator Name: {getattr(op_cls, "op_name", "N/A")}
Operator Description: {getattr(op_cls, "description", "N/A")}

A required parameter is missing:
- Parameter Name: '{field_name}'


Context:
- The signal sampling frequency (fs) is {fs} Hz.

Based on this information, provide a valid JSON value for the '{field_name}' parameter.
Your response MUST be a single JSON object containing only the generated value. For example, if generating a list of bands, respond with: [[0, 50], [50, 100]]

Do not add any other text or explanations.
"""
            try:
                resp = llm.invoke(prompt)
                # The response should be a JSON string representing the value
                generated_value = json.loads(resp.content)
                resolved_params[field_name] = generated_value
                print(f"AI generated missing parameter '{field_name}': {generated_value}")
            except Exception as e:
                print(f"Could not generate or parse parameter '{field_name}': {e}")
                # If generation fails, we cannot proceed with this op if param is required
                state.dag_state.error_log.append(f"Error setting parameter '{field_name}': {e}")
                raise ValueError(f"Failed to generate required parameter '{field_name}' for operator '{op_cls.op_name}'.") from e

    return resolved_params

# - Parameter Description: {field.description}
# - Required Type: {field.annotation}


def _get_results(node: InputData | ProcessedData) -> Dict[str, Any]:
    if isinstance(node, InputData):
        return node.results
    if isinstance(node, ProcessedData):
        return node.results
    return {}


def _execute_multi_variable_op(op, parent_ids, new_nodes):
    """
    Executes a multi-variable operator by gathering data from all parent nodes
    and applying the operator pairwise to the signals within them.
    """
    # 1. Gather the full result dictionaries from each parent
    parent_refs = {pid: _get_results(new_nodes[pid]).get("ref") for pid in parent_ids}
    parent_tsts = {pid: _get_results(new_nodes[pid]).get("tst") for pid in parent_ids}

    # Filter out parents that don't have valid results
    parent_refs = {k: v for k, v in parent_refs.items() if isinstance(v, dict)}
    parent_tsts = {k: v for k, v in parent_tsts.items() if isinstance(v, dict)}

    out_ref, out_tst = None, None

    # --- Process Reference Signals ---
    if parent_refs:
        # 2. Assume all parents share the same signal keys (e.g., 'id1', 'id2')
        #    Get the keys from the first valid parent.
        signal_keys = list(next(iter(parent_refs.values())).keys())
        
        ref_results = {}
        for key in signal_keys:
            # 3. For each signal key, build the input dict for the operator
            #    e.g., {'ch1': signal_for_id1, 'ch2': signal_for_id1}
            single_op_input = {
                parent_id: parent_data.get(key)
                for parent_id, parent_data in parent_refs.items()
            }
            # Filter out any missing signals for this key
            single_op_input = {k: v for k, v in single_op_input.items() if v is not None}
            
            if len(single_op_input) == len(parent_ids): # Ensure all parents have this signal
                ref_results[key] = op.execute(single_op_input)
        out_ref = ref_results

    # --- Process Test Signals (same logic) ---
    if parent_tsts:
        signal_keys = list(next(iter(parent_tsts.values())).keys())
        
        tst_results = {}
        for key in signal_keys:
            single_op_input = {
                parent_id: parent_data.get(key)
                for parent_id, parent_data in parent_tsts.items()
            }
            single_op_input = {k: v for k, v in single_op_input.items() if v is not None}

            if len(single_op_input) == len(parent_ids):
                tst_results[key] = op.execute(single_op_input)
        out_tst = tst_results
        
    return out_ref, out_tst


def _execute_single_variable_op(op, parent_id, new_nodes):
    """
    Executes a single-variable operator on the data from a single parent node.
    """
    parent_node = new_nodes[parent_id]
    parent_results = _get_results(parent_node)
    ref_in = parent_results.get("ref")
    tst_in = parent_results.get("tst")

    # If the input itself is a dictionary of signals, apply the op to each signal
    if isinstance(ref_in, dict):
        out_ref = {key: op.execute(value) for key, value in ref_in.items()} if ref_in else None
    else:
        out_ref = op.execute(ref_in) if ref_in is not None else None

    if isinstance(tst_in, dict):
        out_tst = {key: op.execute(value) for key, value in tst_in.items()} if tst_in else None
    else:
        out_tst = op.execute(tst_in) if tst_in is not None else None
        
    return out_ref, out_tst


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
        parent_ids_str = step.get("parent") # This might be a string like "ch1" or "ch2,ch1"

        if not parent_ids_str:
            state.dag_state.error_log.append(f"Missing parent in step {step}")
            continue
        
        parent_ids = [pid.strip() for pid in parent_ids_str.split(',')]
        
        # Validate all parents exist
        if not all(pid in new_nodes for pid in parent_ids):
            state.dag_state.error_log.append(f"One or more parents not found: {parent_ids} in step {step}")
            continue

        try:
            op_cls = get_operator(op_name)
            
            # Auto-inject 'fs' if required by the operator and not provided in the original plan
            if "fs" in op_cls.model_fields and "fs" not in params:
                # Try to get fs from the first parent's metadata
                fs_val = new_nodes[parent_ids[0]].meta.get("fs")
                if fs_val is None:
                    # Fallback to the global state fs
                    fs_val = getattr(state, "fs", None)
                if fs_val is not None:
                    params["fs"] = fs_val
            
            # Resolve any missing required parameters using the LLM
            params = _resolve_params(llm, op_cls, params, state)
            op = op_cls(**params, parent=parent_ids_str)

            # --- Decoupled Execution Logic ---
            if issubclass(op_cls, MultiVariableOp):
                out_ref, out_tst = _execute_multi_variable_op(op, parent_ids, new_nodes)
            else: # --- Handle Single-Variable Operators ---
                if len(parent_ids) > 1:
                    state.dag_state.error_log.append(f"Operator '{op_name}' is single-variable but received multiple parents: {parent_ids}")
                    continue
                parent_id = parent_ids[0]
                out_ref, out_tst = _execute_single_variable_op(op, parent_id, new_nodes)

            # Determine channel and new node ID
            # For multi-parent nodes, we can concatenate channel names
            channel = ",".join(sorted([new_nodes[pid].meta.get("channel", "unknown") for pid in parent_ids]))
            
            op_abbr = op_name
            # Create a more robust ID for multi-parent nodes
            parent_id_abbr = "_".join(sorted(parent_ids))
            new_id = f"{op_abbr}_{idx:02d}_{parent_id_abbr}"
            
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
                parents=parent_ids, # Use the list of parent IDs
                source_signal_id=parent_ids_str,
                method=op_name,
                results={"ref": out_ref, "tst": out_tst},
                meta={
                    "tool": op_name,
                    "params": params,
                    "parent": parent_ids_str,
                    "channel": channel,
                    "kind": kind,
                    "method": op_name,
                    "saved": saved_meta,
                },
                shape=shape,
            )
            # Update DAG state
            new_nodes[node.node_id] = node
            for pid in parent_ids:
                if pid in new_leaves:
                    new_leaves.remove(pid)
            new_leaves.append(node.node_id)

            executed_steps += 1
        except Exception as exc:
            state.dag_state.error_log.append(f"Error executing step {step}: {exc}")
            break

    # Create the new immutable DAG state
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

    return {"dag_state": new_dag_state, "executed_steps": executed_steps}


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
            {"parent": "ch1,ch2", "op_name": "cross_correlation", "params": {}},
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
    assert updated_dag.leaves == ["fft_01_ch1", "fft_02_ch2", "fft_03_ch3", "cross_correlation_04_ch1_ch2"]
    
    print("✅ Execute Agent test passed!")

