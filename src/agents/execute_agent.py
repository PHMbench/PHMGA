from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any
import json

import numpy as np

from src.model.explainable.operator_catalog import is_contract_allowed, resolve_operator_contract
from src.states.phm_states import PHMState, InputData, ProcessedData
from src.model import get_llm
from src.tools.signal_processing_schemas import get_operator
from src.tools.multi_schemas import MultiVariableOp
from src.utils.logging_setup import get_current_logger, log_event, timed


MAX_STEPS = 20

_OP_ALIASES: Dict[str, str] = {
    "spectral_entropy": "spectral_flatness",
    "hilbert": "hilbert_envelope",
    "zscore": "normalize",
    "welch": "psd",
}


def _categorize_step_failure(exc: Exception | str) -> str:
    text = str(exc).lower()
    if "not installed" in text or "importerror" in text or "required for" in text:
        return "missing_dep"
    if "unknown op_name" in text or "unsupported" in text or "operator" in text and "not registered" in text:
        return "unsupported_op"
    if "parameter" in text or "params" in text or "missing" in text or "cutoff" in text:
        return "param_error"
    return "runtime_error"


def _resolve_operator_name(op_name: str) -> tuple[str | None, str | None]:
    from src.tools.signal_processing_schemas import OP_REGISTRY

    raw = str(op_name or "").strip()
    if not raw:
        return None, "missing op_name"
    normalized = raw.lower()
    if normalized in OP_REGISTRY:
        return normalized, None
    alias = _OP_ALIASES.get(normalized)
    if alias and alias in OP_REGISTRY:
        return alias, f"map '{raw}' -> '{alias}'"
    return None, f"unknown op_name '{raw}'"


def _resolve_fs_fallback(state: PHMState, parent_ids: list[str], nodes: Dict[str, Any]) -> float:
    # priority: parent.meta.fs -> state.fs -> data_cfg.sample_rate -> 12800
    for pid in parent_ids:
        node = nodes.get(pid)
        if node is None:
            continue
        meta = getattr(node, "meta", None)
        if isinstance(meta, dict) and meta.get("fs") is not None:
            try:
                return float(meta["fs"])
            except Exception:
                pass
    fs_state = getattr(state, "fs", None)
    if fs_state is not None:
        try:
            return float(fs_state)
        except Exception:
            pass
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    for key in ("sample_rate", "fs_hz", "fs"):
        if data_cfg.get(key) is None:
            continue
        try:
            return float(data_cfg[key])
        except Exception:
            continue
    return 12800.0


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_params(
    llm,
    op_cls,
    params: Dict[str, Any],
    state: PHMState,
    *,
    parent_ids: list[str] | None = None,
    nodes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Resolves operator parameters. If a required parameter is missing, it uses an LLM to generate a sensible default.
    """
    resolved_params = params.copy()
    model_fields = op_cls.model_fields
    op_name = str(getattr(op_cls, "op_name", "") or "").strip().lower()

    if op_name == "filter":
        if "fs" not in resolved_params:
            fallback_fs = _resolve_fs_fallback(state, parent_ids or [], nodes or {})
            resolved_params["fs"] = float(fallback_fs)
        fs_value = float(resolved_params.get("fs") or 12800.0)
        nyquist = max(100.0, fs_value * 0.5)
        low = max(50.0, 0.02 * fs_value)
        high = min(0.45 * fs_value, nyquist - 50.0)
        if high <= low:
            high = min(nyquist - 1.0, low + max(20.0, 0.05 * fs_value))
        resolved_params.setdefault("filter_type", "band")
        resolved_params.setdefault("cutoff", [float(low), float(high)])
        resolved_params.setdefault("order", 4)

    # Try to get fs from state for context, if available
    fs = getattr(state, "fs", "unknown")

    for field_name, field in model_fields.items():
        # Skip fields that are already provided or are internal/not required for execution
        if field_name in resolved_params or field_name in ["op_name", "parent", "description", "input_spec", "output_spec"]:
            continue

        # If a field is required (i.e., has no default value) and is missing, generate it.
        if field.is_required():
            if field_name == "fs":
                fallback_fs = _resolve_fs_fallback(state, parent_ids or [], nodes or {})
                resolved_params["fs"] = float(fallback_fs)
                continue
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
                logger = get_current_logger()
                with timed(logger, event="llm_call", phase="builder", node="execute", message="auto-parameter generation"):
                    log_event(
                        logger,
                        level="INFO",
                        event="llm.request",
                        phase="builder",
                        node="execute",
                        message=f"Generating missing parameter: {field_name}",
                        payload={
                            "provider": os.getenv("LLM_PROVIDER"),
                            "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
                            "prompt": prompt,
                            "op_name": getattr(op_cls, "op_name", "N/A"),
                            "field_name": field_name,
                        },
                    )
                    resp = llm.invoke(prompt)
                    log_event(
                        logger,
                        level="INFO",
                        event="llm.response",
                        phase="builder",
                        node="execute",
                        message=f"Generated parameter for {field_name}",
                        payload={"response": getattr(resp, "content", "")},
                    )
                # The response should be a JSON string representing the value
                generated_value = json.loads(resp.content)
                resolved_params[field_name] = generated_value
                print(f"AI generated missing parameter '{field_name}': {generated_value}")
            except Exception as e:
                print(f"Could not generate or parse parameter '{field_name}': {e}")
                if field_name == "fs":
                    fallback_fs = _resolve_fs_fallback(state, parent_ids or [], nodes or {})
                    resolved_params["fs"] = float(fallback_fs)
                    log_event(
                        get_current_logger(),
                        level="WARNING",
                        event="execute.param_fallback",
                        phase="builder",
                        node="execute",
                        message="Falling back to deterministic fs after LLM parameter generation failure.",
                        payload={"fallback_fs": fallback_fs, "error": str(e)},
                    )
                    continue
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
    logger = get_current_logger()
    # Optional new mode: run neuro-symbolic (TSPN) training directly and return TrainReport updates.
    task_type = getattr(state, "task_type", "signal_processing_dag")
    if str(task_type).strip().lower() == "neuro_symbolic_train":
        from src.agents.deep_model_train_agent import deep_model_train_agent
        from src.agents.tspn_bootstrap_agent import tspn_bootstrap_agent

        tmp_state = state
        updates: Dict[str, Any] = {}
        log_event(
            logger,
            level="INFO",
            event="execute.neuro_symbolic.start",
            phase="executor",
            node="execute",
            message="Execute in neuro_symbolic_train mode.",
        )
        if not getattr(tmp_state, "model_config_path", None):
            boot = tspn_bootstrap_agent(tmp_state)
            updates.update(boot)
            tmp_state = tmp_state.model_copy(deep=False)
            if "model_config_path" in boot:
                tmp_state.model_config_path = boot["model_config_path"]
            if "current_model_config" in boot:
                tmp_state.current_model_config = boot["current_model_config"]

        train_out = deep_model_train_agent(tmp_state)
        updates.update(train_out)
        log_event(
            logger,
            level="INFO",
            event="execute.neuro_symbolic.finish",
            phase="executor",
            node="execute",
            message="Neuro-symbolic train mode finished.",
            payload={"update_keys": sorted(updates.keys())},
        )
        return updates

    executed_steps = 0
    llm = get_llm(None)
    log_event(
        logger,
        level="INFO",
        event="execute.start",
        phase="builder",
        node="execute",
        message="Execute detailed plan.",
        payload={"n_steps": len(state.detailed_plan)},
    )
    
    # Get the base save directory from environment, fallback to a default
    base_save_dir = (
        getattr(state, "save_dir", None)
        or os.environ.get("PHM_SAVE_DIR")
        or os.environ.get("PHM_DATA_DIR")
        or os.path.join(os.getcwd(), "save")
    )
    # Construct a case-specific directory
    case_name = state.case_name or "case"
    case_save_dir = os.path.join(base_save_dir, case_name, "nodes")

    # 采用不可变模式：创建当前节点和叶子的副本
    new_nodes = state.dag_state.nodes.copy()
    new_leaves = state.dag_state.leaves.copy()
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    operator_contract_name = (
        str(data_cfg.get("operator_contract") or "rm101_closed_v1").strip().lower() or "rm101_closed_v1"
    )
    enforce_closed_world = _parse_bool(data_cfg.get("enforce_tspn_closed_world"), default=True)
    operator_contract = resolve_operator_contract(operator_contract_name)

    for idx, step in enumerate(state.detailed_plan[:MAX_STEPS], start=1):
        op_name = step.get("op_name")
        params = step.get("params", {})
        parent_ids_str = step.get("parent")  # This might be a string like "ch1" or "ch2,ch1"
        # Backward-compat: older plans may embed parent inside params.
        if not parent_ids_str and isinstance(params, dict):
            parent_ids_str = params.get("parent")
            if "parent" in params:
                params = params.copy()
                params.pop("parent", None)

        if not parent_ids_str:
            state.dag_state.error_log.append(f"[param_error] Missing parent in step {step}")
            continue
        
        parent_ids = [pid.strip() for pid in parent_ids_str.split(',')]
        
        # Validate all parents exist
        if not all(pid in new_nodes for pid in parent_ids):
            state.dag_state.error_log.append(f"[param_error] One or more parents not found: {parent_ids} in step {step}")
            continue

        try:
            resolved_op_name, resolve_note = _resolve_operator_name(str(op_name or ""))
            if not resolved_op_name:
                failure_category = "unsupported_op"
                msg = f"Skip step index={idx}: {resolve_note}"
                state.dag_state.error_log.append(f"[{failure_category}] {msg}")
                log_event(
                    logger,
                    level="WARNING",
                    event="execute.step.skip",
                    phase="builder",
                    node="execute",
                    message=msg,
                    payload={"idx": idx, "step": step, "failure_category": failure_category},
                )
                continue
            if resolve_note:
                log_event(
                    logger,
                    level="WARNING",
                    event="execute.step.alias",
                    phase="builder",
                    node="execute",
                    message=resolve_note,
                    payload={"idx": idx, "op_name": op_name, "resolved_op_name": resolved_op_name},
                )
            if enforce_closed_world and not is_contract_allowed(resolved_op_name, operator_contract):
                failure_category = "contract_violation"
                msg = (
                    f"Stop step index={idx}: op '{resolved_op_name}' is outside "
                    f"operator_contract={operator_contract.name}"
                )
                state.dag_state.error_log.append(f"[{failure_category}] {msg}")
                log_event(
                    logger,
                    level="ERROR",
                    event="execute.step.contract_violation",
                    phase="builder",
                    node="execute",
                    message=msg,
                    payload={
                        "idx": idx,
                        "step": step,
                        "resolved_op_name": resolved_op_name,
                        "operator_contract": operator_contract.name,
                        "failure_category": failure_category,
                    },
                )
                break
            log_event(
                logger,
                level="INFO",
                event="execute.step.start",
                phase="builder",
                node="execute",
                message="Execute plan step.",
                payload={"idx": idx, "op_name": resolved_op_name, "parent": parent_ids_str},
            )
            op_cls = get_operator(resolved_op_name)
            
            # Auto-inject 'fs' if required by the operator and not provided in the original plan
            if "fs" in op_cls.model_fields and "fs" not in params:
                params["fs"] = _resolve_fs_fallback(state, parent_ids, new_nodes)
            
            # Resolve any missing required parameters using the LLM
            params = _resolve_params(llm, op_cls, params, state, parent_ids=parent_ids, nodes=new_nodes)
            op = op_cls(**params, parent=parent_ids_str)

            # --- Decoupled Execution Logic ---
            if issubclass(op_cls, MultiVariableOp):
                out_ref, out_tst = _execute_multi_variable_op(op, parent_ids, new_nodes)
            else: # --- Handle Single-Variable Operators ---
                if len(parent_ids) > 1:
                    state.dag_state.error_log.append(
                        f"Operator '{resolved_op_name}' is single-variable but received multiple parents: {parent_ids}"
                    )
                    continue
                parent_id = parent_ids[0]
                out_ref, out_tst = _execute_single_variable_op(op, parent_id, new_nodes)

            # Determine channel and new node ID
            # For multi-parent nodes, we can concatenate channel names
            channel = ",".join(sorted([new_nodes[pid].meta.get("channel", "unknown") for pid in parent_ids]))
            
            # Backward-compatible naming: abbreviate simple tokens (no underscore) to 3 chars.
            op_abbr = resolved_op_name[:3] if "_" not in resolved_op_name and len(resolved_op_name) > 3 else resolved_op_name
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
                method=resolved_op_name,
                results={"ref": out_ref, "tst": out_tst},
                meta={
                    "tool": resolved_op_name,
                    "params": params,
                    "parent": parent_ids_str,
                    "channel": channel,
                    "kind": kind,
                    "method": resolved_op_name,
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
            log_event(
                logger,
                level="INFO",
                event="execute.step.success",
                phase="builder",
                node="execute",
                message="Step executed successfully.",
                payload={"idx": idx, "node_id": node.node_id, "shape": shape},
            )
        except Exception as exc:
            failure_category = _categorize_step_failure(exc)
            state.dag_state.error_log.append(f"[{failure_category}] Error executing step {step}: {exc}")
            log_event(
                logger,
                level="ERROR",
                event="execute.step.fail",
                phase="builder",
                node="execute",
                message=f"Step failed: {exc}",
                payload={"idx": idx, "step": step, "failure_category": failure_category},
            )
            break

    # Create the new immutable DAG state
    new_dag_state = state.dag_state.model_copy(update={"nodes": new_nodes, "leaves": new_leaves})

    # 使用新的 DAGState 创建临时的 tracker 来生成图像
    temp_tracker = state.tracker()
    temp_tracker.update(new_dag_state)
    
    # Save the graph image to a case-specific directory
    case_graph_dir = os.path.join(base_save_dir, case_name, "graphs")
    os.makedirs(case_graph_dir, exist_ok=True)
    png_base = os.path.join(case_graph_dir, f"dag_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    export_ok = temp_tracker.write_png(png_base)
    png_path = f"{png_base}.png"
    dot_path = f"{png_base}.dot"
    if export_ok:
        try:
            size_bytes = os.path.getsize(png_path)
        except OSError:
            size_bytes = -1
        new_dag_state.graph_path = png_path
        log_event(
            logger,
            level="INFO",
            event="graph.export.success",
            phase="builder",
            node="execute",
            message="Exported DAG PNG.",
            payload={"path": png_path, "size_bytes": size_bytes},
        )
    else:
        new_dag_state.graph_path = dot_path if os.path.exists(dot_path) else png_path
        log_event(
            logger,
            level="ERROR",
            event="graph.export.fail",
            phase="builder",
            node="execute",
            message="Failed to export DAG PNG. DOT fallback generated.",
            payload={"png_path": png_path, "dot_path": dot_path},
        )

    log_event(
        logger,
        level="INFO",
        event="execute.finish",
        phase="builder",
        node="execute",
        message="Execute agent finished.",
        payload={"executed_steps": executed_steps, "n_nodes": len(new_nodes)},
    )

    return {"dag_state": new_dag_state, "executed_steps": executed_steps}


if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_execute_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )

    # --- 验证 ---
    assert len(updated_dag.nodes) == len(initial_nodes) + len(state.detailed_plan)
    assert "fft_01_ch1" in updated_dag.nodes
    assert "fft_02_ch2" in updated_dag.nodes
    assert "fft_03_ch3" in updated_dag.nodes
    assert updated_dag.leaves == ["fft_01_ch1", "fft_02_ch2", "fft_03_ch3", "cross_correlation_04_ch1_ch2"]
    
    print("✅ Execute Agent test passed!")
