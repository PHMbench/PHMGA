from typing import Dict, Any, List
import numpy as np
import networkx as nx
from src.states.phm_states import PHMState, InputData, ProcessedData, normalize_result_splits
from src.tools import get_operator, MultiVariableOp
from . import _execute_multi_variable_op, _execute_single_variable_op


def _reset_inputs_and_clear_processed(
    nodes: Dict[str, InputData | ProcessedData],
    new_train: Dict[str, Any] | Any,
    new_test: Dict[str, Any] | Any,
) -> None:
    """Update InputData nodes with new signals and clear ProcessedData results."""
    for node_id, node in nodes.items():
        if isinstance(node, InputData):
            train_val = new_train.get(node_id) if isinstance(new_train, dict) else new_train
            test_val = new_test.get(node_id) if isinstance(new_test, dict) else new_test
            node.results = normalize_result_splits({"train": train_val, "val": {}, "test": test_val})
        elif isinstance(node, ProcessedData):
            node.results = None

def run_dag_on_new_data(
    state: PHMState,
) -> PHMState:
    """Re-execute an existing DAG with new training and test signals.

    Parameters
    ----------
    state : PHMState
        The state containing the DAG to be executed.
    new_train : Dict[str, Any]
        Mapping of input node IDs to new training signals.
    new_test : Dict[str, Any]
        Mapping of input node IDs to new test signals.

    Returns
    -------
    PHMState
        The updated state after running the DAG on new data.
    """

    tracker = state.tracker()
    nodes = state.dag_state.nodes
    import tqdm

    new_leaves: List[str] = []

    # --- 2. Traverse DAG in topological order ---
    for nid in tqdm.tqdm(nx.topological_sort(tracker.g)):
        node = nodes[nid]
        if isinstance(node, InputData):
            if nid not in new_leaves:
                new_leaves.append(nid)
            continue

        tool = node.meta.get("tool")
        params = node.meta.get("params", {}).copy()

        try:
            op_cls = get_operator(tool)
        except KeyError:
            state.dag_state.error_log.append(
                f"Unknown tool '{tool}' for node {nid}"
            )
            continue

        parent_ids = node.parents if isinstance(node.parents, list) else [node.parents]

        # Auto-inject sampling frequency if required
        if "fs" in getattr(op_cls, "model_fields", {}) and "fs" not in params:
            fs_val = nodes[parent_ids[0]].meta.get("fs")
            if fs_val is None:
                fs_val = getattr(state, "fs", None)
            if fs_val is not None:
                params["fs"] = fs_val

        op = op_cls(**params, parent=node.meta.get("parent"))
        print(f"Running {op_cls.__name__} on node {nid} with parents {parent_ids} and params {params}")

        if issubclass(op_cls, MultiVariableOp) and len(parent_ids) > 1:
            split_outputs = _execute_multi_variable_op(op, parent_ids, nodes)
        else:
            split_outputs = _execute_single_variable_op(op, parent_ids[0], nodes)

        node.results = normalize_result_splits(split_outputs)

        for pid in parent_ids:
            if pid in new_leaves:
                new_leaves.remove(pid)
        new_leaves.append(nid)

    # --- 3. Update DAG leaves and tracker ---
    state.dag_state.leaves = new_leaves
    tracker.update(state.dag_state)

    return state
