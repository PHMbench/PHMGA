from __future__ import annotations

import numpy as np
from ..states.phm_states import PHMState, InputData, ProcessedData
from ..tools.signal_processing_schemas import get_operator


def _get_data(state: PHMState, node_id: str) -> np.ndarray:
    node = state.dag_state.nodes.get(node_id)
    if isinstance(node, InputData):
        return np.asarray(node.data.get("signal", []))
    if isinstance(node, ProcessedData):
        return np.asarray(node.processed_data)
    raise ValueError(f"Node {node_id} not found")


def execute_agent(state: PHMState) -> dict:
    """Execute all steps in ``state.detailed_plan`` sequentially."""

    tracker = state.tracker()
    for step in state.detailed_plan:
        try:
            op_name = step.get("op_name")
            params = step.get("params", {})
            op_cls = get_operator(op_name)
            parent = params.get("parent") or params.get("parents")
            parents = parent if isinstance(parent, list) else [parent]
            data = _get_data(state, parents[0])
            op = op_cls(**{k: v for k, v in params.items() if k not in {"parent", "parents"}}, parent=parent)
            result = op.execute(data)
            new_node = ProcessedData(
                node_id=op.node_id,
                parents=parents,
                source_signal_id=parents[0],
                method=op_name,
                processed_data=result,
                shape=getattr(result, "shape", (0,)),
            )
            tracker.add_node(new_node)
        except Exception as e:
            state.error_logs.append(str(e))
            break

    return {"dag_state": state.dag_state, "error_logs": state.error_logs}


if __name__ == "__main__":
    import numpy as np
    from ..states.phm_states import DAGState, InputData

    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="demo",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
        detailed_plan=[{"op_name": "mean", "params": {"parent": "test"}}],
    )
    print("Initial nodes:", list(state.dag_state.nodes.keys()))
    result = execute_agent(state)
    print("Nodes after execution:", list(state.dag_state.nodes.keys()))
    print("Errors:", result.get("error_logs"))
