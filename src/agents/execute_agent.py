"""Execute agent that materializes the minimal structural-prior DAG."""

from __future__ import annotations

from src.dag import DAGTracker, DagJson, DagNode
from src.data import DatasetProtocol
from src.operators import OperatorCatalog
from src.states import WorkflowState


def execute_agent(
    state: WorkflowState,
    protocol: DatasetProtocol,
    catalog: OperatorCatalog,
) -> WorkflowState:
    """Build a deterministic DAG that is valid for all three graph paths."""
    tracker = DAGTracker()
    signal_length = protocol.window.window_size
    fft_length = signal_length // 2 + 1
    for channel_index in range(protocol.samples[0].channels):
        # Each channel starts with an explicit input node so downstream plans can
        # recover channel lineage without inspecting runtime tensors.
        input_node = DagNode(
            node_id=f"input_ch{channel_index}",
            op_uid="input.signal",
            name=f"Input Channel {channel_index}",
            kind="input",
            params={"channel_index": channel_index},
            parents=[],
            in_shape=[1, signal_length],
            out_shape=[1, signal_length],
            backend_availability=["np", "pt", "sym"],
            execution_role="fixed",
        )
        tracker.add_node(input_node)

        normalize_spec = catalog.get("signal.normalize").spec
        # The transform chain is intentionally small: it yields a stable bridge
        # contract while keeping the paper-oriented DAG easy to inspect.
        normalize_node = DagNode(
            node_id=f"normalize_ch{channel_index}",
            op_uid=normalize_spec.op_uid,
            name=normalize_spec.name,
            kind="transform",
            params={"eps": 1e-6},
            parents=[input_node.node_id],
            in_shape=[1, signal_length],
            out_shape=[1, signal_length],
            backend_availability=normalize_spec.backend_availability,
            execution_role=normalize_spec.execution_role,
        )
        tracker.add_node(normalize_node)

        fft_spec = catalog.get("signal.fft_mag").spec
        fft_node = DagNode(
            node_id=f"fft_ch{channel_index}",
            op_uid=fft_spec.op_uid,
            name=fft_spec.name,
            kind="transform",
            params={},
            parents=[normalize_node.node_id],
            in_shape=[1, signal_length],
            out_shape=[1, fft_length],
            backend_availability=fft_spec.backend_availability,
            execution_role=fft_spec.execution_role,
        )
        tracker.add_node(fft_node)

        for feature_op in catalog.feature_ops():
            feature_spec = catalog.get(feature_op).spec
            suffix = feature_op.split(".")[-1]
            feature_node = DagNode(
                node_id=f"feature_ch{channel_index}_{suffix}",
                op_uid=feature_spec.op_uid,
                name=f"{feature_spec.name} Channel {channel_index}",
                kind="feature",
                params={"channel_index": channel_index},
                parents=[fft_node.node_id],
                in_shape=[1, fft_length],
                out_shape=[1],
                backend_availability=feature_spec.backend_availability,
                execution_role=feature_spec.execution_role,
            )
            tracker.add_node(feature_node)
    state.dag = tracker.export()
    state.status = "executed"
    return state
