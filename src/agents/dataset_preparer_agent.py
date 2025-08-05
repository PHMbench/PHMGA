from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np

from src.states.phm_states import PHMState, ProcessedData, DataSetNode


def _load_features_from_path(path: str) -> np.ndarray:
    """Load features from a .npy or .npz file."""
    if not path or not os.path.exists(path):
        return np.array([])
    
    if path.endswith('.npz'):
        with np.load(path) as data:
            # Each item in the npz file is a sample. We stack them.
            if not data.files:
                return np.array([])
            # Sort keys to maintain a consistent order, although not strictly necessary
            features = []
            for key in sorted(data.keys()):
                shape = data[key].shape
                features.append(np.asarray(data[key]).reshape(shape[0], -1))  # Flatten each sample
            return np.vstack(features)
    else:
        # For .npy, we assume the file contains a batch of samples.
        return np.load(path)


def dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> Dict:
    """
    Gathers features from saved files into in-memory datasets for model training.
    Handles both .npy (single array) and .npz (dictionary of features) files.
    """
    cfg = config or {}
    stage = cfg.get("stage", "processed")
    flatten = cfg.get("flatten", False) # Note: flatten is handled by ravel() in load function

    label_map = {channel: i for i, channel in enumerate(state.dag_state.channels)}
    datasets: Dict[str, Dict[str, Any]] = {}
    tracker = state.tracker()

    for node_id, node in list(state.dag_state.nodes.items()):
        if getattr(node, "stage", None) != stage:
            continue
        
        channel_name = node.meta.get("channel")
        if not channel_name:
            continue
        
        label = label_map.get(channel_name, -1)

        saved = node.meta.get("saved", {})
        X_train = _load_features_from_path(saved.get("ref_path"))
        X_test = _load_features_from_path(saved.get("tst_path"))

        if X_train.size == 0 and X_test.size == 0:
            continue

        y_train = np.full(X_train.shape[0], label, dtype=int)
        y_test = np.full(X_test.shape[0], label, dtype=int)
        
        datasets[node_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "origin_node": node_id,
        }
        ds_node = DataSetNode(
            node_id=f"ds_{node_id}",
            parents=[node_id],
            shape=X_train.shape if X_train.size else X_test.shape,
            meta={"origin_node": node_id, "channel": channel_name, "label": label},
        )
        tracker.add_node(ds_node)

    return {"datasets": datasets, "n_nodes": len(datasets)}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import DAGState, InputData

    # --- 1. Setup: Create a realistic test case ---
    save_dir = "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/test_ds"
    os.makedirs(save_dir, exist_ok=True)

    # Mock features that would be saved by execute_agent as .npz
    # B,L,C format with L=20, C=3
    features_ref1 = {
        'id1': np.random.rand(20, 3),  # L=20, C=3
        'id2': np.random.rand(20, 3),  # L=20, C=3
        'id3': np.random.rand(20, 3)   # L=20, C=3
    }
    features_tst1 = {
        'id4': np.random.rand(20, 3),  # L=20, C=3
        'id5': np.random.rand(20, 3)   # L=20, C=3
    }
    features_ref2 = {
        'id1': np.random.rand(20, 3),  # L=20, C=3
        'id2': np.random.rand(20, 3)   # L=20, C=3
    }
    features_tst2 = {
        'id4': np.random.rand(20, 3),  # L=20, C=3
        'id5': np.random.rand(20, 3),  # L=20, C=3
        'id6': np.random.rand(20, 3)   # L=20, C=3
    }
    
    # Option for B,C format (commented out)
    # features_ref1 = {'id1': np.random.rand(3), 'id2': np.random.rand(3), 'id3': np.random.rand(3)}
    # features_tst1 = {'id4': np.random.rand(3), 'id5': np.random.rand(3)}
    # features_ref2 = {'id1': np.random.rand(3), 'id2': np.random.rand(3)}
    # features_tst2 = {'id4': np.random.rand(3), 'id5': np.random.rand(3), 'id6': np.random.rand(3)}

    # Save them as .npz files
    ref1_path = os.path.join(save_dir, "ref1.npz")
    tst1_path = os.path.join(save_dir, "tst1.npz")
    ref2_path = os.path.join(save_dir, "ref2.npz")
    tst2_path = os.path.join(save_dir, "tst2.npz")
    np.savez(ref1_path, **features_ref1)
    np.savez(tst1_path, **features_tst1)
    np.savez(ref2_path, **features_ref2)
    np.savez(tst2_path, **features_tst2)

    # --- 2. Create mock nodes similar to what the graph would have ---
    proc1 = ProcessedData(
        node_id="kurtosis_ch1",
        parents=["fft_ch1"],
        stage="processed",
        source_signal_id="ch1",
        method="kurtosis",
        processed_data={},
        results={},
        meta={"channel": "ch1", "saved": {"ref_path": ref1_path, "tst_path": tst1_path}},
        shape=(),
    )
    proc2 = ProcessedData(
        node_id="kurtosis_ch2",
        parents=["fft_ch2"],
        stage="processed",
        source_signal_id="ch2",
        method="kurtosis",
        processed_data={},
        results={},
        meta={"channel": "ch2", "saved": {"ref_path": ref2_path, "tst_path": tst2_path}},
        shape=(),
    )

    # --- 3. Create the PHMState ---
    dag = DAGState(
        user_instruction="test_dataset_prep",
        channels=["ch1", "ch2"],
        nodes={"kurtosis_ch1": proc1, "kurtosis_ch2": proc2},
        leaves=["kurtosis_ch1", "kurtosis_ch2"],
    )
    state = PHMState(
        user_instruction="test_dataset_prep", 
        reference_signal=InputData(node_id="r", data={}, parents=[], shape=(0,)), 
        test_signal=InputData(node_id="t", data={}, parents=[], shape=(0,)), 
        dag_state=dag
    )

    # --- 4. Run the agent ---
    print({"before": list(state.dag_state.nodes.keys())})
    out = dataset_preparer_agent(state, config={"stage": "processed"})
    
    # --- 5. Verification ---
    print("\n--- Agent Output ---")
    print(f"Number of datasets created: {out['n_nodes']}")
    print(f"Dataset keys: {list(out['datasets'].keys())}")
    
    dataset1 = out['datasets']['kurtosis_ch1']
    print("\n--- Dataset for kurtosis_ch1 ---")
    print(f"X_train shape: {dataset1['X_train'].shape}") # Should be (3, 1)
    print(f"y_train: {dataset1['y_train']}") # Should be [0, 0, 0]
    print(f"X_test shape: {dataset1['X_test'].shape}")   # Should be (2, 1)
    print(f"y_test: {dataset1['y_test']}")     # Should be [0, 0]
    
    print("\n--- Final Nodes in DAG ---")
    print(list(state.dag_state.nodes.keys()))

    assert out['n_nodes'] == 2
    assert dataset1['X_train'].shape == (3, 1)
    assert all(dataset1['y_train'] == 0)
    assert out['datasets']['kurtosis_ch2']['X_train'].shape == (2, 1)
    assert all(out['datasets']['kurtosis_ch2']['y_train'] == 1)
    assert "ds_kurtosis_ch1" in state.dag_state.nodes

    print("\n✅ Dataset Preparer Agent test passed!")
