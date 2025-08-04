from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np

from phm_core import PHMState, ProcessedData, DataSetNode


def dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> Dict:
    """Gather feature-stage nodes into in-memory datasets.

    Parameters
    ----------
    state : PHMState
        Current PHM workflow state.
    config : Dict, optional
        Configuration containing ``stage`` to filter nodes and ``flatten`` flag.

    Returns
    -------
    Dict
        Mapping with ``datasets`` and ``n_nodes`` information.
    """
    cfg = config or {}
    stage = cfg.get("stage", "processed")
    flatten = cfg.get("flatten", False)

    # 创建从故障类型到整数标签的映射
    label_map = {channel: i for i, channel in enumerate(state.dag_state.channels)}

    datasets: Dict[str, Dict[str, Any]] = {}
    tracker = state.tracker()

    for node_id, node in list(state.dag_state.nodes.items()):
        if getattr(node, "stage", None) != stage:
            continue
        
        # 从节点元数据中获取故障类型
        channel_name = node.meta.get("channel", None)
        if not channel_name:
            continue
        
        label = label_map.get(channel_name, -1) # 如果找不到，默认为-1

        saved = getattr(node, "meta", {}).get("saved", {})
        ref_path = saved.get("ref_path")
        tst_path = saved.get("tst_path")
        X_train = np.load(ref_path) if ref_path and os.path.exists(ref_path) else np.array([])
        X_test = np.load(tst_path) if tst_path and os.path.exists(tst_path) else np.array([])
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1) if X_train.ndim > 1 else X_train
            X_test = X_test.reshape(X_test.shape[0], -1) if X_test.ndim > 1 else X_test
        
        # 使用正确的标签
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
            shape=X_train.shape if X_train.size else X_test.shape if X_test.size else (0,),
            meta={"origin_node": node_id},
        )
        tracker.add_node(ds_node)

    return {"datasets": datasets, "n_nodes": len(datasets)}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import DAGState, InputData

    instruction = "轴承故障诊断"
    sig1 = np.arange(6).reshape(2, 3)
    sig2 = np.arange(6, 12).reshape(2, 3)
    ch1 = InputData(node_id="ch1", data={"signal": sig1}, results={"ref": sig1, "tst": sig1}, parents=[], shape=sig1.shape)
    ch2 = InputData(node_id="ch2", data={"signal": sig2}, results={"ref": sig2, "tst": sig2}, parents=[], shape=sig2.shape)
    proc1 = ProcessedData(
        node_id="fft_01_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        processed_data=sig1,
        results={"ref": sig1, "tst": sig1},
        meta={"saved": {"ref_path": "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/ref1.npy", "tst_path": "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/tst1.npy"}},
        shape=sig1.shape,
    )
    proc2 = ProcessedData(
        node_id="fft_01_ch2",
        parents=["ch2"],
        source_signal_id="ch2",
        method="fft",
        processed_data=sig2,
        results={"ref": sig2, "tst": sig2},
        meta={"saved": {"ref_path": "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/ref2.npy", "tst_path": "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/tst2.npy"}},
        shape=sig2.shape,
    )
    os.makedirs("/home/lq/LQcode/2_project/PHMBench/PHMGA/save", exist_ok=True)
    np.save("/home/lq/LQcode/2_project/PHMBench/PHMGA/save/ref1.npy", sig1)
    np.save("/home/lq/LQcode/2_project/PHMBench/PHMGA/save/tst1.npy", sig1 + 1)
    np.save("/home/lq/LQcode/2_project/PHMBench/PHMGA/save/ref2.npy", sig2)
    np.save("/home/lq/LQcode/2_project/PHMBench/PHMGA/save/tst2.npy", sig2 + 1)
    dag = DAGState(
        user_instruction=instruction,
        channels=["ch1", "ch2"],
        nodes={"fft_01_ch1": proc1, "fft_01_ch2": proc2},
        leaves=[],
    )
    state = PHMState(user_instruction=instruction, reference_signal=ch1, test_signal=ch2, dag_state=dag)
    print({"before": list(state.dag_state.nodes.keys())})
    out = dataset_preparer_agent(state, config={"stage": "processed", "flatten": True})
    print({"after": out, "nodes": list(state.dag_state.nodes.keys())})
