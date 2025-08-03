import numpy as np
from typing import Dict, List

from phm_core import PHMState, ProcessedData


def _calc_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "pearson":
        r = np.corrcoef(a, b)[0, 1]
        return float(1 - r)
    raise ValueError(f"unknown metric {metric}")


def inquirer_agent(state: PHMState, metrics: List[str]) -> Dict[str, List[str]]:
    tracker = state.tracker()
    pairs: Dict[tuple, Dict[str, str]] = {}
    for nid, node in state.dag_state.nodes.items():
        if not isinstance(node, ProcessedData):
            continue
        channel = node.meta.get("channel")
        method = node.meta.get("method") or node.method
        kind = node.meta.get("kind")
        if channel is None or method is None or kind not in {"ref", "tst"}:
            continue
        key = (channel, method)
        pairs.setdefault(key, {})[kind] = nid

    new_nodes: List[str] = []
    for (channel, method), sides in pairs.items():
        ref_id, tst_id = sides.get("ref"), sides.get("tst")
        if not ref_id or not tst_id:
            state.dag_state.error_log.append(f"missing pair for {method}_{channel}")
            continue
        ref_node = state.dag_state.nodes[ref_id]
        tst_node = state.dag_state.nodes[tst_id]
        ref_data = ref_node.results.get("ref") if isinstance(ref_node.results, dict) else None
        tst_data = tst_node.results.get("tst") if isinstance(tst_node.results, dict) else None
        if ref_data is None or tst_data is None:
            state.dag_state.error_log.append(f"missing data for {method}_{channel}")
            continue
        a = np.asarray(ref_data).ravel()
        b = np.asarray(tst_data).ravel()
        if a.shape != b.shape:
            state.dag_state.error_log.append(f"shape mismatch for {method}_{channel}")
            continue
        for metric in metrics:
            try:
                val = _calc_metric(a, b, metric)
                new_id = f"sim_{metric}_{method}_{channel}"
                node = ProcessedData(
                    node_id=new_id,
                    parents=[ref_id, tst_id],
                    source_signal_id="",
                    method="similarity",
                    processed_data=val,
                    results={"sim": val},
                    meta={"metric": metric, "method": method, "channel": channel},
                    stage="similarity",
                    shape=()
                )
                tracker.add_node(node)
                new_nodes.append(new_id)
            except Exception as exc:
                state.dag_state.error_log.append(f"{metric} fail {method}_{channel}: {exc}")
    for (channel, method), _ in pairs.items():
        for metric in metrics:
            nid = f"sim_{metric}_{method}_{channel}"
            if nid not in state.dag_state.nodes:
                state.dag_state.error_log.append(f"missing node {nid}")
    return {"new_nodes": new_nodes}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import PHMState, DAGState, InputData

    sig_r1 = np.random.rand(8)
    sig_t1 = np.random.rand(8)
    sig_r2 = np.random.rand(8)
    sig_t2 = np.random.rand(8)
    ref1 = ProcessedData(
        node_id="fft_ref_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        processed_data=sig_r1,
        results={"ref": sig_r1},
        meta={"channel": "ch1", "method": "fft", "kind": "ref"},
        shape=sig_r1.shape,
    )
    tst1 = ProcessedData(
        node_id="fft_tst_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        processed_data=sig_t1,
        results={"tst": sig_t1},
        meta={"channel": "ch1", "method": "fft", "kind": "tst"},
        shape=sig_t1.shape,
    )
    ref2 = ProcessedData(
        node_id="fft_ref_ch2",
        parents=["ch2"],
        source_signal_id="ch2",
        method="fft",
        processed_data=sig_r2,
        results={"ref": sig_r2},
        meta={"channel": "ch2", "method": "fft", "kind": "ref"},
        shape=sig_r2.shape,
    )
    tst2 = ProcessedData(
        node_id="fft_tst_ch2",
        parents=["ch2"],
        source_signal_id="ch2",
        method="fft",
        processed_data=sig_t2,
        results={"tst": sig_t2},
        meta={"channel": "ch2", "method": "fft", "kind": "tst"},
        shape=sig_t2.shape,
    )
    dag = DAGState(
        user_instruction="demo",
        channels=["ch1", "ch2"],
        nodes={
            "fft_ref_ch1": ref1,
            "fft_tst_ch1": tst1,
            "fft_ref_ch2": ref2,
            "fft_tst_ch2": tst2,
        },
        leaves=["fft_ref_ch1", "fft_tst_ch1", "fft_ref_ch2", "fft_tst_ch2"],
    )
    st = PHMState(
        user_instruction="demo",
        reference_signal=InputData(node_id="r", data={}, parents=[], shape=(0,)),
        test_signal=InputData(node_id="t", data={}, parents=[], shape=(0,)),
        dag_state=dag,
    )
    before = list(st.dag_state.nodes.keys())
    result = inquirer_agent(st, metrics=["cosine"])
    after = list(st.dag_state.nodes.keys())
    print({"before": before, "after": after, "result": result})
