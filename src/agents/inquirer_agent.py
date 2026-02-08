import numpy as np
from typing import Any, Dict, List, Tuple

from src.states.phm_states import PHMState, ProcessedData, InputData


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
    """
    Similarity analysis between reference and test signals.

    Two supported data layouts:
    1) Legacy: each leaf node contains both ``results['ref']`` and ``results['tst']`` dicts.
       In this case, we populate ``node.sim`` in-place and do not create new nodes.
    2) Paired leaves: separate ref/tst nodes per (channel, method), with ``meta.kind`` in {"ref","tst"}.
       In this case, we create new similarity nodes (stage="similarity") and return their ids.
    """
    leaf_ids = list(state.dag_state.leaves)
    print(f"Calculating similarity for {len(leaf_ids)} leaf nodes with metrics: {metrics}")

    new_node_ids: List[str] = []

    def _as_array(x: Any) -> np.ndarray | None:
        if x is None:
            return None
        if isinstance(x, dict):
            if not x:
                return None
            x = next(iter(x.values()))
        try:
            return np.asarray(x).ravel()
        except Exception:
            return None

    # --- Path A: single node has both ref/tst dicts ---
    for leaf_id in leaf_ids:
        node = state.dag_state.nodes.get(leaf_id)
        if not isinstance(node, (ProcessedData, InputData)) or not node.results:
            continue

        ref_data_dict = node.results.get("ref")
        tst_data_dict = node.results.get("tst")
        if not (isinstance(ref_data_dict, dict) and isinstance(tst_data_dict, dict)):
            continue

        node.sim = {}
        for metric in metrics:
            sim_matrix: Dict[str, Dict[str, float]] = {}
            for ref_key, ref_val in ref_data_dict.items():
                sim_matrix[ref_key] = {}
                a = np.asarray(ref_val).ravel()

                for tst_key, tst_val in tst_data_dict.items():
                    b = np.asarray(tst_val).ravel()
                    if a.shape != b.shape:
                        state.dag_state.error_log.append(
                            f"Shape mismatch between {ref_key} and {tst_key} in node {leaf_id}"
                        )
                        continue
                    try:
                        sim_matrix[ref_key][tst_key] = _calc_metric(a, b, metric)
                    except Exception as exc:
                        state.dag_state.error_log.append(f"{metric} fail between {ref_key} and {tst_key}: {exc}")
            node.sim[metric] = sim_matrix

    # --- Path B: paired ref/tst leaves, create similarity nodes ---
    groups: Dict[Tuple[str, str], Dict[str, str]] = {}
    for leaf_id in leaf_ids:
        node = state.dag_state.nodes.get(leaf_id)
        if not isinstance(node, ProcessedData) or not node.results:
            continue
        meta = node.meta or {}
        kind = meta.get("kind")
        channel = meta.get("channel")
        method = meta.get("method") or getattr(node, "method", None)
        if kind not in {"ref", "tst"} or not channel or not method:
            continue
        groups.setdefault((str(channel), str(method)), {})[str(kind)] = leaf_id

    tracker = state.tracker()
    for (channel, method), pair in groups.items():
        if "ref" not in pair or "tst" not in pair:
            continue
        ref_node = state.dag_state.nodes[pair["ref"]]
        tst_node = state.dag_state.nodes[pair["tst"]]
        a = _as_array((ref_node.results or {}).get("ref"))
        b = _as_array((tst_node.results or {}).get("tst"))
        if a is None or b is None or a.shape != b.shape:
            continue

        for metric in metrics:
            try:
                val = float(_calc_metric(a, b, metric))
            except Exception:
                continue
            sim_node = ProcessedData(
                node_id=f"sim_{metric}_{method}_{channel}",
                parents=[pair["ref"], pair["tst"]],
                source_signal_id=f"{pair['ref']},{pair['tst']}",
                method=f"sim_{metric}",
                results={"sim": val},
                meta={"channel": channel, "method": method, "metric": metric, "kind": "similarity"},
                stage="similarity",
                shape=(),
            )
            new_node_ids.append(tracker.add_node(sim_node))

    return {"new_nodes": new_node_ids}


if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_inquirer_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )
