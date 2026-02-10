from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import yaml

from src.model.explainable.config_schema import TSPNConfig
from src.model.explainable.operator_catalog import lookup_operator
from src.states.phm_states import InputData, PHMState, ProcessedData


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _infer_channels_and_length(state: PHMState) -> Tuple[int, int]:
    channels = list(state.dag_state.channels or [])
    if not channels:
        raise ValueError("dag_state.channels is empty.")
    first = state.dag_state.nodes.get(channels[0])
    if not isinstance(first, InputData):
        raise ValueError("Expected InputData nodes for channel roots.")
    ref_dict = (first.results or {}).get("ref") or {}
    if not isinstance(ref_dict, dict) or not ref_dict:
        raise ValueError("InputData.results['ref'] is missing or empty.")
    first_arr = next(iter(ref_dict.values()))
    if getattr(first_arr, "shape", None) is None:
        raise ValueError("Expected numpy-like arrays for channel results.")
    if len(first_arr.shape) != 3:
        raise ValueError("Expected channel arrays with shape (1,L,1).")
    _, L, _ = first_arr.shape
    return len(channels), int(L)


def _infer_num_classes(state: PHMState) -> int:
    labels = dict(getattr(state, "labels_ref", {}) or {})
    if not labels:
        # fallback to root meta
        channels = list(state.dag_state.channels or [])
        if channels:
            root = state.dag_state.nodes.get(channels[0])
            if isinstance(root, InputData):
                labels = dict(root.meta.get("labels_ref", {}) or {})
    uniq = sorted({str(v) for v in labels.values()})
    return max(2, len(uniq))


def _build_depth_index(state: PHMState) -> Dict[str, int]:
    g = nx.DiGraph()
    for node_id, node in (state.dag_state.nodes or {}).items():
        g.add_node(node_id)
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        for p in parents:
            if p:
                g.add_edge(p, node_id)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("DAG contains a cycle; cannot bootstrap TSPN config.")

    depth: Dict[str, int] = {}
    for nid in nx.topological_sort(g):
        parents = list(g.predecessors(nid))
        depth[nid] = 0 if not parents else 1 + max(depth[p] for p in parents)
    return depth


def _map_method_to_token(method: str) -> Optional[str]:
    mapping = lookup_operator(method)
    if mapping.status == "unsupported":
        return None
    return mapping.token or "I"


def tspn_bootstrap_agent(
    state: PHMState,
    *,
    max_layers: int = 4,
    parallel_ops_per_layer: int = 4,
    out_channels: int = 3,
    scale: int = 4,
    features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically bootstrap a minimal, valid TSPN `model_config.yaml` from a built DAG.

    This is a *no-LLM* bootstrap step intended to reduce hand-designed priors:
    - infer `in_dim`, `in_channels`, `num_classes`
    - derive an initial token set from DAG node `method` semantics
    - produce a config that satisfies divisibility constraints (SPEC)
    """
    C, L = _infer_channels_and_length(state)
    num_classes = _infer_num_classes(state)

    depth_index = _build_depth_index(state)
    max_depth = max(depth_index.values() or [0])
    n_layers = max(1, min(int(max_layers), int(max_depth)))

    # Collect tokens per depth (ignore input roots).
    tokens_by_depth: Dict[int, List[str]] = {}
    for node_id, node in (state.dag_state.nodes or {}).items():
        if not isinstance(node, ProcessedData):
            continue
        d = int(depth_index.get(node_id, 0))
        if d <= 0:
            continue
        tok = _map_method_to_token(getattr(node, "method", "") or "")
        if not tok:
            continue
        tokens_by_depth.setdefault(d, []).append(tok)

    # Default feature set (align with Unified config_basic.yaml).
    feature_tokens = features or [
        "Mean",
        "Std",
        "Var",
        "Entropy",
        "Max",
        "Min",
        "AbsMean",
        "Kurtosis",
        "RMS",
        "CrestFactor",
        "Skewness",
        "ClearanceFactor",
        "ShapeFactor",
    ]

    # Build layers deterministically.
    layers: List[Dict[str, Any]] = []
    used_any_wf = False
    for layer_idx in range(1, n_layers + 1):
        present = tokens_by_depth.get(layer_idx, [])
        # deterministic unique with priority
        uniq: List[str] = []
        for t in ["WF", "HT", "FFT", "I"]:
            if t in present and t not in uniq:
                uniq.append(t)
        # Always include Identity as a safe baseline op.
        if "I" not in uniq:
            uniq.insert(0, "I")
        # Fill/pad to requested parallel op count.
        ops = (uniq + ["I"] * parallel_ops_per_layer)[:parallel_ops_per_layer]
        used_any_wf = used_any_wf or ("WF" in ops)

        layers.append(
            {
                "gate_temperature": 1.0,
                "ops": [{"token": t, "params": {}} for t in ops],
            }
        )

    # Ensure at least one WF exists (common strong baseline); inject into layer1 if missing.
    if layers and not used_any_wf:
        # Replace the first non-leading Identity slot if possible, otherwise replace the last op.
        replaced = False
        for i in range(1, len(layers[0]["ops"])):
            if layers[0]["ops"][i].get("token") == "I":
                layers[0]["ops"][i] = {"token": "WF", "params": {}}
                replaced = True
                break
        if not replaced and len(layers[0]["ops"]) >= 2:
            layers[0]["ops"][-1] = {"token": "WF", "params": {}}

    cfg_dict: Dict[str, Any] = {
        "model": {
            "name": "tspn",
            "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            "num_classes": int(num_classes),
            "in_dim": int(L),
            "in_channels": int(C),
            "out_channels": int(out_channels),
            "scale": int(scale),
            "skip_connection": True,
            # Wider center-frequency spread + narrower initial bandwidth helps
            # discover frequency-discriminative filters quickly (while keeping parameters learnable).
            "wf_init": {"f_c_mu": 0.0, "f_c_sigma": 1.0, "f_b_mu": -3.0, "f_b_sigma": 0.5},
            "layers": layers,
            "features": list(feature_tokens),
            "disabled_ops": {},
        },
        "train": {
            "seed": 42,
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "val_ratio": 0.2,
            "patience": 10,
            "debug": False,
            "debug_max_samples": 16,
            "debug_epochs": 1,
            "l1_gate": 0.0,
            "entropy_gate": 0.0,
        },
        "explain": {"topk_ops": 3, "save_wavefilters": True},
        "meta": {
            "bootstrap": {
                "source": "tspn_bootstrap_agent",
                "dag_max_depth": int(max_depth),
                "layers": int(n_layers),
                "parallel_ops_per_layer": int(parallel_ops_per_layer),
            }
        },
    }

    # Validate against schema to guarantee downstream build safety.
    cfg = TSPNConfig.model_validate(cfg_dict)

    base_save_dir = (
        getattr(state, "save_dir", None)
        or os.environ.get("PHM_SAVE_DIR")
        or os.path.join(os.getcwd(), "save")
    )
    case_name = getattr(state, "case_name", "") or "case"
    out_dir = Path(base_save_dir) / case_name / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_config_bootstrap_{_now_tag()}.yaml"
    out_path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")

    return {
        "model_config_path": str(out_path),
        "current_model_config": cfg.model_dump(),
    }
