from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.model.explainable.config_schema import TSPNConfig
from src.states.phm_states import DAGState, ProcessedData


def _build_depth_index(dag: DAGState) -> Dict[str, int]:
    g = nx.DiGraph()
    for node_id, node in (dag.nodes or {}).items():
        g.add_node(node_id)
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        for p in parents:
            if p:
                g.add_edge(p, node_id)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("DAG contains a cycle; cannot adapt to TSPN config.")

    depth: Dict[str, int] = {}
    for nid in nx.topological_sort(g):
        parents = list(g.predecessors(nid))
        depth[nid] = 0 if not parents else 1 + max(depth[p] for p in parents)
    return depth


def _map_node_to_token(node: ProcessedData) -> str:
    method = str(getattr(node, "method", "") or "").lower()
    if "fft" in method:
        return "FFT"
    if "hilbert" in method or "envelope" in method or method in {"ht"}:
        return "HT"
    if "filter" in method or "bandpass" in method or "wavelet" in method or "denoise" in method or "wf" in method:
        return "WF"
    if "identity" in method or method in {"i"}:
        return "I"
    return "I"


def _extract_band_hz(node: ProcessedData) -> Optional[Tuple[float, float]]:
    meta = getattr(node, "meta", {}) or {}
    params = meta.get("params") if isinstance(meta, dict) else None
    if not isinstance(params, dict):
        return None

    # Generic forms: bands=[(low,high)] or low_hz/high_hz.
    if "low_hz" in params and "high_hz" in params:
        try:
            return float(params["low_hz"]), float(params["high_hz"])
        except Exception:
            return None

    bands = params.get("bands")
    if isinstance(bands, list) and bands:
        b0 = bands[0]
        if isinstance(b0, (list, tuple)) and len(b0) == 2:
            try:
                return float(b0[0]), float(b0[1])
            except Exception:
                return None

    # PHMGA FilterOp: filter_type + cutoff (float or (low,high))
    if str(params.get("filter_type") or "").lower() == "band":
        cutoff = params.get("cutoff")
        if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
            try:
                return float(cutoff[0]), float(cutoff[1])
            except Exception:
                return None

    return None


@dataclass(frozen=True)
class BridgeResult:
    model_config: Dict[str, Any]
    init_metadata: Dict[str, Any]


class DAG2ConfigAdapter:
    """Adapt a functional DAG to a YAML-compatible TSPN config + init metadata."""

    def __init__(
        self,
        *,
        in_dim: int,
        in_channels: int,
        num_classes: int,
        fs_hz: float | None = None,
        max_layers: int = 4,
        parallel_ops_per_layer: int = 4,
        out_channels: int = 3,
        scale: int = 4,
        feature_tokens: Optional[List[str]] = None,
        fft_align_strategy: str = "interp",
    ):
        self.in_dim = int(in_dim)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.fs_hz = float(fs_hz) if fs_hz is not None else None
        self.max_layers = int(max_layers)
        self.parallel_ops_per_layer = int(parallel_ops_per_layer)
        self.out_channels = int(out_channels)
        self.scale = int(scale)
        self.feature_tokens = feature_tokens or ["Mean", "Std", "RMS"]
        self.fft_align_strategy = str(fft_align_strategy).strip().lower()

    def adapt(self, dag: DAGState) -> BridgeResult:
        depth_index = _build_depth_index(dag)
        max_depth = max(depth_index.values() or [0])
        n_layers = max(1, min(self.max_layers, max_depth))

        tokens_by_depth: Dict[int, List[Tuple[str, ProcessedData]]] = {}
        for node_id, node in (dag.nodes or {}).items():
            if not isinstance(node, ProcessedData):
                continue
            d = int(depth_index.get(node_id, 0))
            if d <= 0:
                continue
            tok = _map_node_to_token(node)
            tokens_by_depth.setdefault(d, []).append((tok, node))

        init_metadata: Dict[str, Any] = {"wf_by_op_uid": {}, "source_nodes": {}}
        layers: List[Dict[str, Any]] = []

        for layer_idx in range(1, n_layers + 1):
            present = tokens_by_depth.get(layer_idx, [])
            # deterministic unique with priority
            uniq: List[Tuple[str, Optional[ProcessedData]]] = []
            for t in ["WF", "HT", "FFT", "I"]:
                for tok, node in present:
                    if tok == t and all(u[0] != t for u in uniq):
                        uniq.append((tok, node))
                        break
            if not any(t == "I" for t, _ in uniq):
                uniq.insert(0, ("I", None))

            ops: List[Dict[str, Any]] = []
            occ: Dict[str, int] = {}
            for tok, node in (uniq + [("I", None)] * self.parallel_ops_per_layer)[: self.parallel_ops_per_layer]:
                occ[tok] = occ.get(tok, 0) + 1
                op_uid = f"L{layer_idx}:{tok}:{occ[tok]-1}"
                params: Dict[str, Any] = {}
                if tok == "FFT":
                    params["align_strategy"] = self.fft_align_strategy

                if tok == "WF" and node is not None and self.fs_hz is not None:
                    band = _extract_band_hz(node)
                    if band is not None:
                        low, high = band
                        fc_hz = 0.5 * (low + high)
                        fb_hz = 0.5 * abs(high - low)
                        init_metadata["wf_by_op_uid"][op_uid] = {
                            "fc_hz": float(fc_hz),
                            "fb_hz": float(fb_hz),
                            "fs_hz": float(self.fs_hz),
                        }
                        init_metadata["source_nodes"][op_uid] = {"node_id": node.node_id, "band_hz": [low, high]}

                ops.append({"token": tok, "params": params, "op_uid": op_uid})

            layers.append({"gate_temperature": 1.0, "ops": [{"token": o["token"], "params": o["params"]} for o in ops]})

        cfg_dict: Dict[str, Any] = {
            "model": {
                "name": "tspn",
                "device": "cuda" if False else "cpu",
                "num_classes": self.num_classes,
                "in_dim": self.in_dim,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "scale": self.scale,
                "skip_connection": True,
                "wf_init": {"f_c_mu": 0.0, "f_c_sigma": 1.0, "f_b_mu": -3.0, "f_b_sigma": 0.5},
                "layers": layers,
                "features": list(self.feature_tokens),
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
            "meta": {"bridge": {"source": "DAG2ConfigAdapter", "max_depth": int(max_depth)}},
        }

        cfg = TSPNConfig.model_validate(cfg_dict)
        return BridgeResult(model_config=cfg.model_dump(), init_metadata=init_metadata)

