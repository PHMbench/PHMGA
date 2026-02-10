from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.model.explainable.config_schema import TSPNConfig
from src.model.explainable.operator_catalog import (
    MAPPING_VERSION,
    UNSUPPORTED_POLICY,
    OperatorMapping,
    lookup_operator,
)
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


def _extract_band_hz(node: ProcessedData) -> Optional[Tuple[float, float]]:
    meta = getattr(node, "meta", {}) or {}
    params = meta.get("params") if isinstance(meta, dict) else None
    if not isinstance(params, dict):
        return None

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

    if str(params.get("filter_type") or "").lower() == "band":
        cutoff = params.get("cutoff")
        if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
            try:
                return float(cutoff[0]), float(cutoff[1])
            except Exception:
                return None
    return None


def _translate_operator_params(
    mapping: OperatorMapping,
    node: ProcessedData,
    *,
    fft_align_strategy: str,
) -> Dict[str, Any]:
    translated = dict(mapping.default_params or {})
    meta = getattr(node, "meta", {}) or {}
    raw_params = meta.get("params") if isinstance(meta, dict) else {}
    raw_params = raw_params if isinstance(raw_params, dict) else {}
    token = mapping.token or ""
    if token == "FFT":
        translated.setdefault("align_strategy", fft_align_strategy)
    if token == "NORM":
        method = raw_params.get("method")
        if isinstance(method, str):
            translated["method"] = method
    if token == "DT":
        detrend_type = raw_params.get("type")
        if isinstance(detrend_type, str):
            translated["type"] = detrend_type
    if token == "STFT":
        if "n_fft" in raw_params:
            translated["n_fft"] = raw_params["n_fft"]
        if "hop_length" in raw_params:
            translated["hop_length"] = raw_params["hop_length"]
        if "nperseg" in raw_params and "n_fft" not in translated:
            translated["n_fft"] = raw_params["nperseg"]
        if "noverlap" in raw_params and "hop_length" not in translated and "n_fft" in translated:
            translated["hop_length"] = max(1, int(translated["n_fft"]) - int(raw_params["noverlap"]))
    if token == "SIN":
        if "frequency" in raw_params:
            translated["frequency"] = raw_params["frequency"]
        elif "fre" in raw_params:
            translated["frequency"] = raw_params["fre"]
    return translated


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
        preserve_dag_topology: bool = True,
        allow_duplicate_tokens: bool = True,
        unsupported_policy: str = "fallback_to_identity",
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
        self.preserve_dag_topology = bool(preserve_dag_topology)
        self.allow_duplicate_tokens = bool(allow_duplicate_tokens)
        self.unsupported_policy = str(unsupported_policy).strip().lower()
        if self.unsupported_policy not in UNSUPPORTED_POLICY:
            raise ValueError(
                f"unsupported_policy={self.unsupported_policy!r} is invalid; "
                f"expected one of {sorted(UNSUPPORTED_POLICY)}"
            )

    def _build_ops_from_nodes(
        self,
        nodes: List[Tuple[ProcessedData, OperatorMapping]],
        *,
        layer_idx: int,
        dropped_nodes: List[Dict[str, Any]],
        unsupported_nodes: List[Dict[str, Any]],
        init_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        selected: List[Tuple[ProcessedData, OperatorMapping]] = list(nodes)
        if not self.allow_duplicate_tokens:
            uniq = []
            seen = set()
            for node, mapping in selected:
                token = mapping.token or ""
                if token in seen:
                    dropped_nodes.append(
                        {
                            "node_id": node.node_id,
                            "method": str(getattr(node, "method", "") or ""),
                            "reason": "duplicate token dropped",
                            "layer": layer_idx,
                        }
                    )
                    continue
                seen.add(token)
                uniq.append((node, mapping))
            selected = uniq

        if len(selected) > self.parallel_ops_per_layer:
            for node, _ in selected[self.parallel_ops_per_layer :]:
                dropped_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": str(getattr(node, "method", "") or ""),
                        "reason": "exceeds parallel_ops_per_layer",
                        "layer": layer_idx,
                    }
                )
            selected = selected[: self.parallel_ops_per_layer]

        occ: Dict[str, int] = {}
        ops: List[Dict[str, Any]] = []
        for node, mapping in selected:
            token = mapping.token
            if not token:
                unsupported_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": str(getattr(node, "method", "") or ""),
                        "status": mapping.status,
                        "reason": mapping.reason,
                    }
                )
                if self.unsupported_policy == "error":
                    raise ValueError(f"Unsupported operator node={node.node_id} method={node.method}")
                if self.unsupported_policy == "drop":
                    dropped_nodes.append(
                        {
                            "node_id": node.node_id,
                            "method": str(getattr(node, "method", "") or ""),
                            "reason": "unsupported dropped",
                            "layer": layer_idx,
                        }
                    )
                    continue
                token = "I"

            occ[token] = occ.get(token, 0) + 1
            op_uid = f"L{layer_idx}:{token}:{occ[token]-1}"
            params = _translate_operator_params(mapping, node, fft_align_strategy=self.fft_align_strategy)

            if token == "WF" and self.fs_hz is not None:
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

            if op_uid not in init_metadata["source_nodes"]:
                init_metadata["source_nodes"][op_uid] = {
                    "node_id": node.node_id,
                    "method": str(getattr(node, "method", "") or ""),
                    "status": mapping.status,
                }
            ops.append({"token": token, "params": params, "op_uid": op_uid})

        if len(ops) < self.parallel_ops_per_layer:
            i_count = sum(1 for op in ops if op["token"] == "I")
            for _ in range(self.parallel_ops_per_layer - len(ops)):
                op_uid = f"L{layer_idx}:I:{i_count}"
                i_count += 1
                ops.append({"token": "I", "params": {}, "op_uid": op_uid})

        return ops

    def _build_ops_deduplicated(
        self,
        nodes: List[Tuple[ProcessedData, OperatorMapping]],
        *,
        layer_idx: int,
        dropped_nodes: List[Dict[str, Any]],
        unsupported_nodes: List[Dict[str, Any]],
        init_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        priority = ["WF", "HT", "FFT", "NORM", "DT", "INT", "DIFF", "STFT", "LOG", "SQU", "SIN", "I"]
        selected: List[Tuple[ProcessedData, OperatorMapping]] = []
        for token in priority:
            for node, mapping in nodes:
                mapped = mapping.token or "I"
                if mapped == token and all((x[1].token or "I") != token for x in selected):
                    selected.append((node, mapping))
                    break
        return self._build_ops_from_nodes(
            selected,
            layer_idx=layer_idx,
            dropped_nodes=dropped_nodes,
            unsupported_nodes=unsupported_nodes,
            init_metadata=init_metadata,
        )

    def adapt(self, dag: DAGState) -> BridgeResult:
        depth_index = _build_depth_index(dag)
        max_depth = max(depth_index.values() or [0])
        n_layers = max(1, min(self.max_layers, max_depth))

        nodes_by_depth: Dict[int, List[Tuple[ProcessedData, OperatorMapping]]] = {}
        dropped_nodes: List[Dict[str, Any]] = []
        unsupported_nodes: List[Dict[str, Any]] = []
        for node_id, node in (dag.nodes or {}).items():
            if not isinstance(node, ProcessedData):
                continue
            d = int(depth_index.get(node_id, 0))
            if d <= 0:
                continue
            mapping = lookup_operator(str(getattr(node, "method", "") or ""))
            if mapping.status == "unsupported":
                if self.unsupported_policy == "error":
                    raise ValueError(f"Unsupported operator node={node.node_id} method={node.method}")
                if self.unsupported_policy == "drop":
                    dropped_nodes.append(
                        {
                            "node_id": node.node_id,
                            "method": str(getattr(node, "method", "") or ""),
                            "reason": "unsupported dropped",
                            "layer": d,
                        }
                    )
                    continue
            nodes_by_depth.setdefault(d, []).append((node, mapping))

        init_metadata: Dict[str, Any] = {"wf_by_op_uid": {}, "source_nodes": {}}
        layers: List[Dict[str, Any]] = []
        for layer_idx in range(1, n_layers + 1):
            nodes = nodes_by_depth.get(layer_idx, [])
            if self.preserve_dag_topology:
                ops = self._build_ops_from_nodes(
                    nodes,
                    layer_idx=layer_idx,
                    dropped_nodes=dropped_nodes,
                    unsupported_nodes=unsupported_nodes,
                    init_metadata=init_metadata,
                )
            else:
                ops = self._build_ops_deduplicated(
                    nodes,
                    layer_idx=layer_idx,
                    dropped_nodes=dropped_nodes,
                    unsupported_nodes=unsupported_nodes,
                    init_metadata=init_metadata,
                )
            layers.append({"gate_temperature": 1.0, "ops": [{"token": o["token"], "params": o["params"]} for o in ops]})

        cfg_dict: Dict[str, Any] = {
            "model": {
                "name": "tspn",
                "device": "cpu",
                "num_classes": self.num_classes,
                "in_dim": self.in_dim,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "scale": self.scale,
                "skip_connection": True,
                "wf_init": {"f_c_mu": 0.0, "f_c_sigma": 1.0, "f_b_mu": -3.0, "f_b_sigma": 0.5},
                "norm_init": {"method": "z_score"},
                "stft_init": {"n_fft": 256, "hop_length": 128},
                "sin_init": {"frequency": 1.0},
                "preserve_topology": self.preserve_dag_topology,
                "allow_duplicate_tokens": self.allow_duplicate_tokens,
                "unsupported_policy": self.unsupported_policy,
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
            "meta": {
                "bridge": {
                    "source": "DAG2ConfigAdapter",
                    "mapping_version": MAPPING_VERSION,
                    "max_depth": int(max_depth),
                    "unsupported_nodes": unsupported_nodes,
                    "dropped_nodes": dropped_nodes,
                }
            },
        }

        cfg = TSPNConfig.model_validate(cfg_dict)
        return BridgeResult(model_config=cfg.model_dump(), init_metadata=init_metadata)
