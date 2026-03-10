from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.model.explainable.config_schema import TSPNConfig
from src.model.explainable.operator_catalog import (
    MAPPING_VERSION,
    UNSUPPORTED_POLICY,
    CompatibilityProfile,
    OperatorContract,
    contract_method_kind,
    is_contract_allowed,
    OperatorMapping,
    lookup_feature,
    lookup_operator,
    normalize_method_name,
    resolve_compatibility_profile,
    resolve_operator_contract,
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
        min_effective_ops_ratio: float = 0.0,
        compat_profile: str | None = None,
        operator_contract: str | None = None,
        enforce_tspn_closed_world: bool = True,
    ):
        self.in_dim = int(in_dim)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.fs_hz = float(fs_hz) if fs_hz is not None else None
        self.max_layers = int(max_layers)
        self.parallel_ops_per_layer = int(parallel_ops_per_layer)
        self.out_channels = int(out_channels)
        self.scale = int(scale)
        self.feature_tokens = list(feature_tokens) if feature_tokens else None
        self.fft_align_strategy = str(fft_align_strategy).strip().lower()
        self.preserve_dag_topology = bool(preserve_dag_topology)
        self.allow_duplicate_tokens = bool(allow_duplicate_tokens)
        self.unsupported_policy = str(unsupported_policy).strip().lower()
        self.min_effective_ops_ratio = float(min_effective_ops_ratio)
        self.compat_profile: CompatibilityProfile = resolve_compatibility_profile(compat_profile)
        self.operator_contract: OperatorContract = resolve_operator_contract(operator_contract)
        self.enforce_tspn_closed_world = bool(enforce_tspn_closed_world)
        if self.min_effective_ops_ratio < 0.0 or self.min_effective_ops_ratio > 1.0:
            raise ValueError("min_effective_ops_ratio must be within [0, 1].")
        if self.unsupported_policy not in UNSUPPORTED_POLICY:
            raise ValueError(
                f"unsupported_policy={self.unsupported_policy!r} is invalid; "
                f"expected one of {sorted(UNSUPPORTED_POLICY)}"
            )

    def _infer_feature_tokens_from_dag(self, dag: DAGState) -> List[str]:
        supported_tokens = {
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
            "ClearanceFactor",
            "Skewness",
            "ShapeFactor",
            "SpectralKurtosis",
            "PeakToPeak",
            "ZeroCrossingRate",
            "SpectralCentroid",
            "SpectralSkewness",
            "SpectralFlatness",
            "HjorthActivity",
            "HjorthMobility",
            "HjorthComplexity",
        }
        if self.feature_tokens:
            filtered = [token for token in self.feature_tokens if token in supported_tokens]
            return filtered if filtered else ["Mean", "Std", "RMS"]

        inferred: List[str] = []
        for node in (dag.nodes or {}).values():
            if not isinstance(node, ProcessedData):
                continue
            method = str(getattr(node, "method", "") or "")
            mapping = lookup_feature(method)
            token = mapping.feature_token
            if method == "hjorth_parameters":
                for hjorth_token in ("HjorthActivity", "HjorthMobility", "HjorthComplexity"):
                    if hjorth_token in supported_tokens and hjorth_token not in inferred:
                        inferred.append(hjorth_token)
                continue
            if token and token in supported_tokens and token not in inferred:
                inferred.append(token)
        if not inferred:
            inferred = ["Mean", "Std", "RMS"]
        return inferred

    def _is_profile_strict_method(self, method_key: str) -> bool:
        strict = self.compat_profile.strict_methods
        return bool(strict) and method_key in strict

    def _is_aggregate_method(self, method_key: str) -> bool:
        return method_key in self.compat_profile.aggregate_methods

    def _record_compat_issue(
        self,
        issues: List[Dict[str, Any]],
        *,
        node: ProcessedData,
        method: str,
        category: str,
        reason: str,
        status: str | None = None,
        layer: int | None = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "node_id": str(getattr(node, "node_id", "") or ""),
            "method": method,
            "category": category,
            "reason": reason,
        }
        if status:
            entry["status"] = status
        if layer is not None:
            entry["layer"] = int(layer)
        issues.append(entry)

    def build_contract_violation_report(self, dag: DAGState) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        for node in (dag.nodes or {}).values():
            if not isinstance(node, ProcessedData):
                continue
            method = str(getattr(node, "method", "") or "")
            method_key = normalize_method_name(method)
            if not method_key:
                continue
            if is_contract_allowed(method_key, self.operator_contract):
                continue
            violations.append(
                {
                    "node_id": str(getattr(node, "node_id", "") or ""),
                    "method": method_key,
                    "method_kind": contract_method_kind(method_key),
                    "reason": f"outside operator_contract={self.operator_contract.name}",
                }
            )
        return {
            "operator_contract": self.operator_contract.name,
            "enforce_tspn_closed_world": bool(self.enforce_tspn_closed_world),
            "pass": len(violations) == 0,
            "violations_count": int(len(violations)),
            "violations": violations,
        }

    def _build_ops_from_nodes(
        self,
        nodes: List[Tuple[ProcessedData, OperatorMapping]],
        *,
        layer_idx: int,
        dropped_nodes: List[Dict[str, Any]],
        proxy_nodes: List[Dict[str, Any]],
        unsupported_nodes: List[Dict[str, Any]],
        compatibility_issues: List[Dict[str, Any]],
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
            method = str(getattr(node, "method", "") or "")
            method_key = normalize_method_name(method)
            token = mapping.token
            if mapping.status == "proxy":
                proxy_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": method,
                        "status": mapping.status,
                        "reason": mapping.reason,
                    }
                )
                self._record_compat_issue(
                    compatibility_issues,
                    node=node,
                    method=method,
                    category="proxy",
                    reason=mapping.reason or "proxy mapping",
                    status=mapping.status,
                    layer=layer_idx,
                )
                if self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_proxy:
                    raise ValueError(
                        f"[{self.compat_profile.name}] proxy operator is forbidden: "
                        f"node={node.node_id} method={method}"
                    )
            if token == "I" and method_key in self.compat_profile.disallow_identity_for:
                self._record_compat_issue(
                    compatibility_issues,
                    node=node,
                    method=method,
                    category="identity_fallback",
                    reason="mapped to identity token",
                    status=mapping.status,
                    layer=layer_idx,
                )
                if self._is_profile_strict_method(method_key):
                    raise ValueError(
                        f"[{self.compat_profile.name}] identity fallback is forbidden: "
                        f"node={node.node_id} method={method}"
                    )
            if not token:
                unsupported_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": method,
                        "status": mapping.status,
                        "reason": mapping.reason,
                    }
                )
                self._record_compat_issue(
                    compatibility_issues,
                    node=node,
                    method=method,
                    category="unsupported",
                    reason=mapping.reason or "unsupported operator",
                    status=mapping.status,
                    layer=layer_idx,
                )
                if self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_unsupported:
                    raise ValueError(
                        f"[{self.compat_profile.name}] unsupported operator is forbidden: "
                        f"node={node.node_id} method={method}"
                    )
                if self.unsupported_policy == "error":
                    raise ValueError(f"Unsupported operator node={node.node_id} method={node.method}")
                if self.unsupported_policy == "drop":
                    dropped_nodes.append(
                        {
                            "node_id": node.node_id,
                            "method": method,
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
        proxy_nodes: List[Dict[str, Any]],
        unsupported_nodes: List[Dict[str, Any]],
        compatibility_issues: List[Dict[str, Any]],
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
            proxy_nodes=proxy_nodes,
            unsupported_nodes=unsupported_nodes,
            compatibility_issues=compatibility_issues,
            init_metadata=init_metadata,
        )

    def adapt(self, dag: DAGState) -> BridgeResult:
        contract_violation_report = self.build_contract_violation_report(dag)
        contract_violations = list(contract_violation_report.get("violations") or [])
        contract_violations_count = int(contract_violation_report.get("violations_count") or len(contract_violations))
        if self.enforce_tspn_closed_world and contract_violations_count > 0:
            first = contract_violations[0]
            raise ValueError(
                f"[{self.operator_contract.name}] contract violation: "
                f"node={first.get('node_id')} method={first.get('method')}"
            )

        depth_index = _build_depth_index(dag)
        max_depth = max(depth_index.values() or [0])
        n_layers = max(1, min(self.max_layers, max_depth))

        nodes_by_depth: Dict[int, List[Tuple[ProcessedData, OperatorMapping]]] = {}
        dropped_nodes: List[Dict[str, Any]] = []
        proxy_nodes: List[Dict[str, Any]] = []
        unsupported_nodes: List[Dict[str, Any]] = []
        aggregate_feature_nodes: List[Dict[str, Any]] = []
        aggregate_unsupported_nodes: List[Dict[str, Any]] = []
        compatibility_issues: List[Dict[str, Any]] = []
        feature_tokens_from_aggregate: Set[str] = set()
        for node_id, node in (dag.nodes or {}).items():
            if not isinstance(node, ProcessedData):
                continue
            d = int(depth_index.get(node_id, 0))
            if d <= 0:
                continue
            method = str(getattr(node, "method", "") or "")
            method_key = normalize_method_name(method)
            feature_mapping = lookup_feature(method_key)

            if self._is_aggregate_method(method_key):
                aggregate_entry: Dict[str, Any] = {
                    "node_id": node.node_id,
                    "method": method,
                    "status": feature_mapping.status,
                    "layer": d,
                }
                if method_key == "hjorth_parameters":
                    for token in ("HjorthActivity", "HjorthMobility", "HjorthComplexity"):
                        feature_tokens_from_aggregate.add(token)
                elif feature_mapping.feature_token:
                    feature_tokens_from_aggregate.add(str(feature_mapping.feature_token))

                if feature_mapping.status == "unsupported":
                    aggregate_entry["reason"] = feature_mapping.reason
                    aggregate_entry["route"] = "unsupported"
                    aggregate_unsupported_nodes.append(aggregate_entry)
                    self._record_compat_issue(
                        compatibility_issues,
                        node=node,
                        method=method,
                        category="unsupported_feature",
                        reason=feature_mapping.reason or "unsupported aggregate feature op",
                        status=feature_mapping.status,
                        layer=d,
                    )
                    if self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_unsupported:
                        raise ValueError(
                            f"[{self.compat_profile.name}] unsupported aggregate feature op: "
                            f"node={node.node_id} method={method}"
                        )
                else:
                    aggregate_entry["route"] = "feature"
                    if method_key == "hjorth_parameters":
                        aggregate_entry["feature_tokens"] = ["HjorthActivity", "HjorthMobility", "HjorthComplexity"]
                    elif feature_mapping.feature_token:
                        aggregate_entry["feature_token"] = str(feature_mapping.feature_token)
                    aggregate_feature_nodes.append(aggregate_entry)
                    if feature_mapping.status == "proxy":
                        self._record_compat_issue(
                            compatibility_issues,
                            node=node,
                            method=method,
                            category="proxy_feature",
                            reason=feature_mapping.reason or "proxy aggregate feature op",
                            status=feature_mapping.status,
                            layer=d,
                        )
                        if self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_proxy:
                            raise ValueError(
                                f"[{self.compat_profile.name}] proxy aggregate feature op is forbidden: "
                                f"node={node.node_id} method={method}"
                            )
                continue

            mapping = lookup_operator(method_key)
            if mapping.status == "unsupported" and self.unsupported_policy == "drop":
                unsupported_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": method,
                        "status": mapping.status,
                        "reason": mapping.reason,
                    }
                )
                dropped_nodes.append(
                    {
                        "node_id": node.node_id,
                        "method": method,
                        "reason": "unsupported dropped",
                        "layer": d,
                    }
                )
                self._record_compat_issue(
                    compatibility_issues,
                    node=node,
                    method=method,
                    category="unsupported",
                    reason=mapping.reason or "unsupported operator",
                    status=mapping.status,
                    layer=d,
                )
                if self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_unsupported:
                    raise ValueError(
                        f"[{self.compat_profile.name}] unsupported operator is forbidden: "
                        f"node={node.node_id} method={method}"
                    )
                continue

            if mapping.status == "proxy" and self._is_profile_strict_method(method_key) and self.compat_profile.fail_on_proxy:
                raise ValueError(
                    f"[{self.compat_profile.name}] proxy operator is forbidden: "
                    f"node={node.node_id} method={method}"
                )
            if (
                mapping.status == "unsupported"
                and self._is_profile_strict_method(method_key)
                and self.compat_profile.fail_on_unsupported
            ):
                raise ValueError(
                    f"[{self.compat_profile.name}] unsupported operator is forbidden: "
                    f"node={node.node_id} method={method}"
                )
            if (
                mapping.token == "I"
                and method_key in self.compat_profile.disallow_identity_for
                and self._is_profile_strict_method(method_key)
            ):
                raise ValueError(
                    f"[{self.compat_profile.name}] identity fallback is forbidden: "
                    f"node={node.node_id} method={method}"
                )
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
                    proxy_nodes=proxy_nodes,
                    unsupported_nodes=unsupported_nodes,
                    compatibility_issues=compatibility_issues,
                    init_metadata=init_metadata,
                )
            else:
                ops = self._build_ops_deduplicated(
                    nodes,
                    layer_idx=layer_idx,
                    dropped_nodes=dropped_nodes,
                    proxy_nodes=proxy_nodes,
                    unsupported_nodes=unsupported_nodes,
                    compatibility_issues=compatibility_issues,
                    init_metadata=init_metadata,
                )
            layers.append({"gate_temperature": 1.0, "ops": [{"token": o["token"], "params": o["params"]} for o in ops]})

        feature_tokens = self._infer_feature_tokens_from_dag(dag)
        for token in sorted(feature_tokens_from_aggregate):
            if token and token not in feature_tokens:
                feature_tokens.append(token)
        layer_token_lists = [[str(op.get("token") or "") for op in layer.get("ops", [])] for layer in layers]
        total_ops = int(sum(len(tokens) for tokens in layer_token_lists))
        non_identity_ops = int(sum(1 for tokens in layer_token_lists for token in tokens if token != "I"))
        identity_ops = int(total_ops - non_identity_ops)
        effective_ops_ratio = float(non_identity_ops / max(1, total_ops))
        strict_mode = bool(self.compat_profile.strict_methods)
        if strict_mode and non_identity_ops <= 0:
            raise ValueError(
                f"[{self.compat_profile.name}] no effective non-identity operators after bridge adaptation."
            )
        bridge_warnings: List[str] = []
        if effective_ops_ratio < self.min_effective_ops_ratio:
            bridge_warnings.append(
                f"effective_ops_ratio={effective_ops_ratio:.4f} below min_effective_ops_ratio={self.min_effective_ops_ratio:.4f}"
            )

        bridge_quality = {
            "total_ops_count": int(total_ops),
            "effective_ops_count": int(non_identity_ops),
            "identity_ops_count": int(identity_ops),
            "effective_ops_ratio": float(effective_ops_ratio),
            "unsupported_nodes_count": int(len(unsupported_nodes)),
            "dropped_nodes_count": int(len(dropped_nodes)),
            "min_effective_ops_ratio": float(self.min_effective_ops_ratio),
            "warnings": bridge_warnings,
        }
        init_metadata["bridge_quality"] = bridge_quality

        identity_fallback_nodes_count = int(
            sum(1 for item in compatibility_issues if str(item.get("category") or "") == "identity_fallback")
        )
        unsupported_total = int(len(unsupported_nodes) + len(aggregate_unsupported_nodes))
        compatibility_warnings: List[str] = list(bridge_warnings)
        if aggregate_unsupported_nodes:
            compatibility_warnings.append(
                f"aggregate_unsupported_nodes={len(aggregate_unsupported_nodes)} routed out of layer ops"
            )
        compatibility_pass = effective_ops_ratio > 0.0
        if strict_mode:
            compatibility_pass = (
                compatibility_pass
                and len(proxy_nodes) == 0
                and unsupported_total == 0
                and identity_fallback_nodes_count == 0
            )
            if not compatibility_pass:
                compatibility_warnings.append("strict compatibility requirements not satisfied")

        compatibility_quality = {
            "profile": self.compat_profile.name,
            "strict_mode": strict_mode,
            "pass": bool(compatibility_pass),
            "effective_ops_ratio": float(effective_ops_ratio),
            "effective_ops_count": int(non_identity_ops),
            "identity_ops_count": int(identity_ops),
            "proxy_nodes_count": int(len(proxy_nodes)),
            "unsupported_nodes_count": int(unsupported_total),
            "aggregate_feature_nodes_count": int(len(aggregate_feature_nodes)),
            "aggregate_unsupported_nodes_count": int(len(aggregate_unsupported_nodes)),
            "identity_fallback_nodes_count": int(identity_fallback_nodes_count),
            "fail_fast_rules": {
                "fail_on_proxy": bool(self.compat_profile.fail_on_proxy),
                "fail_on_unsupported": bool(self.compat_profile.fail_on_unsupported),
                "disallow_identity_for": sorted(self.compat_profile.disallow_identity_for),
            },
            "warnings": compatibility_warnings,
        }
        compile_warnings: List[str] = list(compatibility_warnings)
        if contract_violations_count > 0:
            compile_warnings.append(
                f"contract_violations_count={contract_violations_count} for operator_contract={self.operator_contract.name}"
            )
        closed_world_pass = (
            contract_violations_count == 0
            and effective_ops_ratio > 0.0
            and len(proxy_nodes) == 0
            and unsupported_total == 0
            and identity_fallback_nodes_count == 0
        )
        compile_quality = {
            "operator_contract": self.operator_contract.name,
            "enforce_tspn_closed_world": bool(self.enforce_tspn_closed_world),
            "closed_world_pass": bool(closed_world_pass),
            "effective_ops_ratio": float(effective_ops_ratio),
            "effective_ops_count": int(non_identity_ops),
            "identity_ops_count": int(identity_ops),
            "proxy_nodes_count": int(len(proxy_nodes)),
            "unsupported_nodes_count": int(unsupported_total),
            "identity_fallback_nodes_count": int(identity_fallback_nodes_count),
            "contract_violations_count": int(contract_violations_count),
            "warnings": compile_warnings,
        }
        dag_compile_report = {
            "operator_contract": self.operator_contract.name,
            "compat_profile": self.compat_profile.name,
            "closed_world_pass": bool(closed_world_pass),
            "compile_quality": compile_quality,
            "bridge_quality": bridge_quality,
            "compatibility_quality": compatibility_quality,
            "contract_violation_report": contract_violation_report,
        }
        compatibility_report = {
            "profile": self.compat_profile.name,
            "strict_mode": strict_mode,
            "proxy_nodes": proxy_nodes,
            "unsupported_nodes": unsupported_nodes,
            "aggregate_feature_nodes": aggregate_feature_nodes,
            "aggregate_unsupported_nodes": aggregate_unsupported_nodes,
            "dropped_nodes": dropped_nodes,
            "issues": compatibility_issues,
            "bridge_quality": bridge_quality,
            "compatibility_quality": compatibility_quality,
            "compile_quality": compile_quality,
        }
        init_metadata["compatibility_quality"] = compatibility_quality
        init_metadata["compatibility_report"] = compatibility_report
        init_metadata["contract_violation_report"] = contract_violation_report
        init_metadata["compile_quality"] = compile_quality
        init_metadata["dag_compile_report"] = dag_compile_report

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
                "bridge": {
                    "source": "DAG2ConfigAdapter",
                    "mapping_version": MAPPING_VERSION,
                    "compat_profile": self.compat_profile.name,
                    "operator_contract": self.operator_contract.name,
                    "enforce_tspn_closed_world": bool(self.enforce_tspn_closed_world),
                    "max_depth": int(max_depth),
                    "proxy_nodes": proxy_nodes,
                    "unsupported_nodes": unsupported_nodes,
                    "aggregate_feature_nodes": aggregate_feature_nodes,
                    "aggregate_unsupported_nodes": aggregate_unsupported_nodes,
                    "dropped_nodes": dropped_nodes,
                    "bridge_quality": bridge_quality,
                    "compatibility_quality": compatibility_quality,
                    "compile_quality": compile_quality,
                }
            },
        }

        cfg = TSPNConfig.model_validate(cfg_dict)
        return BridgeResult(model_config=cfg.model_dump(), init_metadata=init_metadata)
