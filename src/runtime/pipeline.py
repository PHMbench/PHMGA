"""Main thesis_2026 runtime: preflight, DAG build, feature materialization, fusion, artifacts."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.agents.execute_agent import execute_agent
from src.agents.plan_agent import plan_agent
from src.agents.reflect_agent import reflect_agent_node
from src.configuration import Configuration
from src.data import build_protocol_from_config, materialize_preview_pair, materialize_split_signals
from src.llm import LLMBackendError, get_llm
from src.runtime.feature_plan import FeaturePlan, build_feature_plan, materialize_feature_views
from src.states.phm_states import DAGState, InputData, PHMState
from src.utils import get_dag_depth

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "model_dump"):
            return value.model_dump()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def _preview_state(protocol, runtime_config: Dict[str, Any]) -> PHMState:
    ref_record, tst_record = materialize_preview_pair(protocol)
    channels = [f"ch{index + 1}" for index in range(ref_record.window.shape[0])]
    nodes: Dict[str, InputData] = {}
    for index, channel_name in enumerate(channels):
        ref_window = ref_record.window[index : index + 1, :].T.reshape(1, ref_record.window.shape[1], 1)
        tst_window = tst_record.window[index : index + 1, :].T.reshape(1, tst_record.window.shape[1], 1)
        nodes[channel_name] = InputData(
            node_id=channel_name,
            results={
                "ref": {ref_record.window_id: ref_window},
                "tst": {tst_record.window_id: tst_window},
            },
            parents=[],
            shape=ref_window.shape,
            meta={
                "channel": channel_name,
                "labels": {
                    ref_record.window_id: int(ref_record.label),
                    tst_record.window_id: int(tst_record.label),
                },
                "fs": int(protocol.samples[0].sampling_rate),
            },
        )

    dag_state = DAGState(
        user_instruction=str(runtime_config["experiment"]["user_instruction"]),
        channels=channels,
        nodes=nodes,
        leaves=list(channels),
    )
    state = PHMState(
        case_name=str(runtime_config["runtime"]["run_name"]),
        user_instruction=str(runtime_config["experiment"]["user_instruction"]),
        reference_signal=nodes[channels[0]],
        test_signal=nodes[channels[0]],
        dag_state=dag_state,
        min_depth=int(runtime_config["runtime"]["min_depth"]),
        min_width=int(runtime_config["runtime"]["min_width"]),
        max_depth=int(runtime_config["runtime"]["max_depth"]),
        fs=int(protocol.samples[0].sampling_rate),
        runtime_config=deepcopy(runtime_config),
    )
    return state


def _build_dag(state: PHMState) -> PHMState:
    max_iterations = int(state.runtime_config.get("runtime", {}).get("max_iterations", 4))
    for _ in range(max_iterations):
        for key, value in plan_agent(state).items():
            if key in state.model_fields:
                setattr(state, key, value)
        for key, value in execute_agent(state).items():
            if key in state.model_fields:
                setattr(state, key, value)
        reflection = reflect_agent_node(state, stage="POST_EXECUTE")
        state.needs_revision = bool(reflection.get("needs_revision", False))
        state.reflection_history = list(reflection.get("reflection_history", state.reflection_history))
        state.last_reflection_decision = str(reflection.get("decision", "halt"))
        depth = get_dag_depth(state.dag_state)
        if depth < state.min_depth:
            state.needs_revision = True
            if state.last_reflection_decision == "finish":
                state.last_reflection_decision = "need_patch"
        if state.last_reflection_decision == "halt":
            break
        if not state.needs_revision:
            break
    return state


def _softmax(scores: List[float]) -> np.ndarray:
    if not scores:
        return np.zeros((0,), dtype=float)
    arr = np.asarray(scores, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp)
    if denom <= 0:
        return np.ones_like(arr) / len(arr)
    return exp / denom


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_branch_classifier(x_train: np.ndarray, y_train: np.ndarray):
    estimator = LogisticRegression(max_iter=200, random_state=0)
    estimator.fit(x_train, y_train)
    return estimator


def _run_ml_backend(feature_plan: FeaturePlan, feature_views: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    branch_ids = [branch.node_id for branch in feature_plan.branches]
    branch_models: Dict[str, Any] = {}
    branch_scores: List[float] = []
    class_count = int(len(np.unique(feature_views["train"]["labels"])))
    if class_count < 2:
        raise ValueError("ML backend requires at least two classes in the training split.")

    for branch_id in branch_ids:
        x_train = feature_views["train"]["branches"][branch_id]
        y_train = feature_views["train"]["labels"]
        x_val = feature_views["val"]["branches"][branch_id]
        y_val = feature_views["val"]["labels"]
        estimator = _fit_branch_classifier(x_train, y_train)
        val_pred = estimator.predict(x_val)
        score = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        branch_models[branch_id] = estimator
        branch_scores.append(score)

    weights = _softmax(branch_scores)
    branch_weights = {branch_id: float(weight) for branch_id, weight in zip(branch_ids, weights)}

    split_metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, List[Dict[str, int | str]]] = {}
    for split_name in ("train", "val", "test"):
        weighted_proba = None
        labels = feature_views[split_name]["labels"]
        sample_ids = feature_views[split_name]["sample_ids"]
        for branch_id, weight in branch_weights.items():
            estimator = branch_models[branch_id]
            probs = estimator.predict_proba(feature_views[split_name]["branches"][branch_id])
            weighted_proba = probs * weight if weighted_proba is None else weighted_proba + probs * weight
        pred = np.argmax(weighted_proba, axis=1)
        split_metrics[split_name] = _metrics(labels, pred)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(value)}
            for sample_id, value in zip(sample_ids, pred)
        ]

    per_branch_metrics = {
        branch_id: {
            "val_macro_f1": float(score),
        }
        for branch_id, score in zip(branch_ids, branch_scores)
    }
    return {
        "backend": "ml",
        "metrics": split_metrics,
        "branch_weights": branch_weights,
        "predictions": predictions,
        "per_branch_metrics": per_branch_metrics,
    }


class WeightedBranchFusionHead(nn.Module):
    """Fuse branch-level logits with global fixed or learned softmax weights."""

    def __init__(self, branch_dims: Dict[str, int], class_count: int, *, mode: str, fixed_weights: Dict[str, float] | None = None):
        if torch is None or nn is None:  # pragma: no cover
            raise ModuleNotFoundError("PyTorch is required for the torch backend.")
        super().__init__()
        self.branch_ids = list(branch_dims.keys())
        self.mode = mode
        self.heads = nn.ModuleDict(
            {branch_id: nn.Linear(int(branch_dims[branch_id]), class_count) for branch_id in self.branch_ids}
        )
        if mode == "learned_softmax":
            self.logits = nn.Parameter(torch.zeros(len(self.branch_ids), dtype=torch.float32))
            self.register_buffer("fixed_weight_buffer", torch.ones(len(self.branch_ids), dtype=torch.float32))
        else:
            weights = np.asarray([float((fixed_weights or {}).get(branch_id, 1.0)) for branch_id in self.branch_ids], dtype=float)
            if np.sum(weights) <= 0:
                weights = np.ones_like(weights)
            weights = weights / np.sum(weights)
            self.register_buffer("fixed_weight_buffer", torch.tensor(weights, dtype=torch.float32))
            self.logits = None

    def branch_weight_vector(self) -> "torch.Tensor":
        if self.mode == "learned_softmax":
            return torch.softmax(self.logits, dim=0)
        return self.fixed_weight_buffer

    def forward(self, branch_tensors: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        weights = self.branch_weight_vector()
        fused = None
        for index, branch_id in enumerate(self.branch_ids):
            logits = self.heads[branch_id](branch_tensors[branch_id])
            fused = logits * weights[index] if fused is None else fused + logits * weights[index]
        return fused


def _run_torch_backend(feature_plan: FeaturePlan, feature_views: Dict[str, Dict[str, Any]], fusion_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required for the torch backend.")

    branch_dims = {
        branch.node_id: int(feature_views["train"]["branches"][branch.node_id].shape[1])
        for branch in feature_plan.branches
    }
    class_count = int(len(np.unique(feature_views["train"]["labels"])))
    if class_count < 2:
        raise ValueError("Torch backend requires at least two classes in the training split.")

    head = WeightedBranchFusionHead(
        branch_dims,
        class_count,
        mode=str(fusion_cfg.get("mode", "learned_softmax")),
        fixed_weights=dict(fusion_cfg.get("fixed_weights", {}) or {}),
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=float(fusion_cfg.get("learning_rate", 1e-2)))
    criterion = nn.CrossEntropyLoss()
    epochs = int(fusion_cfg.get("epochs", 8))
    training_curve: List[Dict[str, float]] = []

    train_tensors = {
        branch_id: torch.tensor(values, dtype=torch.float32)
        for branch_id, values in feature_views["train"]["branches"].items()
    }
    train_labels = torch.tensor(feature_views["train"]["labels"], dtype=torch.long)
    val_tensors = {
        branch_id: torch.tensor(values, dtype=torch.float32)
        for branch_id, values in feature_views["val"]["branches"].items()
    }
    val_labels = torch.tensor(feature_views["val"]["labels"], dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = head(train_tensors)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_logits = head(val_tensors)
            val_loss = criterion(val_logits, val_labels)
        training_curve.append(
            {"epoch": float(epoch + 1), "train_loss": float(loss.item()), "val_loss": float(val_loss.item())}
        )

    branch_weights = {
        branch_id: float(weight)
        for branch_id, weight in zip(head.branch_ids, head.branch_weight_vector().detach().cpu().tolist())
    }

    split_metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, List[Dict[str, int | str]]] = {}
    for split_name in ("train", "val", "test"):
        tensors = {
            branch_id: torch.tensor(values, dtype=torch.float32)
            for branch_id, values in feature_views[split_name]["branches"].items()
        }
        labels = torch.tensor(feature_views[split_name]["labels"], dtype=torch.long)
        with torch.no_grad():
            logits = head(tensors)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        split_metrics[split_name] = _metrics(labels.cpu().numpy(), pred)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(value)}
            for sample_id, value in zip(feature_views[split_name]["sample_ids"], pred)
        ]

    return {
        "backend": "torch",
        "metrics": split_metrics,
        "branch_weights": branch_weights,
        "predictions": predictions,
        "training_curve": training_curve,
    }


def _render_report(runtime_config: Dict[str, Any], feature_plan: FeaturePlan, state: PHMState, results: Dict[str, Any]) -> str:
    llm = get_llm(runtime_config)
    if str(runtime_config.get("llm", {}).get("mode", "offline_stub")) == "offline_stub":
        branch_table = "\n".join(
            f"- `{branch.node_id}`: dim={branch.dimension}, op={branch.op_name}"
            for branch in feature_plan.branches
        )
        weights_table = "\n".join(
            f"- `{branch_id}`: {weight:.4f}"
            for branch_id, weight in results["branch_weights"].items()
        )
        return (
            f"# {runtime_config['runtime']['run_name']} Report\n\n"
            f"## DAG Summary\n"
            f"- Nodes: {len(state.dag_state.nodes)}\n"
            f"- Leaves: {len(state.dag_state.leaves)}\n"
            f"- Depth: {get_dag_depth(state.dag_state)}\n\n"
            f"## Feature Branches\n{branch_table}\n\n"
            f"## Branch Weights\n{weights_table}\n\n"
            f"## Metrics\n```json\n{json.dumps(results['metrics'], ensure_ascii=False, indent=2)}\n```\n"
        )

    prompt = (
        "Write a concise Markdown report for a PHM experiment.\n"
        f"Instruction: {runtime_config['experiment']['user_instruction']}\n"
        f"DAG depth: {get_dag_depth(state.dag_state)}\n"
        f"Feature branches: {json.dumps(feature_plan.model_dump(), ensure_ascii=False)}\n"
        f"Results: {json.dumps(results, ensure_ascii=False)}\n"
    )
    return llm.generate_text(prompt)


def run_preflight(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    protocol = build_protocol_from_config(runtime_config)
    llm_cfg = dict(runtime_config.get("llm", {}))
    provider = str(llm_cfg.get("provider", "offline"))
    mode = str(llm_cfg.get("mode", "offline_stub"))
    if mode != "offline_stub":
        get_llm(runtime_config)  # construct backend early
        if provider == "openrouter":
            api_key_env = str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
            if not os.getenv(api_key_env, "").strip():
                raise RuntimeError(f"OpenRouter preflight failed: missing {api_key_env}.")
        if provider == "gemini":
            api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))
            if not (os.getenv(api_key_env, "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()):
                raise RuntimeError(f"Gemini preflight failed: missing {api_key_env} or GOOGLE_API_KEY.")

    return {
        "status": "ok",
        "run_name": runtime_config["runtime"]["run_name"],
        "dataset_name": protocol.dataset_name,
        "sample_count": len(protocol.samples),
        "splits": {
            "train": len(protocol.splits.train_ids),
            "val": len(protocol.splits.val_ids),
            "test": len(protocol.splits.test_ids),
        },
        "window": protocol.window.model_dump(),
        "llm": {
            "provider": provider,
            "mode": mode,
            "model": str(llm_cfg.get("model", "")),
        },
        "output_dir": str(runtime_config["runtime"]["output_dir"]),
    }


def run_experiment(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    protocol = build_protocol_from_config(runtime_config)
    state = _preview_state(protocol, runtime_config)
    state = _build_dag(state)
    if getattr(state, "last_reflection_decision", "") == "halt":
        raise RuntimeError("Builder halted before reaching a valid terminal feature plan.")

    feature_plan = build_feature_plan(state.dag_state)
    split_records = materialize_split_signals(protocol)
    feature_views = materialize_feature_views(state.dag_state, feature_plan, split_records)

    graph_path = str(runtime_config["experiment"]["graph_path"])
    if graph_path == "ml":
        results = _run_ml_backend(feature_plan, feature_views)
    elif graph_path == "torch":
        results = _run_torch_backend(feature_plan, feature_views, dict(runtime_config.get("fusion", {})))
    else:
        results = {
            "backend": "dag_only",
            "metrics": {},
            "branch_weights": {},
            "predictions": {},
        }

    output_dir = _ensure_dir(runtime_config["runtime"]["output_dir"])
    dag_payload = {
        "nodes": {
            node_id: node.model_dump(exclude={"data", "results"})
            for node_id, node in state.dag_state.nodes.items()
        },
        "leaves": list(state.dag_state.leaves),
        "channels": list(state.dag_state.channels),
    }
    _write_json(output_dir / "resolved_config.json", runtime_config)
    _write_json(output_dir / "protocol.json", protocol.model_dump())
    _write_json(output_dir / "dag.json", dag_payload)
    _write_json(output_dir / "feature_plan.json", feature_plan.model_dump())
    _write_json(output_dir / "metrics.json", results["metrics"])
    _write_json(output_dir / "branch_weights.json", results["branch_weights"])
    _write_json(output_dir / "predictions.json", results["predictions"])
    report = _render_report(runtime_config, feature_plan, state, results)
    _write_text(output_dir / "final_report.md", report)

    payload = {
        "status": "ok",
        "run_name": runtime_config["runtime"]["run_name"],
        "graph_path": graph_path,
        "output_dir": str(output_dir),
        "artifacts": {
            "resolved_config": str(output_dir / "resolved_config.json"),
            "protocol": str(output_dir / "protocol.json"),
            "dag": str(output_dir / "dag.json"),
            "feature_plan": str(output_dir / "feature_plan.json"),
            "metrics": str(output_dir / "metrics.json"),
            "branch_weights": str(output_dir / "branch_weights.json"),
            "predictions": str(output_dir / "predictions.json"),
            "report": str(output_dir / "final_report.md"),
        },
    }
    if "training_curve" in results:
        _write_json(output_dir / "training_curve.json", results["training_curve"])
        payload["artifacts"]["training_curve"] = str(output_dir / "training_curve.json")
    return payload
