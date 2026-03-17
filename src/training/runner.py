"""Minimal runners for the ``ml`` and ``torch`` graph paths.

The paper-oriented training layer now delegates dataset assembly to
`src.data.dataset_preparer` and shallow baselines to `src.model.shallow_ml`.
This keeps the main workflow agents focused on DAG construction while the
path-specific runners own data/model execution details.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.bridge import FeaturePipelinePlan, ModelBuildPlan
from src.data import DatasetView, SignalRecord, TorchDatasetView, build_dataset_views
from src.model import build_similarity_artifacts, run_shallow_ml_baseline
from src.operators import OperatorCatalog
from .module_runtime import GraphModule

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the two primary metrics used by the rebuilt repo."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _output_widths(plan: FeaturePipelinePlan | ModelBuildPlan) -> Dict[str, int]:
    manifest_lookup = {node.node_id: node for node in plan.manifest.nodes}
    return {
        spec.output_node_id: math.prod(manifest_lookup[spec.output_node_id].shape_inference["out"])
        for spec in plan.output_specs
    }


def _importance_by_output_node(plan: FeaturePipelinePlan | ModelBuildPlan, importance_by_index: Dict[int, float]) -> Dict[str, float]:
    widths = _output_widths(plan)
    start = 0
    importance: Dict[str, float] = {}
    for spec in plan.output_specs:
        width = widths[spec.output_node_id]
        values = [importance_by_index.get(index, 0.0) for index in range(start, start + width)]
        importance[spec.output_node_id] = float(sum(values))
        start += width
    return importance


def _importance_from_weight_matrix(plan: ModelBuildPlan, weight_matrix: np.ndarray) -> Dict[str, float]:
    widths = _output_widths(plan)
    start = 0
    importance: Dict[str, float] = {}
    for spec in plan.output_specs:
        width = widths[spec.output_node_id]
        block = weight_matrix[:, start : start + width]
        importance[spec.output_node_id] = float(np.linalg.norm(block))
        start += width
    return importance


def run_ml_pipeline(
    plan: FeaturePipelinePlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    algorithm: str = "logistic_regression",
    max_iter: int = 200,
) -> Dict[str, object]:
    """Run the lightweight ML baseline on bridge-generated feature matrices."""
    dataset_views = build_dataset_views(plan, split_records, catalog, backend="np")
    baseline = run_shallow_ml_baseline(
        dataset_views,
        algorithm,
        max_iter=max_iter,
        random_state=0,
    )
    importance = _importance_by_output_node(plan, baseline["importance_by_index"])
    return {
        "feature_pipeline": plan.model_dump(),
        "algorithm": baseline["algorithm"],
        "metrics": baseline["metrics"],
        "predictions": baseline["predictions"],
        "importance": importance,
        "similarity_artifacts": build_similarity_artifacts(dataset_views),
    }


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for the graph-level torch path.")
    return torch


def _resolve_torch_device(device_spec: str) -> "torch.device":
    torch_module = _require_torch()
    normalized = device_spec.strip().lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch_module.device("cpu")
    if normalized == "cuda":
        return torch_module.device("cuda")
    if normalized.startswith("cuda:"):
        return torch_module.device(normalized)
    raise ValueError(f"Unsupported model.torch.device setting: {device_spec}")


def _compute_tensor_metrics(y_true: "torch.Tensor", y_pred: "torch.Tensor") -> Dict[str, float]:
    return _compute_metrics(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def _torch_views_to_numpy(dataset_views: Dict[str, TorchDatasetView]) -> Dict[str, DatasetView]:
    return {
        split_name: DatasetView(
            X=view.X.detach().cpu().numpy(),
            y=view.y.detach().cpu().numpy(),
            sample_ids=view.sample_ids,
        )
        for split_name, view in dataset_views.items()
    }


def _build_raw_tensor_views(split_records: Dict[str, List[SignalRecord]], *, device: "torch.device") -> Dict[str, TorchDatasetView]:
    torch_module = _require_torch()
    outputs: Dict[str, TorchDatasetView] = {}
    for split_name, records in split_records.items():
        windows: list["torch.Tensor"] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        for record in records:
            for window in record.windows:
                windows.append(torch_module.as_tensor(window, dtype=torch_module.float32, device=device))
                labels.append(record.label)
                sample_ids.append(record.sample_id)
        x_tensor = (
            torch_module.stack(windows, dim=0)
            if windows
            else torch_module.zeros((0, 0, 0), dtype=torch_module.float32, device=device)
        )
        y_tensor = torch_module.as_tensor(labels, dtype=torch_module.long, device=device)
        outputs[split_name] = TorchDatasetView(X=x_tensor, y=y_tensor, sample_ids=sample_ids)
    return outputs


def _feature_views_from_graph_module(
    graph_module: GraphModule,
    split_views: Dict[str, TorchDatasetView],
) -> Dict[str, TorchDatasetView]:
    feature_views: Dict[str, TorchDatasetView] = {}
    for split_name, view in split_views.items():
        features = graph_module(view.X) if view.X.numel() else view.X.new_zeros((0, 0))
        feature_views[split_name] = TorchDatasetView(X=features, y=view.y, sample_ids=view.sample_ids)
    return feature_views


def _graph_module_checkpoint(
    graph_module: GraphModule,
    head: "torch.nn.Linear",
    *,
    device: "torch.device",
    phase: str,
    control_default_mode: str,
    tau: float,
    attention_heads: int,
    attention_dropout: float,
) -> Dict[str, object]:
    return {
        "graph_module_state": {
            key: value.detach().cpu().tolist()
            for key, value in graph_module.state_dict().items()
        },
        "head_state": {
            "weight": head.weight.detach().cpu().tolist(),
            "bias": head.bias.detach().cpu().tolist(),
        },
        "device": str(device),
        "runtime_config": {
            "phase": phase,
            "control_default_mode": control_default_mode,
            "tau": float(tau),
            "attention_heads": int(attention_heads),
            "attention_dropout": float(attention_dropout),
        },
    }


def _run_compiled_torch_pipeline(
    plan: ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
) -> Dict[str, object]:
    torch_module = _require_torch()
    resolved_device = _resolve_torch_device(device)
    dataset_views = build_dataset_views(plan, split_records, catalog, backend="pt", device=resolved_device)
    train_view = dataset_views["train"]
    x_train = train_view.X
    y_train = train_view.y
    num_classes = int(torch_module.max(y_train).item()) + 1
    model = torch_module.nn.Linear(x_train.shape[1], num_classes, device=resolved_device)
    optimizer = torch_module.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch_module.nn.CrossEntropyLoss()
    curves: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        with torch_module.no_grad():
            preds = torch_module.argmax(logits, dim=1)
            metrics = _compute_tensor_metrics(y_train, preds)
        curves.append({"epoch": epoch + 1, "loss": float(loss.item()), "accuracy": metrics["accuracy"]})
    split_metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        view = dataset_views[split_name]
        with torch_module.no_grad():
            logits = model(view.X)
            preds = torch_module.argmax(logits, dim=1)
        split_metrics[split_name] = _compute_tensor_metrics(view.y, preds)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(pred)}
            for sample_id, pred in zip(view.sample_ids, preds.detach().cpu().tolist())
        ]
    weight_matrix = model.weight.detach().cpu().numpy()
    importance = _importance_from_weight_matrix(plan, weight_matrix)
    checkpoint = {
        "weight": model.weight.detach().cpu().tolist(),
        "bias": model.bias.detach().cpu().tolist(),
        "device": str(resolved_device),
    }
    return {
        "model_build_plan": plan.model_dump(),
        "runtime_backend": "torch_tensor_runtime",
        "training_curves": curves,
        "checkpoint": checkpoint,
        "metrics": split_metrics,
        "predictions": predictions,
        "importance": importance,
        "similarity_artifacts": build_similarity_artifacts(_torch_views_to_numpy(dataset_views)),
    }


def _run_graph_module_torch_pipeline(
    plan: ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    phase: str,
    control_default_mode: str,
    tau: float,
    attention_heads: int,
    attention_dropout: float,
) -> Dict[str, object]:
    torch_module = _require_torch()
    resolved_device = _resolve_torch_device(device)
    split_views = _build_raw_tensor_views(split_records, device=resolved_device)
    train_view = split_views["train"]
    num_classes = int(torch_module.max(train_view.y).item()) + 1
    graph_module = GraphModule(
        plan,
        catalog,
        phase=phase,
        control_default_mode=control_default_mode,
        tau=tau,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
    ).to(resolved_device)
    head = torch_module.nn.Linear(plan.trainable_head["input_dim"], num_classes, device=resolved_device)
    optimizer = torch_module.optim.SGD(
        list(graph_module.parameters()) + list(head.parameters()),
        lr=learning_rate,
    )
    loss_fn = torch_module.nn.CrossEntropyLoss()
    curves: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        features = graph_module(train_view.X)
        logits = head(features)
        loss = loss_fn(logits, train_view.y)
        loss.backward()
        optimizer.step()
        with torch_module.no_grad():
            preds = torch_module.argmax(logits, dim=1)
            metrics = _compute_tensor_metrics(train_view.y, preds)
        curves.append({"epoch": epoch + 1, "loss": float(loss.item()), "accuracy": metrics["accuracy"]})

    feature_views = _feature_views_from_graph_module(graph_module, split_views)
    split_metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        view = feature_views[split_name]
        with torch_module.no_grad():
            logits = head(view.X)
            preds = torch_module.argmax(logits, dim=1)
        split_metrics[split_name] = _compute_tensor_metrics(view.y, preds)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(pred)}
            for sample_id, pred in zip(view.sample_ids, preds.detach().cpu().tolist())
        ]
    weight_matrix = head.weight.detach().cpu().numpy()
    importance = _importance_from_weight_matrix(plan, weight_matrix)
    checkpoint = _graph_module_checkpoint(
        graph_module,
        head,
        device=resolved_device,
        phase=phase,
        control_default_mode=control_default_mode,
        tau=tau,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
    )
    runtime_backend = "torch_module_runtime" if phase == "module_runtime" else "torch_learnable_runtime"
    return {
        "model_build_plan": plan.model_dump(),
        "runtime_backend": runtime_backend,
        "training_curves": curves,
        "checkpoint": checkpoint,
        "metrics": split_metrics,
        "predictions": predictions,
        "importance": importance,
        "control_statistics": graph_module.control_statistics(),
        "similarity_artifacts": build_similarity_artifacts(_torch_views_to_numpy(feature_views)),
    }


def run_torch_pipeline(
    plan: ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    epochs: int = 12,
    learning_rate: float = 0.2,
    device: str = "auto",
    phase: str = "compiled",
    module_runtime_enabled: bool = False,
    control_default_mode: str = "fixed",
    tau: float = 1.0,
    attention_heads: int = 1,
    attention_dropout: float = 0.0,
) -> Dict[str, object]:
    """Run the current trainable path with operator-level PT execution."""
    effective_phase = str(phase).strip().lower()
    if module_runtime_enabled and effective_phase == "compiled":
        effective_phase = "module_runtime"
    if effective_phase == "compiled":
        return _run_compiled_torch_pipeline(
            plan,
            split_records,
            catalog,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
    if effective_phase not in {"module_runtime", "learnable_control"}:
        raise ValueError(f"Unsupported model.torch.phase setting: {phase}")
    return _run_graph_module_torch_pipeline(
        plan,
        split_records,
        catalog,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        phase=effective_phase,
        control_default_mode=str(control_default_mode).strip().lower(),
        tau=tau,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
    )
