"""Minimal runners for the ``ml`` and ``torch`` graph paths.

The paper-oriented training layer now delegates dataset assembly to
`src.data.dataset_preparer` and shallow baselines to `src.model.shallow_ml`.
This keeps the main workflow agents focused on DAG construction while the
path-specific runners own data/model execution details.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.bridge import FeaturePipelinePlan, ModelBuildPlan
from src.data import DatasetView, SignalRecord, TorchDatasetView, build_dataset_views
from src.model import build_similarity_artifacts, run_shallow_ml_baseline
from src.operators import OperatorCatalog

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
    importance = {
        spec.feature_node_id: float(baseline["importance_by_index"].get(index, 0.0))
        for index, spec in enumerate(plan.feature_specs)
    }
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


def run_torch_pipeline(
    plan: ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    epochs: int = 12,
    learning_rate: float = 0.2,
    device: str = "auto",
) -> Dict[str, object]:
    """Run the current trainable path with operator-level PT execution."""

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
    importance = {
        spec.feature_node_id: float(np.linalg.norm(weight_matrix[:, idx]))
        for idx, spec in enumerate(plan.feature_specs)
    }
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
