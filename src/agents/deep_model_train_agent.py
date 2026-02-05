from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.model.explainable import build_tspn_from_config, load_tspn_config
from src.states.phm_states import InputData, PHMState


@dataclass(frozen=True)
class _SplitData:
    x: np.ndarray  # (N, L, C)
    y: np.ndarray  # (N,)
    sample_ids: List[str]


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_copy(src: str | None, dst: Path) -> None:
    if not src:
        return
    try:
        shutil.copy2(src, dst)
    except Exception:
        pass


def _infer_channels_and_length(state: PHMState) -> Tuple[int, int]:
    channels = list(state.dag_state.channels)
    if not channels:
        raise ValueError("dag_state.channels is empty.")
    first = state.dag_state.nodes.get(channels[0])
    if not isinstance(first, InputData):
        raise ValueError("Expected InputData nodes for channel roots.")
    # results['ref'] holds {sample_id: (1,L,1)}
    ref_dict = (first.results or {}).get("ref") or {}
    if not isinstance(ref_dict, dict) or not ref_dict:
        raise ValueError("InputData.results['ref'] is missing or empty.")
    first_arr = next(iter(ref_dict.values()))
    if not isinstance(first_arr, np.ndarray) or first_arr.ndim != 3:
        raise ValueError("Expected channel arrays with shape (1,L,1).")
    _, L, _ = first_arr.shape
    return len(channels), int(L)


def _build_fused_view(
    state: PHMState, *, split: str, labels_map: Dict[str, Any]
) -> _SplitData:
    channels = list(state.dag_state.channels)
    nodes = state.dag_state.nodes

    per_ch: List[Dict[str, np.ndarray]] = []
    for ch in channels:
        n = nodes.get(ch)
        if not isinstance(n, InputData):
            raise ValueError(f"Channel node '{ch}' is not InputData.")
        res = n.results or {}
        split_dict = res.get(split)
        if not isinstance(split_dict, dict):
            raise ValueError(f"Channel node '{ch}' missing results['{split}'] dict.")
        per_ch.append(split_dict)

    # Intersection across channels and labels.
    common = set(labels_map.keys())
    for d in per_ch:
        common &= set(d.keys())
    sample_ids = sorted(common)
    if not sample_ids:
        return _SplitData(x=np.empty((0, 0, 0)), y=np.empty((0,), dtype=np.int64), sample_ids=[])

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for sid in sample_ids:
        ch_arrays = []
        for d in per_ch:
            arr = d[sid]
            if not isinstance(arr, np.ndarray) or arr.ndim != 3:
                raise ValueError(f"Expected (1,L,1) array for sample '{sid}'.")
            if arr.shape[0] != 1 or arr.shape[2] != 1:
                raise ValueError(f"Expected (1,L,1) array for sample '{sid}', got {arr.shape}.")
            ch_arrays.append(arr[0, :, 0])  # (L,)
        x_lc = np.stack(ch_arrays, axis=-1)  # (L, C)
        xs.append(x_lc.astype(np.float32, copy=False))
        ys.append(int(labels_map[sid]))

    x = np.stack(xs, axis=0)  # (N, L, C)
    y = np.asarray(ys, dtype=np.int64)
    return _SplitData(x=x, y=y, sample_ids=sample_ids)


def _make_label_to_index(labels_ref: Dict[str, Any]) -> Dict[str, int]:
    uniq = sorted({str(v) for v in labels_ref.values()})
    if len(uniq) < 2:
        raise ValueError("Need at least 2 classes in labels_ref.")
    return {lab: i for i, lab in enumerate(uniq)}


def _remap_labels(labels: Dict[str, Any], label_to_index: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in labels.items():
        out[str(k)] = label_to_index[str(v)]
    return out


def _train_val_split(y: np.ndarray, *, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(y.shape[0])
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * float(val_ratio))))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size == 0:
        tr_idx = val_idx
    return tr_idx, val_idx


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0.0 and (fp + fn) == 0.0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def deep_model_train_agent(state: PHMState, *, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Inner-loop trainer for the torch-side TSPN model.

    This agent enforces the label boundary:
    - training/validation use labels_ref only
    - test labels are used only when allow_test_labels_for_reporting=true
    """
    cfg = config or {}

    base_save_dir = (
        state.save_dir
        or os.getenv("PHM_SAVE_DIR")
        or cfg.get("save_dir")
        or str(Path.cwd() / "save")
    )
    case_name = state.case_name or cfg.get("case_name") or "case"
    run_dir = Path(base_save_dir) / case_name / _now_tag()
    _ensure_dir(run_dir)

    model_config_path = state.model_config_path or cfg.get("model_config_path")
    if not model_config_path:
        # Provide a sane default if the user didn't specify one.
        default_path = Path("config") / "model_tspn_basic.yaml"
        model_config_path = str(default_path) if default_path.exists() else None

    if not model_config_path or not Path(model_config_path).exists():
        err = f"TSPN model_config_path not found: {model_config_path!r}"
        state.error_logs.append(err)
        ml = dict(state.ml_results)
        ml["tspn"] = {"error": err, "artifacts_dir": str(run_dir)}
        return {"ml_results": ml, "run_dir": str(run_dir)}

    # Copy configs for reproducibility (best-effort).
    _safe_copy(model_config_path, run_dir / "model_config.yaml")

    tspn_cfg = load_tspn_config(model_config_path)

    # Fail-fast shape checks (SPEC).
    C, L = _infer_channels_and_length(state)
    if int(tspn_cfg.model.in_channels) != int(C):
        raise ValueError(f"in_channels mismatch: cfg={tspn_cfg.model.in_channels}, data={C}")
    if int(tspn_cfg.model.in_dim) != int(L):
        raise ValueError(f"in_dim mismatch: cfg={tspn_cfg.model.in_dim}, data={L}")

    # Build label mapping from ref labels only.
    if not state.labels_ref:
        # fallback to root meta (backward-compatible)
        first_ch = state.dag_state.channels[0]
        root = state.dag_state.nodes.get(first_ch)
        if isinstance(root, InputData):
            state.labels_ref = root.meta.get("labels_ref", {}) or {}
            state.labels_tst = root.meta.get("labels_tst", {}) or {}

    label_to_index = _make_label_to_index(state.labels_ref)
    labels_ref_idx = _remap_labels(state.labels_ref, label_to_index)

    # Build fused views.
    ref = _build_fused_view(state, split="ref", labels_map=labels_ref_idx)
    if ref.x.size == 0:
        raise ValueError("Empty training data after channel/label intersection.")

    # Train/val split within ref.
    tr_idx, val_idx = _train_val_split(ref.y, val_ratio=float(tspn_cfg.train.val_ratio), seed=int(tspn_cfg.train.seed))

    # Optional test split (reporting only).
    test = None
    if bool(getattr(state, "allow_test_labels_for_reporting", False)) and state.labels_tst:
        labels_tst_idx = _remap_labels(state.labels_tst, label_to_index)
        test = _build_fused_view(state, split="tst", labels_map=labels_tst_idx)

    # Optional dependency: torch.
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        err = "PyTorch is not installed; cannot run TSPN training."
        state.error_logs.append(err)
        ml = dict(state.ml_results)
        ml["tspn"] = {"error": err, "artifacts_dir": str(run_dir)}
        return {"ml_results": ml, "run_dir": str(run_dir)}

    class _ArrayDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray, sample_ids: List[str]):
            self.x = x
            self.y = y
            self.sample_ids = sample_ids

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, i: int):
            return self.x[i], int(self.y[i]), self.sample_ids[i]

    def _collate(batch):
        xs, ys, sids = zip(*batch)
        x = torch.from_numpy(np.stack(xs, axis=0))  # (B,L,C)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y, list(sids)

    device = torch.device(tspn_cfg.model.device)
    model, manifest = build_tspn_from_config(tspn_cfg, device=str(device))

    # Debug mode clamps (SPEC: smoke-run).
    epochs = int(tspn_cfg.train.debug_epochs if tspn_cfg.train.debug else tspn_cfg.train.epochs)
    max_samples = int(tspn_cfg.train.debug_max_samples) if tspn_cfg.train.debug else None

    def _subset(split_data: _SplitData, indices: np.ndarray) -> _SplitData:
        idx = indices
        if max_samples is not None:
            idx = idx[: min(len(idx), max_samples)]
        x = split_data.x[idx]
        y = split_data.y[idx]
        sids = [split_data.sample_ids[i] for i in idx.tolist()]
        return _SplitData(x=x, y=y, sample_ids=sids)

    train_data = _subset(ref, tr_idx)
    val_data = _subset(ref, val_idx)

    train_loader = DataLoader(
        _ArrayDataset(train_data.x, train_data.y, train_data.sample_ids),
        batch_size=int(tspn_cfg.train.batch_size),
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        _ArrayDataset(val_data.x, val_data.y, val_data.sample_ids),
        batch_size=int(tspn_cfg.train.batch_size),
        shuffle=False,
        collate_fn=_collate,
    )

    # Seeds for reproducibility.
    torch.manual_seed(int(tspn_cfg.train.seed))
    np.random.seed(int(tspn_cfg.train.seed))

    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tspn_cfg.train.lr),
        weight_decay=float(tspn_cfg.train.weight_decay),
    )

    best = {"epoch": 0, "val_macro_f1": -1.0}
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"

    patience = int(tspn_cfg.train.patience)
    patience_left = patience

    num_classes = int(tspn_cfg.model.num_classes)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        # Validation
        model.eval()
        val_y_true: List[int] = []
        val_y_pred: List[int] = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                val_y_pred.extend(pred)
                val_y_true.extend(yb.numpy().tolist())

        y_true = np.asarray(val_y_true, dtype=np.int64)
        y_pred = np.asarray(val_y_pred, dtype=np.int64)
        val_acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
        val_f1 = _macro_f1(y_true, y_pred, num_classes=num_classes) if y_true.size else 0.0

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, last_path)

        if val_f1 > float(best["val_macro_f1"]):
            best = {"epoch": epoch, "val_macro_f1": float(val_f1), "val_acc": float(val_acc)}
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Load best for reporting.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

    # Artifacts: mappings and manifest
    (run_dir / "channels.json").write_text(json.dumps(state.dag_state.channels, indent=2), encoding="utf-8")
    (run_dir / "label_to_index.json").write_text(json.dumps(label_to_index, indent=2), encoding="utf-8")
    (run_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Explainability artifacts
    explain_dir = run_dir / "explain"
    _ensure_dir(explain_dir)
    op_imp = model.export_operator_importance()
    (explain_dir / "operator_importance.json").write_text(json.dumps(op_imp, indent=2), encoding="utf-8")
    if bool(tspn_cfg.explain.save_wavefilters):
        wf = model.export_wavefilters_params(fs_hz=getattr(state, "fs", None))
        (explain_dir / "wavefilters_params.json").write_text(json.dumps(wf, indent=2), encoding="utf-8")

    # Predictions export (val + optional test).
    def _predict(split_name: str, split_data: _SplitData) -> List[Dict[str, Any]]:
        ds = _ArrayDataset(split_data.x, split_data.y, split_data.sample_ids)
        loader = DataLoader(ds, batch_size=int(tspn_cfg.train.batch_size), shuffle=False, collate_fn=_collate)
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for xb, yb, sids in loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
                conf = np.max(prob, axis=1)
                for i, sid in enumerate(sids):
                    row: Dict[str, Any] = {
                        "split": split_name,
                        "sample_id": sid,
                        "true": int(yb[i]),
                        "pred": int(pred[i]),
                        "confidence": float(conf[i]),
                    }
                    for c in range(num_classes):
                        row[f"proba_{c}"] = float(prob[i, c])
                    rows.append(row)
        return rows

    val_rows = _predict("val", val_data)
    test_rows: List[Dict[str, Any]] = []
    test_metrics: Dict[str, Any] = {}
    if test and test.x.size:
        test_rows = _predict("test", test)
        y_true = np.asarray([r["true"] for r in test_rows], dtype=np.int64)
        y_pred = np.asarray([r["pred"] for r in test_rows], dtype=np.int64)
        test_metrics = {
            "test_acc": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
            "test_macro_f1": _macro_f1(y_true, y_pred, num_classes=num_classes) if y_true.size else 0.0,
        }

    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list((val_rows[:1] or test_rows[:1])[0].keys()))
        writer.writeheader()
        for row in val_rows + test_rows:
            writer.writerow(row)

    metrics = {
        "best": best,
        "val": {"val_acc": float(best.get("val_acc", 0.0)), "val_macro_f1": float(best.get("val_macro_f1", 0.0))},
        **test_metrics,
        "num_classes": num_classes,
        "n_train": int(train_data.y.shape[0]),
        "n_val": int(val_data.y.shape[0]),
        "n_test": int(test.y.shape[0]) if test is not None and test.y.size else 0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_markdown = (
        "| split | acc | macro_f1 |\n"
        "|---|---:|---:|\n"
        f"| val | {metrics['val']['val_acc']:.6f} | {metrics['val']['val_macro_f1']:.6f} |\n"
        + (
            f"| test | {metrics.get('test_acc', 0.0):.6f} | {metrics.get('test_macro_f1', 0.0):.6f} |\n"
            if test_metrics
            else ""
        )
    )

    ml = dict(state.ml_results)
    ml["tspn"] = {
        "metrics": metrics,
        "metrics_markdown": metrics_markdown,
        "artifacts_dir": str(run_dir),
        "model_config_path": model_config_path,
    }

    return {"ml_results": ml, "run_dir": str(run_dir)}

