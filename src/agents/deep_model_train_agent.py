from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from src.model.explainable import build_tspn_from_config, load_tspn_config
from src.model.explainable.config_schema import TSPNConfig
from src.model.explainable.bridge import DAG2ConfigAdapter
from src.states.phm_states import InputData, PHMState, TrainReport
from src.utils.logging_setup import get_current_logger, log_event, timed
from src.utils.preflight import build_preflight_report, write_preflight_report


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


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _profile_to_model_config_path(profile: str | None) -> str | None:
    mapping = {
        "tspn_basic": "config/model_tspn_basic.yaml",
        "tspn_wf_heavy": "config/model_tspn_basic.yaml",
    }
    if not profile:
        return None
    return mapping.get(str(profile).strip())


def _resolve_model_config_path(state: PHMState, cfg: Dict[str, Any] | None = None) -> str | None:
    cfg = cfg or {}
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    profile_path = _profile_to_model_config_path(str(data_cfg.get("model_profile") or ""))
    candidate = (
        state.model_config_path
        or data_cfg.get("model_config_path")
        or cfg.get("model_config_path")
        or profile_path
    )
    if not candidate:
        default_path = Path("config") / "model_tspn_basic.yaml"
        candidate = str(default_path) if default_path.exists() else None
    return str(candidate) if candidate else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_run_preflight_report(state: PHMState, run_dir: Path) -> None:
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    preflight_cfg: Dict[str, Any] = {
        "data": data_cfg,
        "metadata_path": data_cfg.get("metadata_path"),
        "h5_path": data_cfg.get("h5_path"),
        "ref_ids": data_cfg.get("ref_ids"),
        "test_ids": data_cfg.get("test_ids"),
        "model_config_path": _resolve_model_config_path(state, {}),
    }
    report = build_preflight_report(preflight_cfg)
    write_preflight_report(report, run_dir / "preflight_report.json")


def _resolve_tspn_config(
    *,
    source_model_config_path: str,
    inferred_in_dim: int,
    inferred_in_channels: int,
    inferred_num_classes: int,
    autofit_dims: bool,
    autofit_num_classes: bool,
) -> Tuple[TSPNConfig, Dict[str, Any]]:
    source_cfg = load_tspn_config(source_model_config_path)
    cfg_dict = source_cfg.model_dump()
    overrides: Dict[str, Any] = {}

    cfg_dims = (int(cfg_dict["model"]["in_dim"]), int(cfg_dict["model"]["in_channels"]))
    if autofit_dims:
        cfg_dict["model"]["in_dim"] = int(inferred_in_dim)
        cfg_dict["model"]["in_channels"] = int(inferred_in_channels)
        if cfg_dims != (int(inferred_in_dim), int(inferred_in_channels)):
            overrides["dims"] = {
                "from": {"in_dim": cfg_dims[0], "in_channels": cfg_dims[1]},
                "to": {"in_dim": int(inferred_in_dim), "in_channels": int(inferred_in_channels)},
            }
    elif cfg_dims != (int(inferred_in_dim), int(inferred_in_channels)):
        raise ValueError(
            f"Model/data dimension mismatch: cfg(in_dim={cfg_dims[0]}, in_channels={cfg_dims[1]}) "
            f"!= data(in_dim={inferred_in_dim}, in_channels={inferred_in_channels})."
        )

    cfg_num_classes = int(cfg_dict["model"]["num_classes"])
    if autofit_num_classes:
        cfg_dict["model"]["num_classes"] = int(inferred_num_classes)
        if cfg_num_classes != int(inferred_num_classes):
            overrides["num_classes"] = {"from": cfg_num_classes, "to": int(inferred_num_classes)}
    elif cfg_num_classes != int(inferred_num_classes):
        raise ValueError(
            f"num_classes mismatch: cfg={cfg_num_classes} vs labels={inferred_num_classes}. "
            "Set model.autofit_num_classes=true or align labels/model config."
        )

    resolved_cfg = TSPNConfig.model_validate(cfg_dict)
    resolve_info = {
        "source_model_config_path": source_model_config_path,
        "autofit_dims": autofit_dims,
        "autofit_num_classes": autofit_num_classes,
        "inferred": {
            "in_dim": int(inferred_in_dim),
            "in_channels": int(inferred_in_channels),
            "num_classes": int(inferred_num_classes),
        },
        "overrides": overrides,
    }
    return resolved_cfg, resolve_info


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


def _train_with_vibench_factory(state: PHMState, *, run_dir: Path, data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_current_logger()
    log_event(
        logger,
        level="INFO",
        event="train.vibench.start",
        phase="train",
        node="train",
        message="Start vibench TSPN training.",
        payload={"run_dir": str(run_dir), "backend": "vibench"},
    )
    # Optional dependency: torch.
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:  # pragma: no cover
        err = f"PyTorch is required for vibench training: {type(e).__name__}: {e}"
        state.error_logs.append(err)
        ml = dict(state.ml_results)
        ml["tspn"] = {"error": err, "artifacts_dir": str(run_dir)}
        return {"ml_results": ml, "run_dir": str(run_dir)}

    from src.utils.data_factory_wrapper import PHMVibenchDataFactory

    built = PHMVibenchDataFactory(data_cfg).build()
    train_loader = built.train_loader
    val_loader = built.val_loader
    test_loader = built.test_loader
    label_to_index = dict(built.label_to_index)

    # Infer shape from first batch.
    first = next(iter(train_loader))
    x0 = first["x"]
    B, L, C = tuple(int(x) for x in x0.shape)
    num_classes = max(2, len(label_to_index))

    source_model_config_path = _resolve_model_config_path(state, {"model_config_path": data_cfg.get("model_config_path")})
    autofit_dims = _parse_bool(data_cfg.get("autofit_dims"), default=True)
    autofit_num_classes = _parse_bool(data_cfg.get("autofit_num_classes"), default=True)

    # Bridge: DAG -> init metadata (+ fallback config).
    fs_hz = None
    try:
        fs_hz = float(data_cfg.get("fs_hz")) if data_cfg.get("fs_hz") is not None else None
    except Exception:
        fs_hz = None
    if fs_hz is None:
        # fallback to root meta if present
        try:
            ch0 = state.dag_state.channels[0]
            root = state.dag_state.nodes.get(ch0)
            if isinstance(root, InputData):
                fs_hz = float((root.meta or {}).get("fs")) if (root.meta or {}).get("fs") is not None else None
        except Exception:
            fs_hz = None

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=C,
        num_classes=num_classes,
        fs_hz=fs_hz,
        max_layers=int(data_cfg.get("max_layers") or 4),
        parallel_ops_per_layer=int(data_cfg.get("parallel_ops_per_layer") or 4),
        out_channels=int(data_cfg.get("out_channels") or 3),
        scale=int(data_cfg.get("scale") or 4),
        feature_tokens=list(data_cfg.get("features") or ["Mean", "Std", "RMS"]),
        fft_align_strategy=str(data_cfg.get("fft_align_strategy") or "interp"),
        preserve_dag_topology=bool(data_cfg.get("preserve_topology", True)),
        allow_duplicate_tokens=bool(data_cfg.get("allow_duplicate_tokens", True)),
        unsupported_policy=str(data_cfg.get("unsupported_policy") or "fallback_to_identity"),
    )
    bridge = adapter.adapt(state.dag_state)

    cfg_dict: Dict[str, Any]
    resolve_info: Dict[str, Any]
    if source_model_config_path and Path(source_model_config_path).exists():
        resolved_cfg, resolve_info = _resolve_tspn_config(
            source_model_config_path=source_model_config_path,
            inferred_in_dim=L,
            inferred_in_channels=C,
            inferred_num_classes=num_classes,
            autofit_dims=autofit_dims,
            autofit_num_classes=autofit_num_classes,
        )
        cfg_dict = resolved_cfg.model_dump()
    else:
        cfg_dict = dict(bridge.model_config)
        cfg_dict.setdefault("model", {})
        cfg_dict["model"]["in_dim"] = int(L)
        cfg_dict["model"]["in_channels"] = int(C)
        cfg_dict["model"]["num_classes"] = int(num_classes)
        resolve_info = {
            "source_model_config_path": source_model_config_path,
            "autofit_dims": autofit_dims,
            "autofit_num_classes": autofit_num_classes,
            "inferred": {"in_dim": int(L), "in_channels": int(C), "num_classes": int(num_classes)},
            "overrides": {"mode": "dag_bridge"},
        }
    init_metadata = bridge.init_metadata

    # Training overrides from data_cfg (optional).
    device_str = str(data_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_dict["model"]["device"] = device_str
    if "train" not in cfg_dict:
        cfg_dict["train"] = {}
    for k in ("seed", "epochs", "batch_size", "lr", "weight_decay", "patience", "debug", "debug_max_samples", "debug_epochs"):
        if k in data_cfg and data_cfg[k] is not None:
            cfg_dict["train"][k] = data_cfg[k]

    tspn_cfg = TSPNConfig.model_validate(cfg_dict)
    model_cfg_snapshot = tspn_cfg.model_dump()

    # Write artifacts: source/resolved config + init metadata + resolve summary.
    model_config_path = run_dir / "model_config.yaml"
    model_config_path.write_text(yaml.safe_dump(model_cfg_snapshot, sort_keys=False), encoding="utf-8")
    resolved_model_path = run_dir / "model_config.resolved.yaml"
    resolved_model_path.write_text(yaml.safe_dump(model_cfg_snapshot, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "config_resolve.json", resolve_info)
    (run_dir / "init_metadata.json").write_text(json.dumps(init_metadata, indent=2), encoding="utf-8")
    _write_run_preflight_report(state, run_dir)

    # Reproducibility
    seed = int(getattr(tspn_cfg.train, "seed", 42) or 42)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    device = torch.device(str(tspn_cfg.model.device))
    model, manifest = build_tspn_from_config(tspn_cfg, device=str(device))
    # Initialize learnable params from DAG metadata.
    try:
        model.init_weights_from_metadata(init_metadata)
    except Exception:
        pass

    epochs = int(tspn_cfg.train.debug_epochs if tspn_cfg.train.debug else tspn_cfg.train.epochs)
    l1_gate = float(getattr(tspn_cfg.train, "l1_gate", 0.0) or 0.0)
    entropy_gate = float(getattr(tspn_cfg.train, "entropy_gate", 0.0) or 0.0)
    grad_clip_norm = float(getattr(tspn_cfg.train, "grad_clip_norm", 1.0) or 0.0)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tspn_cfg.train.lr),
        weight_decay=float(tspn_cfg.train.weight_decay),
    )

    def _gate_regularization() -> "torch.Tensor":
        if l1_gate == 0.0 and entropy_gate == 0.0:
            return torch.as_tensor(0.0, device=device)
        l1 = torch.as_tensor(0.0, device=device)
        ent = torch.as_tensor(0.0, device=device)
        for layer in getattr(model, "signal_layers", []):
            if l1_gate:
                l1 = l1 + torch.mean(layer.op_gates())
            if entropy_gate:
                p = layer.op_probs()
                ent = ent + (-(p * torch.log(p + 1e-12)).sum())
        return l1_gate * l1 + entropy_gate * ent

    def _eval(loader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        y_true: List[int] = []
        y_pred: List[int] = []
        ids: List[str] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                sids = list(batch.get("file_id") or [])
                logits = model(xb)
                pred = torch.argmax(logits, dim=-1)
                y_true.extend(yb.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                ids.extend([str(s) for s in sids])
        return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), ids

    best = {"epoch": 0, "val_macro_f1": -1.0, "val_acc": 0.0}
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    patience = int(tspn_cfg.train.patience)
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if l1_gate or entropy_gate:
                loss = loss + _gate_regularization()
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, last_path)

        val_true, val_pred, _ = _eval(val_loader)
        val_acc = float(np.mean(val_true == val_pred)) if val_true.size else 0.0
        val_f1 = _macro_f1(val_true, val_pred, num_classes=num_classes) if val_true.size else 0.0
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

    # Basic artifacts.
    (run_dir / "channels.json").write_text(json.dumps(state.dag_state.channels, indent=2), encoding="utf-8")
    (run_dir / "label_to_index.json").write_text(json.dumps(label_to_index, indent=2), encoding="utf-8")
    (run_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Predictions + metrics.
    val_true, val_pred, val_ids = _eval(val_loader)
    test_metrics: Dict[str, Any] = {}
    allow_test = bool(getattr(state, "allow_test_labels_for_reporting", False))
    test_true = np.asarray([], dtype=np.int64)
    test_pred = np.asarray([], dtype=np.int64)
    test_ids: List[str] = []
    if allow_test:
        test_true, test_pred, test_ids = _eval(test_loader)
        if test_true.size:
            test_metrics = {
                "test_acc": float(np.mean(test_true == test_pred)),
                "test_macro_f1": _macro_f1(test_true, test_pred, num_classes=num_classes),
            }

    def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
        labels = [str(i) for i in range(int(num_classes))]
        mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                mat[int(t)][int(p)] += 1
        return {"labels": labels, "matrix": mat}

    metrics = {
        "best": best,
        "val": {"val_acc": float(best.get("val_acc", 0.0)), "val_macro_f1": float(best.get("val_macro_f1", 0.0))},
        **test_metrics,
        "num_classes": int(num_classes),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    dataset_manifest = {
        "source_mode": "vibench",
        "dataset_name": data_cfg.get("dataset_name"),
        "task_type": data_cfg.get("task_type"),
        "task_name": data_cfg.get("task_name"),
        "split_protocol": "vibench_factory(train/val/test)",
        "n_train": int(len(getattr(train_loader, "dataset", []))),
        "n_val": int(len(getattr(val_loader, "dataset", []))),
        "n_test": int(len(getattr(test_loader, "dataset", []))),
        "num_classes": int(num_classes),
        "label_to_index": label_to_index,
    }
    _write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    log_event(
        logger,
        level="INFO",
        event="train.vibench.metrics",
        phase="train",
        node="train",
        message="vibench training metrics generated.",
        payload={"metrics": metrics},
    )

    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "sample_id", "true", "pred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid, t, p in zip(val_ids, val_true.tolist(), val_pred.tolist()):
            writer.writerow({"split": "val", "sample_id": sid, "true": int(t), "pred": int(p)})
        for sid, t, p in zip(test_ids, test_true.tolist(), test_pred.tolist()):
            writer.writerow({"split": "test", "sample_id": sid, "true": int(t), "pred": int(p)})

    # Explainability artifacts
    explain_dir = run_dir / "explain"
    _ensure_dir(explain_dir)
    op_imp = model.export_operator_importance()
    (explain_dir / "operator_importance.json").write_text(json.dumps(op_imp, indent=2), encoding="utf-8")
    if bool(getattr(tspn_cfg.explain, "save_wavefilters", True)):
        wf = model.export_wavefilters_params(fs_hz=fs_hz)
        (explain_dir / "wavefilters_params.json").write_text(json.dumps(wf, indent=2), encoding="utf-8")

    explain_summary = {"operator_importance": op_imp, "wavefilters": {"fs_hz": fs_hz}}

    report = TrainReport(
        run_id=run_dir.name,
        dataset_id=str(getattr(state, "case_name", "") or ""),
        task_id="",
        split_protocol={"train": "vibench_train", "val": "vibench_val", "test": "vibench_test", "allow_test_labels_for_reporting": allow_test},
        metrics=metrics,
        confusion_matrix=_confusion(val_true, val_pred, num_classes=num_classes),
        explain_summary=explain_summary,
        error_modes=[],
        artifacts={
            "artifacts_dir": str(run_dir),
            "model_config_path": str(model_config_path),
            "model_config_resolved_path": str(resolved_model_path),
            "dataset_manifest_path": str(run_dir / "dataset_manifest.json"),
            "config_resolve_path": str(run_dir / "config_resolve.json"),
            "preflight_report_path": str(run_dir / "preflight_report.json"),
        },
    )

    # Update state-like outputs.
    history = list(getattr(state, "train_history", []) or []) + [report]
    ml = dict(getattr(state, "ml_results", {}) or {})
    ml["tspn"] = {
        "metrics": metrics,
        "artifacts_dir": str(run_dir),
        "model_config_path": str(resolved_model_path),
    }
    return {
        "ml_results": ml,
        "run_dir": str(run_dir),
        "train_history": history,
        "current_model_config": model_cfg_snapshot,
        "model_config_path": str(resolved_model_path),
    }


def deep_model_train_agent(state: PHMState, *, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Inner-loop trainer for the torch-side TSPN model.

    This agent enforces the label boundary:
    - training/validation use labels_ref only
    - test labels are used only when allow_test_labels_for_reporting=true
    """
    cfg = config or {}
    logger = get_current_logger()

    base_save_dir = (
        state.save_dir
        or os.getenv("PHM_SAVE_DIR")
        or cfg.get("save_dir")
        or str(Path.cwd() / "save")
    )
    case_name = state.case_name or cfg.get("case_name") or "case"
    run_dir = Path(base_save_dir) / case_name / _now_tag()
    _ensure_dir(run_dir)
    log_event(
        logger,
        level="INFO",
        event="train.start",
        phase="train",
        node="train",
        message="Start deep_model_train_agent.",
        payload={"case_name": case_name, "run_dir": str(run_dir)},
    )

    # --- Real-data backend: PHM-Vibench data_factory ---
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    if str(data_cfg.get("backend") or "").strip().lower() == "vibench":
        with timed(logger, event="train.vibench", phase="train", node="train"):
            return _train_with_vibench_factory(state, run_dir=run_dir, data_cfg=data_cfg)

    model_config_path = _resolve_model_config_path(state, cfg)
    if not model_config_path or not Path(model_config_path).exists():
        err = f"TSPN model_config_path not found: {model_config_path!r}"
        state.error_logs.append(err)
        ml = dict(state.ml_results)
        ml["tspn"] = {"error": err, "artifacts_dir": str(run_dir)}
        log_event(
            logger,
            level="ERROR",
            event="train.config_missing",
            phase="train",
            node="train",
            message=err,
            payload={"model_config_path": model_config_path},
        )
        return {"ml_results": ml, "run_dir": str(run_dir)}

    # Infer dims from data.
    C, L = _infer_channels_and_length(state)

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
    inferred_num_classes = len(label_to_index)

    autofit_dims = _parse_bool(data_cfg.get("autofit_dims"), default=True)
    autofit_num_classes = _parse_bool(data_cfg.get("autofit_num_classes"), default=True)

    tspn_cfg, resolve_info = _resolve_tspn_config(
        source_model_config_path=model_config_path,
        inferred_in_dim=L,
        inferred_in_channels=C,
        inferred_num_classes=inferred_num_classes,
        autofit_dims=autofit_dims,
        autofit_num_classes=autofit_num_classes,
    )
    model_cfg_snapshot = tspn_cfg.model_dump()
    # Persist config lineage (source + resolved).
    _safe_copy(model_config_path, run_dir / "model_config.source.yaml")
    resolved_model_path = run_dir / "model_config.resolved.yaml"
    resolved_model_path.write_text(yaml.safe_dump(model_cfg_snapshot, sort_keys=False), encoding="utf-8")
    (run_dir / "model_config.yaml").write_text(yaml.safe_dump(model_cfg_snapshot, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "config_resolve.json", resolve_info)
    _write_run_preflight_report(state, run_dir)

    # Build fused views.
    ref = _build_fused_view(state, split="ref", labels_map=labels_ref_idx)
    if ref.x.size == 0:
        raise ValueError("Empty training data after channel/label intersection.")

    seed = int(getattr(tspn_cfg.train, "seed", 42) or 42)
    random.seed(seed)
    np.random.seed(seed)

    # Train/val split within ref.
    tr_idx, val_idx = _train_val_split(ref.y, val_ratio=float(tspn_cfg.train.val_ratio), seed=seed)

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

    # Reproducibility: align torch RNGs for stable model init / training.
    try:  # pragma: no cover
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

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

    l1_gate = float(getattr(tspn_cfg.train, "l1_gate", 0.0) or 0.0)
    entropy_gate = float(getattr(tspn_cfg.train, "entropy_gate", 0.0) or 0.0)
    grad_clip_norm = float(getattr(tspn_cfg.train, "grad_clip_norm", 1.0) or 0.0)

    def _gate_regularization() -> "torch.Tensor":
        if l1_gate == 0.0 and entropy_gate == 0.0:
            return torch.as_tensor(0.0, device=device)
        l1 = torch.as_tensor(0.0, device=device)
        ent = torch.as_tensor(0.0, device=device)
        for layer in getattr(model, "signal_layers", []):
            if l1_gate:
                l1 = l1 + torch.mean(layer.op_gates())
            if entropy_gate:
                p = layer.op_probs()
                ent = ent + (-(p * torch.log(p + 1e-12)).sum())
        return l1_gate * l1 + entropy_gate * ent

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if l1_gate or entropy_gate:
                # Sparsity/peakedness regularization on operator gates.
                loss = loss + _gate_regularization()
            opt.zero_grad()
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
    dataset_manifest = {
        "source_mode": str(data_cfg.get("source_mode") or "fixed_ids"),
        "dataset_name": data_cfg.get("dataset_name"),
        "split_protocol": "fixed_ids(ref->train/val, tst->test)",
        "n_train": int(train_data.y.shape[0]),
        "n_val": int(val_data.y.shape[0]),
        "n_test": int(test.y.shape[0]) if test is not None and test.y.size else 0,
        "num_classes": int(num_classes),
        "label_to_index": label_to_index,
    }
    _write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    log_event(
        logger,
        level="INFO",
        event="train.metrics",
        phase="train",
        node="train",
        message="Training metrics generated.",
        payload={"metrics": metrics},
    )

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
        "model_config_path": str(resolved_model_path),
    }

    # Build a minimal TrainReport aligned with AGENT_IO.md for downstream reflection.
    def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
        labels = [str(i) for i in range(int(num_classes))]
        mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                mat[int(t)][int(p)] += 1
        return {"labels": labels, "matrix": mat}

    # Derive val confusion from exported predictions.
    val_true = np.asarray([r["true"] for r in val_rows], dtype=np.int64)
    val_pred = np.asarray([r["pred"] for r in val_rows], dtype=np.int64)

    # Top-k operator importance per layer (agent-readable).
    topk_ops = int(getattr(tspn_cfg.explain, "topk_ops", 3) or 3)
    op_topk: List[Dict[str, Any]] = []
    for layer in (op_imp.get("layers") or []):
        ops = list(layer.get("operators") or [])
        top = sorted(ops, key=lambda d: float(d.get("prob", 0.0)), reverse=True)[:topk_ops]
        op_topk.append(
            {
                "layer": int(layer.get("layer", 0)),
                "topk": [{"op_uid": t.get("op_uid"), "score": float(t.get("prob", 0.0))} for t in top],
            }
        )

    # Feature stats (lightweight, agent-readable).
    feature_stats: Dict[str, Any] = {"tokens": list(tspn_cfg.model.features), "per_token": {}}
    try:
        with torch.no_grad():
            seen = 0
            for xb, _, _ in val_loader:
                xb = xb.to(device)
                x_mid = xb
                for layer in getattr(model, "signal_layers", []):
                    x_mid = layer(x_mid)
                x_bcl = x_mid.permute(0, 2, 1)  # (B,C,L)
                for token, mod in getattr(model.feature_layer, "feature_modules", {}).items():
                    feat = mod(x_bcl)  # (B,C,1)
                    feature_stats["per_token"].setdefault(token, {})
                    feature_stats["per_token"][token]["mean_abs"] = float(feat.abs().mean().cpu())
                seen += 1
                if seen >= 2:
                    break
    except Exception:
        pass

    (explain_dir / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")

    explain_summary: Dict[str, Any] = {
        "operator_importance": op_topk,
        "wavefilters": {"enabled": bool(tspn_cfg.explain.save_wavefilters), "fs_hz": getattr(state, "fs", None)},
        "feature_stats": feature_stats,
    }

    run_id = run_dir.name
    report = TrainReport(
        run_id=run_id,
        dataset_id=str(getattr(state, "case_name", "") or ""),
        task_id="",
        split_protocol={
            "train": "ref_train",
            "val": "ref_val",
            "test": "tst",
            "allow_test_labels_for_reporting": bool(getattr(state, "allow_test_labels_for_reporting", False)),
        },
        metrics={
            "val_macro_f1": float(metrics["val"]["val_macro_f1"]),
            "val_acc": float(metrics["val"]["val_acc"]),
            "test_macro_f1": metrics.get("test_macro_f1") if metrics.get("n_test", 0) else None,
            "test_acc": metrics.get("test_acc") if metrics.get("n_test", 0) else None,
        },
        confusion_matrix=_confusion(val_true, val_pred, num_classes=num_classes),
        explain_summary=explain_summary,
        error_modes=[],
        artifacts={
            "metrics_json": str(run_dir / "metrics.json"),
            "predictions_csv": str(pred_path),
            "operator_importance": str(explain_dir / "operator_importance.json"),
            "feature_stats": str(explain_dir / "feature_stats.json"),
            "wavefilters_params": str(explain_dir / "wavefilters_params.json")
            if bool(tspn_cfg.explain.save_wavefilters)
            else None,
            "model_manifest": str(run_dir / "model_manifest.json"),
            "dataset_manifest": str(run_dir / "dataset_manifest.json"),
            "config_resolve": str(run_dir / "config_resolve.json"),
            "preflight_report": str(run_dir / "preflight_report.json"),
            "model_config_resolved": str(resolved_model_path),
            "artifacts_dir": str(run_dir),
        },
    )

    history = list(getattr(state, "train_history", []) or [])
    history.append(report)

    return {
        "ml_results": ml,
        "run_dir": str(run_dir),
        "train_history": history,
        "current_model_config": model_cfg_snapshot,
        "model_config_path": str(resolved_model_path),
    }
