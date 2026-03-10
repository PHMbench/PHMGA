from __future__ import annotations

from dataclasses import asdict
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .config_schema import TSPNConfig

LOGGER = logging.getLogger(__name__)


def load_tspn_config(path: str | Path) -> TSPNConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Legacy adapter: Unified_X_fault_diagnosis/config_basic.yaml style
    # - signal_processing_configs: {layer1: [I, WF, ...], ...}
    # - feature_extractor_configs: [Mean, Std, ...]
    # - args: contains model/train hyperparams
    if "signal_processing_configs" in raw and "args" in raw:
        args = raw.get("args") or {}
        sp = raw.get("signal_processing_configs") or {}
        feat = raw.get("feature_extractor_configs") or []

        # Sort layer keys by trailing number if present.
        def _layer_key(k: str) -> int:
            s = "".join(ch for ch in str(k) if ch.isdigit())
            return int(s) if s else 0

        layers = []
        for lk in sorted(sp.keys(), key=_layer_key):
            tokens = sp.get(lk) or []
            layers.append(
                {
                    "gate_temperature": 1.0,
                    "ops": [{"token": str(t).strip(), "params": {}} for t in tokens],
                }
            )

        adapted = {
            "model": {
                "name": "tspn",
                "device": args.get("device", "cpu"),
                "num_classes": int(args.get("num_classes", 2)),
                "in_dim": int(args.get("in_dim", 4096)),
                "in_channels": int(args.get("in_channels", 1)),
                "out_channels": int(args.get("out_channels", 3)),
                "scale": int(args.get("scale", 4)),
                "skip_connection": bool(args.get("skip_connection", True)),
                "wf_init": {
                    "f_c_mu": float(args.get("f_c_mu", 0.0)),
                    "f_c_sigma": float(args.get("f_c_sigma", 0.1)),
                    "f_b_mu": float(args.get("f_b_mu", 0.0)),
                    "f_b_sigma": float(args.get("f_b_sigma", 0.1)),
                },
                "layers": layers,
                "features": list(feat),
                "disabled_ops": {},
            },
            "train": {
                "seed": int(args.get("seed", 42)),
                "epochs": int(args.get("num_epochs", 10)),
                "batch_size": int(args.get("batch_size", 32)),
                "lr": float(args.get("learning_rate", 1e-3)),
                "weight_decay": float(args.get("weight_decay", 0.0)),
                "val_ratio": 0.2,
                "patience": int(args.get("patience", 10)),
                "debug": bool(args.get("debug", False)),
                "debug_max_samples": 16,
                "debug_epochs": 1,
            },
            "explain": {"topk_ops": 3, "save_wavefilters": True},
            "meta": {"legacy_source": str(p)},
        }
        return TSPNConfig.model_validate(adapted)

    return TSPNConfig.model_validate(raw)


def _make_op_uid(layer_idx: int, token: str, occurrence_idx: int) -> str:
    return f"L{int(layer_idx)}:{token}:{int(occurrence_idx)}"


def _resolve_out_channels_for_layers(
    *,
    original_out_channels: int,
    scale: int,
    module_counts: List[int],
) -> Tuple[int, Dict[str, Any] | None]:
    if not module_counts:
        return int(original_out_channels), None
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if any(n <= 0 for n in module_counts):
        raise ValueError(f"Each layer must have at least one op, got module counts: {module_counts}")

    out_total_before = int(original_out_channels) * int(scale)
    lcm_ops = 1
    for n in module_counts:
        lcm_ops = math.lcm(lcm_ops, int(n))

    if out_total_before % lcm_ops == 0:
        return int(original_out_channels), None

    out_total_after = ((out_total_before + lcm_ops - 1) // lcm_ops) * lcm_ops
    while out_total_after % int(scale) != 0:
        out_total_after += lcm_ops

    out_channels_after = out_total_after // int(scale)
    return out_channels_after, {
        "module_counts": [int(n) for n in module_counts],
        "lcm_ops": int(lcm_ops),
        "out_total_before": int(out_total_before),
        "out_total_after": int(out_total_after),
        "out_channels_before": int(original_out_channels),
        "out_channels_after": int(out_channels_after),
        "reason": "auto_adjusted_for_layer_divisibility",
    }


def build_tspn_from_config(
    cfg: TSPNConfig, *, device: str | None = None
) -> Tuple["TransparentSignalProcessingNetwork", Dict[str, Any]]:
    """
    Build a :class:`TransparentSignalProcessingNetwork` from a validated config.

    Returns
    -------
    (model, manifest)
        manifest includes op_uid mapping and the resolved operator list.
    """
    # Optional dependency: torch is required only when actually building the model.
    try:
        import torch.nn as nn  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required to build and train TSPN. Install torch, then retry."
        ) from e

    from .ops import make_op  # local import to avoid importing torch at module import time
    from .tspn import TSPNArgs, TransparentSignalProcessingNetwork

    model_cfg = cfg.model
    runtime_device = device or model_cfg.device
    module_counts = [len(layer_cfg.ops) for layer_cfg in model_cfg.layers]
    resolved_out_channels, channel_adjustment = _resolve_out_channels_for_layers(
        original_out_channels=int(model_cfg.out_channels),
        scale=int(model_cfg.scale),
        module_counts=module_counts,
    )
    if channel_adjustment:
        LOGGER.warning(
            "TSPN channel auto-adjust: out_total %s -> %s (out_channels %s -> %s), module_counts=%s",
            channel_adjustment["out_total_before"],
            channel_adjustment["out_total_after"],
            channel_adjustment["out_channels_before"],
            channel_adjustment["out_channels_after"],
            channel_adjustment["module_counts"],
        )

    args = TSPNArgs(
        device=runtime_device,
        num_classes=model_cfg.num_classes,
        in_dim=model_cfg.in_dim,
        in_channels=model_cfg.in_channels,
        out_channels=resolved_out_channels,
        scale=model_cfg.scale,
        skip_connection=model_cfg.skip_connection,
        f_c_mu=float(model_cfg.wf_init.get("f_c_mu", 0.0)),
        f_c_sigma=float(model_cfg.wf_init.get("f_c_sigma", 0.1)),
        f_b_mu=float(model_cfg.wf_init.get("f_b_mu", 0.0)),
        f_b_sigma=float(model_cfg.wf_init.get("f_b_sigma", 0.1)),
    )

    layer_modules: List[nn.ModuleDict] = []
    layer_op_uids: List[List[str]] = []
    op_uid_to_module_key: Dict[str, str] = {}
    layer_gate_temperatures: List[float] = []

    # In the current TSPN implementation, each layer expands to a fixed channel width:
    # out_total = out_channels * scale, then split evenly per op.
    out_total = int(args.out_channels * args.scale)

    for layer_idx, layer_cfg in enumerate(model_cfg.layers, start=1):
        modules = nn.ModuleDict()
        uids: List[str] = []
        layer_gate_temperatures.append(float(getattr(layer_cfg, "gate_temperature", 1.0)))

        token_counts: Dict[str, int] = {}
        module_num = len(layer_cfg.ops)
        if module_num <= 0:
            raise ValueError(f"Layer {layer_idx}: requires at least one op.")
        if out_total % module_num != 0:
            raise ValueError(
                f"Layer {layer_idx}: out_total={out_total} must be divisible by num_ops={module_num}"
            )
        out_per = out_total // module_num

        for op_cfg in layer_cfg.ops:
            token = op_cfg.token.strip()
            occurrence = token_counts.get(token, 0)
            token_counts[token] = occurrence + 1

            op_uid = _make_op_uid(layer_idx, token, occurrence)
            module_key = f"{token}_{occurrence}"

            params = dict(op_cfg.params or {})
            # Provide WF init defaults unless explicitly overridden.
            if token == "WF":
                for k, v in model_cfg.wf_init.items():
                    params.setdefault(k, v)
            elif token == "NORM":
                for k, v in (model_cfg.norm_init or {}).items():
                    params.setdefault(k, v)
            elif token == "STFT":
                for k, v in (model_cfg.stft_init or {}).items():
                    params.setdefault(k, v)
            elif token == "SIN":
                for k, v in (model_cfg.sin_init or {}).items():
                    params.setdefault(k, v)
            elif token == "FFT":
                # Keep FFT behavior deterministic unless explicitly overridden.
                params.setdefault("align_strategy", "interp")

            modules[module_key] = make_op(token, channels=out_per, **params)
            uids.append(op_uid)
            op_uid_to_module_key[op_uid] = module_key

        layer_modules.append(modules)
        layer_op_uids.append(uids)

    model = TransparentSignalProcessingNetwork(
        args=args,
        layer_modules=layer_modules,
        layer_op_uids=layer_op_uids,
        feature_tokens=list(model_cfg.features),
        disabled_ops=dict(model_cfg.disabled_ops),
        layer_gate_temperatures=layer_gate_temperatures,
    )

    manifest: Dict[str, Any] = {
        "args": asdict(args),
        "op_uid_to_module_key": op_uid_to_module_key,
        "layers": [
            {
                "layer": i + 1,
                "ops": [
                    {"op_uid": uid, "module_key": op_uid_to_module_key[uid]}
                    for uid in layer_uids
                ],
            }
            for i, layer_uids in enumerate(layer_op_uids)
        ],
        "features": list(model_cfg.features),
    }
    if channel_adjustment:
        manifest["channel_adjustment"] = channel_adjustment

    return model, manifest
