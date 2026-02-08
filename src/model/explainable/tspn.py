from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import WaveFilters, make_op
from .feature_ops import make_feature


@dataclass(frozen=True)
class TSPNArgs:
    device: str = "cpu"
    num_classes: int = 2
    in_dim: int = 4096
    in_channels: int = 2
    out_channels: int = 3
    scale: int = 4
    skip_connection: bool = True
    f_c_mu: float = 0.0
    f_c_sigma: float = 0.1
    f_b_mu: float = 0.0
    f_b_sigma: float = 0.1


class SignalProcessingLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        modules: nn.ModuleDict,
        op_uids: List[str],
        in_channels: int,
        out_channels: int,
        skip_connection: bool = True,
        gate_temperature: float = 1.0,
        disabled_ops: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.modules_dict = modules
        self.op_uids = list(op_uids)
        self.module_num = len(self.modules_dict)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gate_temperature = float(gate_temperature)

        if self.module_num == 0:
            raise ValueError("SignalProcessingLayer requires at least 1 op module.")
        if self.out_channels % self.module_num != 0:
            raise ValueError(
                f"out_channels={self.out_channels} must be divisible by module_num={self.module_num}"
            )

        self.norm = nn.InstanceNorm1d(self.in_channels)
        self.weight_connection = nn.Linear(self.in_channels, self.out_channels)

        self.op_gate_logits = nn.Parameter(torch.zeros(self.module_num))
        if disabled_ops:
            # Initialize disabled ops gates to requested values (default 1e-6)
            for i, uid in enumerate(self.op_uids):
                if uid in disabled_ops:
                    gate = float(disabled_ops[uid])
                    gate = min(max(gate, 1e-12), 1.0 - 1e-12)
                    with torch.no_grad():
                        self.op_gate_logits[i] = torch.log(torch.tensor(gate / (1.0 - gate)))

        if skip_connection:
            self.skip_connection = nn.Linear(self.in_channels, self.out_channels)

    def op_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.op_gate_logits / self.gate_temperature)

    def op_probs(self) -> torch.Tensor:
        g = self.op_gates()
        return g / (g.sum() + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in)
        x_bcl = x.permute(0, 2, 1)
        x_bcl = self.norm(x_bcl)
        x = x_bcl.permute(0, 2, 1)

        x_proj = self.weight_connection(x)  # (B, L, out_channels)
        out_per = self.out_channels // self.module_num
        splits = torch.split(x_proj, out_per, dim=-1)

        gates = self.op_gates()  # (module_num,)
        outputs = []
        for i, (module, split) in enumerate(zip(self.modules_dict.values(), splits)):
            y = module(split)
            outputs.append(y * gates[i])
        y = torch.cat(outputs, dim=-1)

        if hasattr(self, "skip_connection"):
            y = y + self.skip_connection(x)
        return y

    def export_operator_importance(self) -> List[Dict[str, Any]]:
        probs = self.op_probs().detach().cpu().tolist()
        return [
            {"op_uid": uid, "module_key": k, "prob": float(p)}
            for (uid, k, p) in zip(self.op_uids, self.modules_dict.keys(), probs)
        ]


class FeatureExtractorLayer(nn.Module):
    def __init__(self, feature_modules: nn.ModuleDict, *, in_channels: int):
        super().__init__()
        self.feature_modules = feature_modules
        self.in_channels = int(in_channels)
        self.weight_connection = nn.Linear(self.in_channels, self.in_channels)
        self.pre_norm = nn.InstanceNorm1d(self.in_channels)
        self.norm = nn.BatchNorm1d(self.in_channels * len(self.feature_modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C)
        x_bcl = x.permute(0, 2, 1)
        x_bcl = self.pre_norm(x_bcl)
        x = x_bcl.permute(0, 2, 1)

        x = self.weight_connection(x)  # (B,L,C)
        x_bcl = x.permute(0, 2, 1)  # (B,C,L) for feature ops
        feats = [m(x_bcl) for m in self.feature_modules.values()]  # each (B,C,1)
        res = torch.cat(feats, dim=1).squeeze(-1)  # (B, C*F)
        res = res.reshape(res.size(0), -1)
        return self.norm(res)


class Classifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(x)


class TransparentSignalProcessingNetwork(nn.Module):
    def __init__(
        self,
        *,
        args: TSPNArgs,
        layer_modules: List[nn.ModuleDict],
        layer_op_uids: List[List[str]],
        feature_tokens: List[str],
        disabled_ops: Optional[Dict[str, float]] = None,
        layer_gate_temperatures: Optional[List[float]] = None,
    ):
        super().__init__()
        self.args = args
        self.layer_num = len(layer_modules)
        self.disabled_ops = disabled_ops or {}

        in_channels = int(args.in_channels)
        out_channels = int(args.out_channels * args.scale)

        self.signal_layers = nn.ModuleList()
        for i in range(self.layer_num):
            modules = layer_modules[i]
            uids = layer_op_uids[i]
            gate_temp = (
                float(layer_gate_temperatures[i])
                if layer_gate_temperatures and i < len(layer_gate_temperatures)
                else 1.0
            )
            self.signal_layers.append(
                SignalProcessingLayer(
                    layer_idx=i + 1,
                    modules=modules,
                    op_uids=uids,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skip_connection=bool(args.skip_connection),
                    gate_temperature=gate_temp,
                    disabled_ops=self.disabled_ops,
                )
            )
            in_channels = out_channels

        feat_modules = nn.ModuleDict({t: make_feature(t) for t in feature_tokens})
        self.feature_layer = FeatureExtractorLayer(feat_modules, in_channels=in_channels)
        self.classifier = Classifier(in_dim=in_channels * len(feature_tokens), num_classes=args.num_classes)

        self.to(torch.device(args.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.signal_layers:
            x = layer(x)
        feats = self.feature_layer(x)
        return self.classifier(feats)

    def export_operator_importance(self) -> Dict[str, Any]:
        return {
            "layers": [
                {"layer": i + 1, "operators": layer.export_operator_importance()}
                for i, layer in enumerate(self.signal_layers)
            ]
        }

    def export_wavefilters_params(self, *, fs_hz: Optional[float] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"layers": [], "fs_hz": fs_hz}
        for i, layer in enumerate(self.signal_layers):
            layer_info = {"layer": i + 1, "wavefilters": []}
            for uid, module_key, module in zip(layer.op_uids, layer.modules_dict.keys(), layer.modules_dict.values()):
                if isinstance(module, WaveFilters):
                    fc = module.fc_norm().detach().cpu()
                    fb = module.fb_norm().detach().cpu()
                    entry = {
                        "op_uid": uid,
                        "module_key": module_key,
                        "fc_norm": fc.tolist(),
                        "fb_norm": fb.tolist(),
                    }
                    if fs_hz is not None:
                        entry["fc_hz"] = (fc * float(fs_hz)).tolist()
                        entry["fb_hz"] = (fb * float(fs_hz)).tolist()
                    layer_info["wavefilters"].append(entry)
            out["layers"].append(layer_info)
        return out

    def init_weights_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """Initialize learnable parameters from DAG/bridge metadata.

        Supported metadata keys
        -----------------------
        - ``wf_by_op_uid``: {op_uid: {fc_hz, fb_hz, fs_hz}}
        """
        if not isinstance(metadata, dict):
            return
        wf_map = metadata.get("wf_by_op_uid")
        if not isinstance(wf_map, dict) or not wf_map:
            return

        def _logit(p: "torch.Tensor") -> "torch.Tensor":
            p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
            return torch.log(p / (1.0 - p))

        def _softplus_inv(y: "torch.Tensor") -> "torch.Tensor":
            # Inverse of softplus: x = log(exp(y) - 1)
            y = torch.clamp(y, 1e-8)
            return torch.log(torch.expm1(y))

        import torch

        with torch.no_grad():
            for layer in getattr(self, "signal_layers", []):
                for uid, module in zip(layer.op_uids, layer.modules_dict.values()):
                    if not isinstance(module, WaveFilters):
                        continue
                    spec = wf_map.get(uid)
                    if not isinstance(spec, dict):
                        continue
                    fs_hz = spec.get("fs_hz")
                    fc_hz = spec.get("fc_hz")
                    fb_hz = spec.get("fb_hz")
                    if fs_hz is None or fc_hz is None or fb_hz is None:
                        continue

                    fs = float(fs_hz)
                    if fs <= 0:
                        continue
                    fc_norm = float(fc_hz) / fs
                    fb_norm = float(fb_hz) / fs

                    # Clamp to operator's normalized domain.
                    fc_norm = min(max(fc_norm, 1e-4), 0.5 - 1e-4)
                    fb_norm = max(fb_norm, 1e-6)

                    # WaveFilters.fc_norm = 0.5*sigmoid(_fc)  => sigmoid(_fc) = 2*fc_norm
                    target_sig = torch.full_like(module._fc, 2.0 * fc_norm)
                    module._fc.copy_(_logit(target_sig))

                    # WaveFilters.fb_norm = softplus(_fb) + 1e-6  => _fb = softplus_inv(fb_norm-1e-6)
                    target_fb = torch.full_like(module._fb, fb_norm - 1e-6)
                    module._fb.copy_(_softplus_inv(target_fb))
