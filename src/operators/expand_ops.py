"""EXPAND operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as scipy_signal

from .base import BaseIsomorphicOperator, OperatorSpec
from .common import (
    as_channel_first,
    ensure_channel_first_tensor,
    require_torch,
    require_torch_f,
    safe_noverlap,
    safe_welch_nperseg,
)


class STFTOperator(BaseIsomorphicOperator):
    """Short-time Fourier transform that expands one signal into time-frequency patches."""

    spec = OperatorSpec(
        op_uid="signal.stft",
        op_name="stft",
        name="STFT",
        schema_category="EXPAND",
        rank_class="rank_up",
        description="Expand a waveform into a time-frequency magnitude tensor using STFT.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_time_frequency", "rank_behavior": "expand"},
        param_schema={"nperseg": "int", "noverlap": "int"},
        param_defaults={"nperseg": 128, "noverlap": 64},
        param_docs={
            "nperseg": "Segment length used by STFT before overlap.",
            "noverlap": "Overlap size between adjacent STFT segments.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxFxS",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use when the workflow needs time-frequency evidence rather than one global spectrum.",
        llm_tunable_params=["nperseg", "noverlap"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        nperseg = safe_welch_nperseg(array.shape[-1], int(kwargs.get("nperseg", 128)))
        noverlap = safe_noverlap(nperseg, int(kwargs.get("noverlap", nperseg // 2)))
        outputs = []
        for channel in array:
            _, _, zxx = scipy_signal.stft(channel, nperseg=nperseg, noverlap=noverlap, boundary=None)
            outputs.append(np.abs(zxx))
        return np.stack(outputs, axis=0)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"stft({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        nperseg = safe_welch_nperseg(array.shape[-1], int(kwargs.get("nperseg", 128)))
        noverlap = safe_noverlap(nperseg, int(kwargs.get("noverlap", nperseg // 2)))
        hop_length = max(1, nperseg - noverlap)
        window = torch.hann_window(nperseg, device=array.device, dtype=array.dtype)
        spectrum = torch.stft(
            array,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=False,
            return_complex=True,
        )
        return torch.abs(spectrum) / window.sum()


class PatchOperator(BaseIsomorphicOperator):
    """Local patch extraction for window-level feature diversity."""

    spec = OperatorSpec(
        op_uid="signal.patch",
        op_name="patch",
        name="Patch",
        schema_category="EXPAND",
        rank_class="rank_up",
        description="Split a waveform into overlapping local patches for local feature extraction.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_patch_signal", "rank_behavior": "expand"},
        param_schema={"patch_length": "int", "stride": "int"},
        param_defaults={"patch_length": 128, "stride": 64},
        param_docs={
            "patch_length": "Length of each local signal patch.",
            "stride": "Step size between adjacent patches.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxPxL",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use when localized transient behavior matters more than one global summary.",
        llm_tunable_params=["patch_length", "stride"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        patch_length = max(8, int(kwargs.get("patch_length", 128)))
        stride = max(1, int(kwargs.get("stride", 64)))
        patches_by_channel = []
        for channel in array:
            if channel.shape[-1] < patch_length:
                padded = np.pad(channel, (0, patch_length - channel.shape[-1]))
                patches = [padded]
            else:
                starts = range(0, channel.shape[-1] - patch_length + 1, stride)
                patches = [channel[start : start + patch_length] for start in starts]
            patches_by_channel.append(np.stack(patches, axis=0))
        return np.stack(patches_by_channel, axis=0)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"patch({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        torch_f = require_torch_f()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        patch_length = max(8, int(kwargs.get("patch_length", 128)))
        stride = max(1, int(kwargs.get("stride", 64)))
        patches_by_channel = []
        for channel in array:
            if channel.shape[-1] < patch_length:
                padded = torch_f.pad(channel, (0, patch_length - channel.shape[-1]))
                patches = padded.unsqueeze(0)
            else:
                patches = channel.unfold(0, patch_length, stride)
            patches_by_channel.append(patches)
        return torch.stack(patches_by_channel, dim=0)


def get_expand_operators() -> Tuple[BaseIsomorphicOperator, ...]:
    """Return the runnable EXPAND operators."""

    return (
        STFTOperator(),
        PatchOperator(),
    )
