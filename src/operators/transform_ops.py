"""TRANSFORM operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy import signal as scipy_signal

from .base import BaseIsomorphicOperator, OperatorSpec
from .common import (
    as_channel_first,
    bridge_pt_via_numpy,
    ensure_channel_first_tensor,
    require_torch,
    safe_welch_nperseg,
)


class NormalizeOperator(BaseIsomorphicOperator):
    """Per-channel normalization before downstream transforms."""

    spec = OperatorSpec(
        op_uid="signal.normalize",
        op_name="normalize",
        name="Normalize",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Per-channel z-score normalization that preserves the time axis.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"eps": "float"},
        param_defaults={"eps": 1e-6},
        param_docs={"eps": "Small epsilon added to the per-channel standard deviation to avoid divide-by-zero."},
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use before spectral transforms when channel amplitudes vary across sensors.",
        llm_tunable_params=["eps"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        eps = float(kwargs.get("eps", 1e-6))
        mean = array.mean(axis=-1, keepdims=True)
        std = array.std(axis=-1, keepdims=True) + eps
        return (array - mean) / std

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"normalize({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        eps = float(kwargs.get("eps", 1e-6))
        mean = array.mean(dim=-1, keepdim=True)
        std = array.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (array - mean) / std


class FFTMagnitudeOperator(BaseIsomorphicOperator):
    """Frequency-domain transform that preserves per-channel structure."""

    spec = OperatorSpec(
        op_uid="signal.fft_mag",
        op_name="fft",
        name="FFT Magnitude",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Convert a time-domain channel signal into a one-sided magnitude spectrum.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_spectrum", "rank_behavior": "preserve"},
        input_shape_rule="CxT",
        output_shape_rule="CxF",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="A classic PHM bridge from time-domain waveforms into spectral features.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        array = as_channel_first(x)
        return np.abs(np.fft.rfft(array, axis=-1))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"abs(rfft({x_sym}))"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        return torch.abs(torch.fft.rfft(array, dim=-1))


class FilterOperator(BaseIsomorphicOperator):
    """Band-limited filtering with conservative defaults and bounded tuning."""

    spec = OperatorSpec(
        op_uid="signal.filter",
        op_name="filter",
        name="Filter",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Apply a Butterworth low/high/band-pass filter while preserving signal layout.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={
            "mode": "str",
            "fs": "float",
            "low_cut_hz": "float",
            "high_cut_hz": "float",
            "order": "int",
        },
        param_defaults={"mode": "bandpass", "low_cut_hz": 5.0, "high_cut_hz": 200.0, "order": 4},
        param_docs={
            "mode": "One of lowpass, highpass, or bandpass.",
            "fs": "Sampling rate in Hz.",
            "low_cut_hz": "Low cutoff frequency in Hz.",
            "high_cut_hz": "High cutoff frequency in Hz.",
            "order": "Butterworth filter order.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use before envelope or spectral estimation when the fault band is approximately known.",
        llm_tunable_params=["low_cut_hz", "high_cut_hz", "order"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        fs = float(kwargs.get("fs", 1.0))
        nyquist = max(fs / 2.0, 1.0)
        mode = str(kwargs.get("mode", "bandpass")).strip().lower()
        low = max(float(kwargs.get("low_cut_hz", 5.0)), 1e-6)
        high = min(float(kwargs.get("high_cut_hz", nyquist * 0.8)), nyquist * 0.95)
        order = max(1, int(kwargs.get("order", 4)))

        if mode == "lowpass":
            wn: Any = min(high, nyquist * 0.95) / nyquist
            btype = "lowpass"
        elif mode == "highpass":
            wn = min(low, nyquist * 0.95) / nyquist
            btype = "highpass"
        else:
            if high <= low:
                high = min(nyquist * 0.95, low * 2.0)
            if high <= low:
                return array
            wn = [low / nyquist, high / nyquist]
            btype = "bandpass"

        sos = scipy_signal.butter(order, wn, btype=btype, output="sos")
        if array.shape[-1] <= 3 * order:
            return scipy_signal.sosfilt(sos, array, axis=-1)
        return scipy_signal.sosfiltfilt(sos, array, axis=-1)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"filter({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        return bridge_pt_via_numpy(x, self.forward_np, op_name=self.spec.op_uid, **kwargs)


class HilbertEnvelopeOperator(BaseIsomorphicOperator):
    """Envelope transform for amplitude modulation cues."""

    spec = OperatorSpec(
        op_uid="signal.hilbert_envelope",
        op_name="hilbert_envelope",
        name="Hilbert Envelope",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Compute the analytic-signal envelope with the Hilbert transform.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Often useful after band filtering to isolate bearing or gearbox modulation patterns.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        array = as_channel_first(x)
        return np.abs(scipy_signal.hilbert(array, axis=-1))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"abs(hilbert({x_sym}))"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        length = array.shape[-1]
        spectrum = torch.fft.fft(array, dim=-1)
        multiplier = torch.zeros(length, device=array.device, dtype=array.dtype)
        if length % 2 == 0:
            multiplier[0] = 1.0
            multiplier[length // 2] = 1.0
            multiplier[1 : length // 2] = 2.0
        else:
            multiplier[0] = 1.0
            multiplier[1 : (length + 1) // 2] = 2.0
        analytic = torch.fft.ifft(spectrum * multiplier, dim=-1)
        return torch.abs(analytic)


class PSDOperator(BaseIsomorphicOperator):
    """Welch PSD transform for robust spectral features."""

    spec = OperatorSpec(
        op_uid="signal.psd",
        op_name="psd",
        name="PSD",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Estimate one-sided power spectral density with Welch averaging.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_spectrum", "rank_behavior": "preserve"},
        param_schema={"fs": "float", "nperseg": "int"},
        param_defaults={"nperseg": 128},
        param_docs={
            "fs": "Sampling rate in Hz.",
            "nperseg": "Segment length used by Welch PSD estimation.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxF",
        backend_availability=["np", "pt", "sym"],
        execution_role="fixed",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use when a smoother spectral estimate is preferable to raw FFT magnitude.",
        llm_tunable_params=["nperseg"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        fs = float(kwargs.get("fs", 1.0))
        nperseg = safe_welch_nperseg(array.shape[-1], int(kwargs.get("nperseg", 128)))
        outputs = []
        for channel in array:
            _, pxx = scipy_signal.welch(channel, fs=fs, nperseg=nperseg)
            outputs.append(pxx)
        return np.stack(outputs, axis=0)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"psd({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        fs = float(kwargs.get("fs", 1.0))
        nperseg = safe_welch_nperseg(array.shape[-1], int(kwargs.get("nperseg", 128)))
        hop_length = max(1, nperseg // 2)
        window = torch.hann_window(nperseg, device=array.device, dtype=array.dtype)
        stft = torch.stft(
            array,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().pow(2) / (max(fs, 1e-12) * window.pow(2).sum())
        if power.shape[-2] > 1:
            if nperseg % 2 == 0 and power.shape[-2] > 2:
                power[:, 1:-1, :] *= 2.0
            elif nperseg % 2 == 1:
                power[:, 1:, :] *= 2.0
        return power.mean(dim=-1)


def get_transform_operators() -> Tuple[BaseIsomorphicOperator, ...]:
    """Return the runnable TRANSFORM operators."""

    return (
        NormalizeOperator(),
        FFTMagnitudeOperator(),
        FilterOperator(),
        HilbertEnvelopeOperator(),
        PSDOperator(),
    )
