"""TRANSFORM operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

import math
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


def _fft_convolution_np(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    signal = as_channel_first(array)
    kernel_array = np.asarray(kernel)
    if kernel_array.ndim == 1:
        kernel_array = kernel_array.reshape(1, -1)
    if kernel_array.shape[0] == 1 and signal.shape[0] > 1:
        kernel_array = np.repeat(kernel_array, signal.shape[0], axis=0)
    if kernel_array.shape[0] != signal.shape[0]:
        raise ValueError("Wavelet kernel channel count must match the input signal channels.")
    signal_fft = np.fft.fft(signal, axis=-1)
    kernel_fft = np.fft.fft(kernel_array, n=signal.shape[-1], axis=-1)
    return np.fft.ifft(signal_fft * kernel_fft, axis=-1).real


def _fft_convolution_pt(x, kernel):
    torch = require_torch()
    signal = ensure_channel_first_tensor(x, min_rank=2, op_name="signal.wavelet")
    kernel_tensor = kernel if torch.is_tensor(kernel) else torch.as_tensor(kernel, device=signal.device)
    if kernel_tensor.ndim == 1:
        kernel_tensor = kernel_tensor.reshape(1, -1)
    if kernel_tensor.shape[0] == 1 and signal.shape[0] > 1:
        kernel_tensor = kernel_tensor.repeat(signal.shape[0], 1)
    if kernel_tensor.shape[0] != signal.shape[0]:
        raise ValueError("Wavelet kernel channel count must match the input signal channels.")
    kernel_tensor = kernel_tensor.to(device=signal.device)
    signal_fft = torch.fft.fft(signal, dim=-1)
    kernel_fft = torch.fft.fft(kernel_tensor, n=signal.shape[-1], dim=-1)
    return torch.fft.ifft(signal_fft * kernel_fft, dim=-1).real


class GaussianWaveFiltersOperator(BaseIsomorphicOperator):
    """Gaussian frequency-domain filtering inspired by WaveFilters."""

    spec = OperatorSpec(
        op_uid="signal.wavefilters",
        op_name="wavefilters",
        name="WaveFilters",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Apply a Gaussian frequency-domain filter with learnable-compatible center and bandwidth parameters.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"center_ratio": "float", "bandwidth_ratio": "float"},
        param_defaults={"center_ratio": 0.15, "bandwidth_ratio": 0.08},
        param_docs={
            "center_ratio": "Normalized center frequency in [0, 0.5].",
            "bandwidth_ratio": "Normalized Gaussian bandwidth in (0, 0.5].",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="trainable",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use as a smooth band-selection transform before envelope, PSD, or aggregate operators.",
        llm_tunable_params=["center_ratio", "bandwidth_ratio"],
    )

    @staticmethod
    def _safe_params(center_ratio: Any, bandwidth_ratio: Any, *, backend: str):
        if backend == "np":
            center = float(np.clip(float(center_ratio), 0.0, 0.5))
            bandwidth = float(np.clip(float(bandwidth_ratio), 1e-4, 0.5))
            return center, bandwidth
        torch = require_torch()
        center = torch.clamp(torch.as_tensor(center_ratio, dtype=torch.float32), 0.0, 0.5)
        bandwidth = torch.clamp(torch.as_tensor(bandwidth_ratio, dtype=torch.float32), 1e-4, 0.5)
        return center, bandwidth

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        center, bandwidth = self._safe_params(
            kwargs.get("center_ratio", 0.15),
            kwargs.get("bandwidth_ratio", 0.08),
            backend="np",
        )
        omega = np.linspace(0.0, 0.5, array.shape[-1] // 2 + 1, dtype=float).reshape(1, -1)
        filters = np.exp(-((omega - center) / (2.0 * bandwidth)) ** 2)
        freq = np.fft.rfft(array, axis=-1)
        filtered_freq = freq * filters
        return np.fft.irfft(filtered_freq, n=array.shape[-1], axis=-1).real

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        center, bandwidth = self._safe_params(
            kwargs.get("center_ratio", 0.15),
            kwargs.get("bandwidth_ratio", 0.08),
            backend="pt",
        )
        center = center.to(device=array.device, dtype=array.dtype)
        bandwidth = bandwidth.to(device=array.device, dtype=array.dtype)
        omega = torch.linspace(0.0, 0.5, array.shape[-1] // 2 + 1, device=array.device, dtype=array.dtype).reshape(1, -1)
        filters = torch.exp(-((omega - center) / (2.0 * bandwidth)) ** 2)
        freq = torch.fft.rfft(array, dim=-1)
        filtered_freq = freq * filters
        return torch.fft.irfft(filtered_freq, n=array.shape[-1], dim=-1).real

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"wavefilters({x_sym})"


class RickerWaveletOperator(BaseIsomorphicOperator):
    """Ricker wavelet filtering over channel-first signals."""

    spec = OperatorSpec(
        op_uid="signal.wavelet_ricker",
        op_name="wavelet_ricker",
        name="Ricker Wavelet",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Convolve each channel with a Ricker wavelet filter.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"sigma": "float"},
        param_defaults={"sigma": 0.25},
        param_docs={"sigma": "Wavelet scale parameter controlling the width of the Ricker kernel."},
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="trainable",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use as a transient-sensitive transform when impulsive faults are expected.",
        llm_tunable_params=["sigma"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        sigma = max(float(kwargs.get("sigma", 0.25)), 1e-4)
        t = np.linspace(-1.0, 1.0, array.shape[-1], dtype=float)
        term1 = 2.0 / (np.sqrt(3.0 * sigma) * np.pi ** 0.25)
        kernel = term1 * (1.0 - (t**2 / sigma**2)) * np.exp(-(t**2) / (2.0 * sigma**2))
        return _fft_convolution_np(array, kernel)

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        sigma = torch.clamp(torch.as_tensor(kwargs.get("sigma", 0.25), device=array.device, dtype=array.dtype), min=1e-4)
        t = torch.linspace(-1.0, 1.0, array.shape[-1], device=array.device, dtype=array.dtype)
        term1 = 2.0 / (torch.sqrt(3.0 * sigma) * (torch.tensor(math.pi, device=array.device, dtype=array.dtype) ** 0.25))
        kernel = term1 * (1.0 - (t**2 / sigma**2)) * torch.exp(-(t**2) / (2.0 * sigma**2))
        return _fft_convolution_pt(array, kernel)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"ricker_wavelet({x_sym})"


class ChirpletWaveletOperator(BaseIsomorphicOperator):
    """Complex chirplet wavelet filtering with real-valued output."""

    spec = OperatorSpec(
        op_uid="signal.wavelet_chirplet",
        op_name="wavelet_chirplet",
        name="Chirplet Wavelet",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Convolve each channel with a chirplet wavelet filter.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"sigma": "float", "omega": "float", "alpha": "float"},
        param_defaults={"sigma": 0.2, "omega": 18.0, "alpha": 4.0},
        param_docs={
            "sigma": "Scale parameter controlling the envelope width.",
            "omega": "Base angular frequency of the chirplet.",
            "alpha": "Quadratic chirp coefficient.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="trainable",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use when a fault signature is expected to sweep in frequency over time.",
        llm_tunable_params=["sigma", "omega", "alpha"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        sigma = max(float(kwargs.get("sigma", 0.2)), 1e-4)
        omega = float(kwargs.get("omega", 18.0))
        alpha = float(kwargs.get("alpha", 4.0))
        t = np.linspace(-1.0, 1.0, array.shape[-1], dtype=float)
        kernel = (1.0 / sigma) * np.exp(-0.5 * (t / sigma) ** 2) * np.exp(-1j * (0.5 * alpha * t**2 + omega * t))
        return _fft_convolution_np(array, kernel)

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        sigma = torch.clamp(torch.as_tensor(kwargs.get("sigma", 0.2), device=array.device, dtype=array.dtype), min=1e-4)
        omega = torch.as_tensor(kwargs.get("omega", 18.0), device=array.device, dtype=array.dtype)
        alpha = torch.as_tensor(kwargs.get("alpha", 4.0), device=array.device, dtype=array.dtype)
        t = torch.linspace(-1.0, 1.0, array.shape[-1], device=array.device, dtype=array.dtype)
        kernel = (1.0 / sigma) * torch.exp(-0.5 * (t / sigma) ** 2) * torch.exp(
            -1j * (0.5 * alpha * t**2 + omega * t)
        )
        return _fft_convolution_pt(array, kernel)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"chirplet_wavelet({x_sym})"


class LaplaceWaveletOperator(BaseIsomorphicOperator):
    """Damped sinusoidal Laplace wavelet filtering."""

    spec = OperatorSpec(
        op_uid="signal.wavelet_laplace",
        op_name="wavelet_laplace",
        name="Laplace Wavelet",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Convolve each channel with a Laplace wavelet filter.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"A": "float", "ep": "float", "tal": "float", "f": "float"},
        param_defaults={"A": 1.0, "ep": 0.2, "tal": 0.1, "f": 4.0},
        param_docs={
            "A": "Wavelet amplitude.",
            "ep": "Damping factor before sigmoid squashing.",
            "tal": "Temporal shift term.",
            "f": "Base frequency.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="trainable",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use as a damped resonant transform when exponentially decaying oscillations are relevant.",
        llm_tunable_params=["A", "ep", "tal", "f"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        amplitude = float(kwargs.get("A", 1.0))
        ep = 1.0 / (1.0 + np.exp(-float(kwargs.get("ep", 0.2))))
        tal = float(kwargs.get("tal", 0.1))
        frequency = float(kwargs.get("f", 4.0))
        p = np.linspace(0.0, 1.0, array.shape[-1], dtype=float)
        q = max(1.0 - ep**2, 1e-6)
        omega = 2.0 * np.pi * frequency
        kernel = amplitude * np.exp((-ep / np.sqrt(q)) * (omega * (p - tal))) * np.sin(omega * (p - tal))
        return _fft_convolution_np(array, kernel)

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        amplitude = torch.as_tensor(kwargs.get("A", 1.0), device=array.device, dtype=array.dtype)
        ep_raw = torch.as_tensor(kwargs.get("ep", 0.2), device=array.device, dtype=array.dtype)
        ep = torch.sigmoid(ep_raw)
        tal = torch.as_tensor(kwargs.get("tal", 0.1), device=array.device, dtype=array.dtype)
        frequency = torch.as_tensor(kwargs.get("f", 4.0), device=array.device, dtype=array.dtype)
        p = torch.linspace(0.0, 1.0, array.shape[-1], device=array.device, dtype=array.dtype)
        q = torch.clamp(1.0 - ep**2, min=1e-6)
        omega = 2.0 * torch.tensor(math.pi, device=array.device, dtype=array.dtype) * frequency
        kernel = amplitude * torch.exp((-ep / torch.sqrt(q)) * (omega * (p - tal))) * torch.sin(omega * (p - tal))
        return _fft_convolution_pt(array, kernel)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"laplace_wavelet({x_sym})"


class MorletWaveletOperator(BaseIsomorphicOperator):
    """Morlet-like complex Gaussian wavelet filtering."""

    spec = OperatorSpec(
        op_uid="signal.wavelet_morlet",
        op_name="wavelet_morlet",
        name="Morlet Wavelet",
        schema_category="TRANSFORM",
        rank_class="rank_same",
        description="Convolve each channel with a Morlet wavelet filter.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "channel_first_signal"},
        output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
        param_schema={"f_b": "float", "f_c": "float"},
        param_defaults={"f_b": 2.0, "f_c": 6.0},
        param_docs={
            "f_b": "Bandwidth parameter of the Gaussian envelope.",
            "f_c": "Center frequency parameter of the complex carrier.",
        },
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "pt", "sym"],
        execution_role="trainable",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use as a wavelet-style time-frequency localization transform.",
        llm_tunable_params=["f_b", "f_c"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        array = as_channel_first(x)
        f_b = max(float(kwargs.get("f_b", 2.0)), 1e-4)
        f_c = float(kwargs.get("f_c", 6.0))
        n = np.linspace(-1.0, 1.0, array.shape[-1], dtype=float)
        kernel = (f_b / np.sqrt(np.pi)) * np.exp(-(f_b**2) * (n**2)) * np.exp(1j * 2.0 * np.pi * f_c * n)
        return _fft_convolution_np(array, kernel)

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        array = ensure_channel_first_tensor(x, min_rank=2, op_name=self.spec.op_uid)
        f_b = torch.clamp(torch.as_tensor(kwargs.get("f_b", 2.0), device=array.device, dtype=array.dtype), min=1e-4)
        f_c = torch.as_tensor(kwargs.get("f_c", 6.0), device=array.device, dtype=array.dtype)
        n = torch.linspace(-1.0, 1.0, array.shape[-1], device=array.device, dtype=array.dtype)
        pi_tensor = torch.tensor(math.pi, device=array.device, dtype=array.dtype)
        kernel = (f_b / torch.sqrt(pi_tensor)) * torch.exp(-(f_b**2) * (n**2)) * torch.exp(1j * 2.0 * pi_tensor * f_c * n)
        return _fft_convolution_pt(array, kernel)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"morlet_wavelet({x_sym})"


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
        GaussianWaveFiltersOperator(),
        RickerWaveletOperator(),
        ChirpletWaveletOperator(),
        LaplaceWaveletOperator(),
        MorletWaveletOperator(),
    )
