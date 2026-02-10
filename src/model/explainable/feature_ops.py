from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class FeatureExtractionBase(nn.Module):
    """Feature extractor that maps (B, C, L) -> (B, C, 1)."""

    def __init__(self, method_name: str, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.method_name = method_name
        self._fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)


def _eps_like(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)


def _moment(x: torch.Tensor, k: int) -> torch.Tensor:
    mu = torch.mean(x, dim=-1, keepdim=True)
    return torch.mean((x - mu) ** k, dim=-1, keepdim=True)


def _safe_diff(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] <= 1:
        return torch.zeros_like(x)
    return x[..., 1:] - x[..., :-1]


def _spectrum_mag(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.fft.rfft(x, dim=-1, norm="ortho"))


def make_feature(name: str) -> nn.Module:
    name = name.strip()
    if name == "Mean":
        return FeatureExtractionBase("mean", lambda x: torch.mean(x, dim=-1, keepdim=True))
    if name == "Std":
        return FeatureExtractionBase("std", lambda x: torch.std(x, dim=-1, keepdim=True))
    if name == "Var":
        return FeatureExtractionBase("var", lambda x: torch.var(x, dim=-1, keepdim=True))
    if name == "Entropy":
        return FeatureExtractionBase(
            "entropy",
            lambda x: (torch.softmax(x, dim=-1) * torch.log_softmax(x, dim=-1)).sum(dim=-1, keepdim=True).neg(),
        )
    if name == "Max":
        return FeatureExtractionBase("max", lambda x: torch.max(x, dim=-1, keepdim=True)[0])
    if name == "Min":
        return FeatureExtractionBase("min", lambda x: torch.min(x, dim=-1, keepdim=True)[0])
    if name == "AbsMean":
        return FeatureExtractionBase("abs_mean", lambda x: torch.mean(torch.abs(x), dim=-1, keepdim=True))
    if name == "Kurtosis":
        def _kurt(x: torch.Tensor) -> torch.Tensor:
            m2 = _moment(x, 2)
            m4 = _moment(x, 4)
            return m4 / (m2**2 + _eps_like(x))

        return FeatureExtractionBase("kurtosis", _kurt)
    if name == "RMS":
        return FeatureExtractionBase("rms", lambda x: torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x)))
    if name == "CrestFactor":
        def _crest(x: torch.Tensor) -> torch.Tensor:
            peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x))
            return peak / (rms + _eps_like(x))

        return FeatureExtractionBase("crest_factor", _crest)
    if name == "ClearanceFactor":
        def _clr(x: torch.Tensor) -> torch.Tensor:
            peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            denom = torch.mean(torch.abs(x), dim=-1, keepdim=True) + _eps_like(x)
            return peak / denom

        return FeatureExtractionBase("clearance_factor", _clr)
    if name == "Skewness":
        def _skew(x: torch.Tensor) -> torch.Tensor:
            m2 = _moment(x, 2)
            m3 = _moment(x, 3)
            std = torch.sqrt(m2 + _eps_like(x))
            return m3 / (std**3 + _eps_like(x))

        return FeatureExtractionBase("skewness", _skew)
    if name == "ShapeFactor":
        def _shape(x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x))
            denom = torch.mean(torch.abs(x), dim=-1, keepdim=True) + _eps_like(x)
            return rms / denom

        return FeatureExtractionBase("shape_factor", _shape)
    if name == "SpectralKurtosis":
        def _spectral_kurt(x: torch.Tensor) -> torch.Tensor:
            mag = _spectrum_mag(x)
            m2 = _moment(mag, 2)
            m4 = _moment(mag, 4)
            return m4 / (m2**2 + _eps_like(mag))

        return FeatureExtractionBase("spectral_kurtosis", _spectral_kurt)
    if name == "PeakToPeak":
        return FeatureExtractionBase(
            "peak_to_peak", lambda x: torch.max(x, dim=-1, keepdim=True)[0] - torch.min(x, dim=-1, keepdim=True)[0]
        )
    if name == "ZeroCrossingRate":
        def _zcr(x: torch.Tensor) -> torch.Tensor:
            signs = torch.sign(x)
            crossings = torch.abs(_safe_diff(signs))
            return torch.mean(crossings, dim=-1, keepdim=True) * 0.5

        return FeatureExtractionBase("zero_crossing_rate", _zcr)
    if name == "SpectralCentroid":
        def _centroid(x: torch.Tensor) -> torch.Tensor:
            mag = _spectrum_mag(x)
            F = mag.shape[-1]
            freq_idx = torch.linspace(0.0, 1.0, F, device=x.device, dtype=x.dtype).view(1, 1, F)
            num = torch.sum(freq_idx * mag, dim=-1, keepdim=True)
            den = torch.sum(mag, dim=-1, keepdim=True) + _eps_like(mag)
            return num / den

        return FeatureExtractionBase("spectral_centroid", _centroid)
    if name == "SpectralSkewness":
        def _s_skew(x: torch.Tensor) -> torch.Tensor:
            mag = _spectrum_mag(x)
            m2 = _moment(mag, 2)
            m3 = _moment(mag, 3)
            std = torch.sqrt(m2 + _eps_like(mag))
            return m3 / (std**3 + _eps_like(mag))

        return FeatureExtractionBase("spectral_skewness", _s_skew)
    if name == "SpectralFlatness":
        def _flatness(x: torch.Tensor) -> torch.Tensor:
            mag = _spectrum_mag(x) + _eps_like(x)
            gm = torch.exp(torch.mean(torch.log(mag), dim=-1, keepdim=True))
            am = torch.mean(mag, dim=-1, keepdim=True)
            return gm / (am + _eps_like(mag))

        return FeatureExtractionBase("spectral_flatness", _flatness)
    if name == "HjorthActivity":
        return FeatureExtractionBase("hjorth_activity", lambda x: torch.var(x, dim=-1, keepdim=True))
    if name == "HjorthMobility":
        def _mobility(x: torch.Tensor) -> torch.Tensor:
            dx = _safe_diff(x)
            activity = torch.var(x, dim=-1, keepdim=True)
            return torch.sqrt(torch.var(dx, dim=-1, keepdim=True) / (activity + _eps_like(x)))

        return FeatureExtractionBase("hjorth_mobility", _mobility)
    if name == "HjorthComplexity":
        def _complexity(x: torch.Tensor) -> torch.Tensor:
            dx = _safe_diff(x)
            ddx = _safe_diff(dx)
            mobility = torch.sqrt(torch.var(dx, dim=-1, keepdim=True) / (torch.var(x, dim=-1, keepdim=True) + _eps_like(x)))
            return torch.sqrt(torch.var(ddx, dim=-1, keepdim=True) / (torch.var(dx, dim=-1, keepdim=True) + _eps_like(x))) / (
                mobility + _eps_like(x)
            )

        return FeatureExtractionBase("hjorth_complexity", _complexity)
    if name == "CrestFactorDelta":
        def _crest_delta(x: torch.Tensor) -> torch.Tensor:
            dx = _safe_diff(x)
            rms_delta = torch.sqrt(torch.mean(dx**2, dim=-1, keepdim=True) + _eps_like(x))
            denom = torch.mean(torch.abs(x), dim=-1, keepdim=True) + _eps_like(x)
            return rms_delta / denom

        return FeatureExtractionBase("crest_factor_delta", _crest_delta)
    if name == "KurtosisDelta":
        def _kurt_delta(x: torch.Tensor) -> torch.Tensor:
            dx = _safe_diff(x)
            m2 = _moment(dx, 2)
            m4 = _moment(dx, 4)
            return m4 / (m2**2 + _eps_like(x))

        return FeatureExtractionBase("kurtosis_delta", _kurt_delta)
    raise ValueError(f"Unknown feature token: {name}")
