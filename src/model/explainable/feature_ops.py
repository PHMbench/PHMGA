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
    # A small epsilon with dtype/device matching x.
    return torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)


def _moment(x: torch.Tensor, k: int) -> torch.Tensor:
    # Central moment along the last dimension, keepdim=True.
    mu = torch.mean(x, dim=-1, keepdim=True)
    return torch.mean((x - mu) ** k, dim=-1, keepdim=True)


def make_feature(name: str) -> nn.Module:
    name = name.strip()
    if name == "Mean":
        return FeatureExtractionBase("mean", lambda x: torch.mean(x, dim=-1, keepdim=True))
    if name == "Std":
        return FeatureExtractionBase("std", lambda x: torch.std(x, dim=-1, keepdim=True))
    if name == "Var":
        return FeatureExtractionBase("var", lambda x: torch.var(x, dim=-1, keepdim=True))
    if name == "Entropy":
        # Soft entropy proxy; stable for gradients
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
        # E[(x-mu)^4] / (Var(x)^2 + eps)
        def _kurt(x: torch.Tensor) -> torch.Tensor:
            m2 = _moment(x, 2)
            m4 = _moment(x, 4)
            return m4 / (m2**2 + _eps_like(x))

        return FeatureExtractionBase("kurtosis", _kurt)
    if name == "RMS":
        return FeatureExtractionBase("rms", lambda x: torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x)))
    if name == "CrestFactor":
        # max(|x|) / (rms + eps)
        def _crest(x: torch.Tensor) -> torch.Tensor:
            peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x))
            return peak / (rms + _eps_like(x))

        return FeatureExtractionBase("crest_factor", _crest)
    if name == "ClearanceFactor":
        # Unified baseline uses max(x)/mean(abs(x)); keep stable with abs + eps.
        def _clr(x: torch.Tensor) -> torch.Tensor:
            peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            denom = torch.mean(torch.abs(x), dim=-1, keepdim=True) + _eps_like(x)
            return peak / denom

        return FeatureExtractionBase("clearance_factor", _clr)
    if name == "Skewness":
        # E[(x-mu)^3] / (Std^3 + eps)
        def _skew(x: torch.Tensor) -> torch.Tensor:
            m2 = _moment(x, 2)
            m3 = _moment(x, 3)
            std = torch.sqrt(m2 + _eps_like(x))
            return m3 / (std**3 + _eps_like(x))

        return FeatureExtractionBase("skewness", _skew)
    if name == "ShapeFactor":
        # rms / (mean(|x|) + eps)
        def _shape(x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + _eps_like(x))
            denom = torch.mean(torch.abs(x), dim=-1, keepdim=True) + _eps_like(x)
            return rms / denom

        return FeatureExtractionBase("shape_factor", _shape)
    if name == "SpectralKurtosis":
        # Kurtosis over the (real) magnitude spectrum along frequency axis.
        def _sk(x: torch.Tensor) -> torch.Tensor:
            mag = torch.abs(torch.fft.rfft(x, dim=-1, norm="ortho"))
            m2 = _moment(mag, 2)
            m4 = _moment(mag, 4)
            return m4 / (m2**2 + _eps_like(mag))

        return FeatureExtractionBase("spectral_kurtosis", _sk)
    raise ValueError(f"Unknown feature token: {name}")
