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
    if name == "RMS":
        return FeatureExtractionBase("rms", lambda x: torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-12))
    raise ValueError(f"Unknown feature token: {name}")

