from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OpIdentity:
    token: Literal["I"] = "I"


@dataclass(frozen=True)
class OpHilbertEnvelope:
    token: Literal["HT"] = "HT"


@dataclass(frozen=True)
class OpFFTMag:
    token: Literal["FFT"] = "FFT"


@dataclass(frozen=True)
class OpWaveFilters:
    token: Literal["WF"] = "WF"


class Identity(nn.Module):
    op_token: str = "I"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HilbertEnvelope(nn.Module):
    """Hilbert envelope (real-valued) for each channel.

    Input/Output: (B, L, C) float -> (B, L, C) float
    """

    op_token: str = "HT"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if x.ndim != 3:
            raise ValueError(f"HilbertEnvelope expects (B,L,C), got {tuple(x.shape)}")
        x_bcL = x.permute(0, 2, 1)  # (B, C, L)
        L = x_bcL.shape[-1]
        Xf = torch.fft.fft(x_bcL, dim=-1)

        h = torch.zeros(L, device=x.device, dtype=Xf.dtype)
        if L % 2 == 0:
            h[0] = 1
            h[L // 2] = 1
            h[1 : L // 2] = 2
        else:
            h[0] = 1
            h[1 : (L + 1) // 2] = 2

        analytic = torch.fft.ifft(Xf * h, dim=-1)
        env = torch.abs(analytic).real
        return env.permute(0, 2, 1)  # (B, L, C)


class FFTMagnitude(nn.Module):
    """FFT magnitude with length-preserving projection.

    The magnitude spectrum length is F=L//2+1. We interpolate it back to L so
    the operator contract stays (B, L, C). This is a pragmatic choice to keep
    a consistent tensor contract for downstream nn modules.
    """

    op_token: str = "FFT"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FFTMagnitude expects (B,L,C), got {tuple(x.shape)}")
        B, L, C = x.shape
        mag = torch.abs(torch.fft.rfft(x, dim=1, norm="ortho"))  # (B, F, C)
        mag_bcf = mag.permute(0, 2, 1)  # (B, C, F)
        # Interpolate to length L
        mag_bcl = F.interpolate(mag_bcf, size=L, mode="linear", align_corners=False)
        return mag_bcl.permute(0, 2, 1)  # (B, L, C)


class WaveFilters(nn.Module):
    """Channel-wise learnable Gaussian filters in the frequency domain.

    Uses normalized frequency omega in [0, 0.5]. Output stays in time domain.
    """

    op_token: str = "WF"

    def __init__(
        self,
        *,
        channels: int,
        f_c_mu: float = 0.0,
        f_c_sigma: float = 0.1,
        f_b_mu: float = 0.0,
        f_b_sigma: float = 0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        # Unconstrained parameters; transformed to valid ranges in forward.
        self._fc = nn.Parameter(torch.empty(self.channels))
        self._fb = nn.Parameter(torch.empty(self.channels))
        nn.init.normal_(self._fc, mean=float(f_c_mu), std=float(f_c_sigma))
        nn.init.normal_(self._fb, mean=float(f_b_mu), std=float(f_b_sigma))

    def fc_norm(self) -> torch.Tensor:
        # Map to (0, 0.5)
        return 0.5 * torch.sigmoid(self._fc)

    def fb_norm(self) -> torch.Tensor:
        # Positive bandwidth
        return F.softplus(self._fb) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"WaveFilters expects (B,L,C), got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.channels:
            raise ValueError(f"WaveFilters channels mismatch: expected {self.channels}, got {C}")

        X = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, F, C)
        F_len = X.shape[1]
        omega = torch.linspace(0, 0.5, F_len, device=x.device).view(1, F_len, 1)  # (1,F,1)
        fc = self.fc_norm().view(1, 1, C)
        fb = self.fb_norm().view(1, 1, C)
        filt = torch.exp(-((omega - fc) / (2.0 * fb)) ** 2)  # (1,F,C)
        Y = X * filt
        y = torch.fft.irfft(Y, n=L, dim=1, norm="ortho")
        return y.real


def make_op(token: str, *, channels: int, **kwargs) -> nn.Module:
    token = token.strip()
    if token == "I":
        return Identity(channels=channels)
    if token == "HT":
        return HilbertEnvelope(channels=channels)
    if token == "FFT":
        return FFTMagnitude(channels=channels)
    if token == "WF":
        return WaveFilters(channels=channels, **kwargs)
    raise ValueError(f"Unknown TSPN op token: {token}")

