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


class Normalize(nn.Module):
    op_token: str = "NORM"

    def __init__(self, *, channels: int, method: str = "z_score"):
        super().__init__()
        self.channels = int(channels)
        self.method = str(method).strip().lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Normalize expects (B,L,C), got {tuple(x.shape)}")
        if self.method in {"z_score", "zscore"}:
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True)
            return (x - mean) / (std + 1e-9)
        if self.method in {"min_max", "minmax"}:
            min_v = torch.min(x, dim=1, keepdim=True)[0]
            max_v = torch.max(x, dim=1, keepdim=True)[0]
            return (x - min_v) / (max_v - min_v + 1e-9)
        raise ValueError(f"Unknown normalize method: {self.method!r}")


class Detrend(nn.Module):
    op_token: str = "DT"

    def __init__(self, *, channels: int, type: str = "linear"):
        super().__init__()
        self.channels = int(channels)
        self.type = str(type).strip().lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Detrend expects (B,L,C), got {tuple(x.shape)}")
        if self.type in {"constant"}:
            return x - torch.mean(x, dim=1, keepdim=True)
        if self.type not in {"linear", "line"}:
            raise ValueError(f"Unknown detrend type: {self.type!r}")
        L = x.shape[1]
        t = torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype).view(1, L, 1)
        slope = x[:, -1:, :] - x[:, :1, :]
        trend = x[:, :1, :] + slope * t
        return x - trend


class Integrate(nn.Module):
    op_token: str = "INT"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Integrate expects (B,L,C), got {tuple(x.shape)}")
        return torch.cumsum(x, dim=1)


class Differentiate(nn.Module):
    op_token: str = "DIFF"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Differentiate expects (B,L,C), got {tuple(x.shape)}")
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        # Keep contract length by prepending the first-step derivative.
        d = x[:, 1:, :] - x[:, :-1, :]
        first = d[:, :1, :]
        return torch.cat([first, d], dim=1)


class HilbertEnvelope(nn.Module):
    """Hilbert envelope (real-valued) for each channel.

    Input/Output: (B, L, C) float -> (B, L, C) float
    """

    op_token: str = "HT"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return env.permute(0, 2, 1)


class FFTMagnitude(nn.Module):
    """FFT magnitude with length-preserving projection."""

    op_token: str = "FFT"

    def __init__(self, *, channels: int, align_strategy: str = "interp"):
        super().__init__()
        self.channels = int(channels)
        self.align_strategy = str(align_strategy).strip().lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FFTMagnitude expects (B,L,C), got {tuple(x.shape)}")
        _, L, _ = x.shape
        mag = torch.abs(torch.fft.rfft(x, dim=1, norm="ortho"))  # (B, F, C)
        mag_bcf = mag.permute(0, 2, 1)  # (B, C, F)
        if self.align_strategy in {"interp", "interpolate"}:
            mag_bcl = F.interpolate(mag_bcf, size=L, mode="linear", align_corners=False)
            return mag_bcl.permute(0, 2, 1)

        if self.align_strategy in {"mirror", "mirroring"}:
            if L % 2 == 0:
                tail = mag_bcf[..., 1:-1].flip(-1)
            else:
                tail = mag_bcf[..., 1:].flip(-1)
            full = torch.cat([mag_bcf, tail], dim=-1)
            if full.shape[-1] != L:
                raise ValueError(
                    f"FFTMagnitude mirror alignment produced length {full.shape[-1]} != L={L}"
                )
            return full.permute(0, 2, 1)

        raise ValueError(f"Unknown FFT align_strategy: {self.align_strategy!r}")


class STFTMagnitude(nn.Module):
    op_token: str = "STFT"

    def __init__(self, *, channels: int, n_fft: int = 256, hop_length: int = 128):
        super().__init__()
        self.channels = int(channels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"STFTMagnitude expects (B,L,C), got {tuple(x.shape)}")
        B, L, C = x.shape
        xc = x.permute(0, 2, 1).reshape(B * C, L)
        spec = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            center=True,
        )
        mag = torch.abs(spec).mean(dim=1)  # (B*C, T)
        mag = mag.view(B, C, -1)  # (B,C,T)
        mag = F.interpolate(mag, size=L, mode="linear", align_corners=False)
        return mag.permute(0, 2, 1)


class WaveFilters(nn.Module):
    """Channel-wise learnable Gaussian filters in the frequency domain."""

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
        self._fc = nn.Parameter(torch.empty(self.channels))
        self._fb = nn.Parameter(torch.empty(self.channels))
        nn.init.normal_(self._fc, mean=float(f_c_mu), std=float(f_c_sigma))
        nn.init.normal_(self._fb, mean=float(f_b_mu), std=float(f_b_sigma))

    def fc_norm(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self._fc)

    def fb_norm(self) -> torch.Tensor:
        return F.softplus(self._fb) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"WaveFilters expects (B,L,C), got {tuple(x.shape)}")
        _, L, C = x.shape
        if C != self.channels:
            raise ValueError(f"WaveFilters channels mismatch: expected {self.channels}, got {C}")

        X = torch.fft.rfft(x, dim=1, norm="ortho")
        F_len = X.shape[1]
        omega = torch.linspace(0, 0.5, F_len, device=x.device).view(1, F_len, 1)
        fc = self.fc_norm().view(1, 1, C)
        fb = self.fb_norm().view(1, 1, C)
        filt = torch.exp(-((omega - fc) / (2.0 * fb)) ** 2)
        Y = X * filt
        y = torch.fft.irfft(Y, n=L, dim=1, norm="ortho")
        return y.real


class LogOperation(nn.Module):
    op_token: str = "LOG"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.abs(x) + 1e-9)


class SquOperation(nn.Module):
    op_token: str = "SQU"

    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class SinOperation(nn.Module):
    op_token: str = "SIN"

    def __init__(self, *, channels: int, frequency: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        self.frequency = float(frequency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.frequency * x)


def make_op(token: str, *, channels: int, **kwargs) -> nn.Module:
    token = token.strip()
    if token == "I":
        return Identity(channels=channels)
    if token == "HT":
        return HilbertEnvelope(channels=channels)
    if token == "FFT":
        return FFTMagnitude(channels=channels, **kwargs)
    if token == "WF":
        return WaveFilters(channels=channels, **kwargs)
    if token == "NORM":
        return Normalize(channels=channels, **kwargs)
    if token == "DT":
        return Detrend(channels=channels, **kwargs)
    if token == "INT":
        return Integrate(channels=channels, **kwargs)
    if token == "DIFF":
        return Differentiate(channels=channels, **kwargs)
    if token == "STFT":
        return STFTMagnitude(channels=channels, **kwargs)
    if token == "LOG":
        return LogOperation(channels=channels, **kwargs)
    if token == "SQU":
        return SquOperation(channels=channels, **kwargs)
    if token == "SIN":
        return SinOperation(channels=channels, **kwargs)
    raise ValueError(f"Unknown TSPN op token: {token}")
