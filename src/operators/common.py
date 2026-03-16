"""Shared numeric helpers for the unified operator modules."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn.functional as torch_f
except ModuleNotFoundError:  # pragma: no cover - exercised by env validation tests instead
    torch = None
    torch_f = None


def as_channel_first(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def flatten_numeric(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def safe_welch_nperseg(length: int, requested: int) -> int:
    return max(2, min(int(requested), int(length)))


def safe_noverlap(nperseg: int, requested: int) -> int:
    return max(0, min(int(requested), max(int(nperseg) - 1, 0)))


def scalar_feature(value: float) -> np.ndarray:
    return np.asarray([float(value)], dtype=float)


def spectral_matrix(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.mean(axis=-1)


def concat_symbolic(x_sym: Any, op_name: str) -> str:
    if isinstance(x_sym, (list, tuple)):
        return f"{op_name}({', '.join(map(str, x_sym))})"
    return f"{op_name}({x_sym})"


def require_torch():
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for operator-level forward_pt execution.")
    return torch


def require_torch_f():
    if torch_f is None:
        raise ModuleNotFoundError("PyTorch is required for operator-level forward_pt execution.")
    return torch_f


def validate_finite_tensor(x: Any, *, op_name: str) -> Any:
    torch_module = require_torch()
    tensor = x if torch_module.is_tensor(x) else torch_module.as_tensor(x)
    if not bool(torch_module.isfinite(tensor).all()):
        bad_count = int((~torch_module.isfinite(tensor)).sum().item())
        raise ValueError(f"{op_name} received non-finite tensor values ({bad_count} invalid entries).")
    return tensor


def ensure_float_tensor(x: Any, *, like: Any | None = None, op_name: str = "operator input"):
    torch_module = require_torch()
    tensor = x if torch_module.is_tensor(x) else torch_module.as_tensor(x)
    if like is not None and torch_module.is_tensor(like):
        target_dtype = like.dtype if like.is_floating_point() else torch_module.float32
        tensor = tensor.to(device=like.device, dtype=target_dtype)
    elif not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch_module.float32)
    validate_finite_tensor(tensor, op_name=op_name)
    return tensor


def ensure_channel_first_tensor(x: Any, *, min_rank: int = 1, op_name: str = "operator input"):
    tensor = ensure_float_tensor(x, op_name=op_name)
    if tensor.ndim == 0:
        raise ValueError(f"{op_name} expects at least rank-1 numeric input, got scalar input.")
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim < min_rank:
        raise ValueError(f"{op_name} expects rank>={min_rank} input after channel normalization, got rank {tensor.ndim}.")
    return tensor


def scalar_output_tensor(value: float | Any, *, like: Any | None = None):
    torch_module = require_torch()
    if torch_module.is_tensor(value):
        tensor = ensure_float_tensor(value, like=like, op_name="scalar_output")
    else:
        tensor = torch_module.tensor([float(value)], dtype=torch_module.float32)
    if like is not None and torch_module.is_tensor(like):
        target_dtype = like.dtype if like.is_floating_point() else torch_module.float32
        tensor = tensor.to(device=like.device, dtype=target_dtype)
    return tensor.reshape(1)


def bridge_pt_via_numpy(
    x: Any,
    fn_np: Callable[..., np.ndarray],
    *,
    like: Any | None = None,
    op_name: str = "operator input",
    **kwargs: Any,
):
    tensor = ensure_float_tensor(x, like=like, op_name=op_name)
    result = fn_np(to_numpy_cpu(tensor), **kwargs)
    return from_numpy_like(result, like=tensor if like is None else like)


def as_channel_first_pt(x: Any):
    return ensure_channel_first_tensor(x, min_rank=1, op_name="channel_first_tensor")


def flatten_numeric_pt(x: Any):
    tensor = ensure_float_tensor(x, op_name="numeric_tensor")
    return tensor.reshape(-1)


def scalar_feature_pt(value: float, *, like: Any | None = None):
    return scalar_output_tensor(value, like=like)


def spectral_matrix_pt(x: Any):
    tensor = ensure_channel_first_tensor(x, min_rank=1, op_name="spectral_tensor")
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim == 2:
        return tensor
    return tensor.mean(dim=-1)


def to_numpy_cpu(x: Any) -> np.ndarray:
    torch_module = require_torch()
    tensor = x if torch_module.is_tensor(x) else torch_module.as_tensor(x, dtype=torch_module.float32)
    return tensor.detach().cpu().numpy()


def from_numpy_like(array: np.ndarray, *, like: Any | None = None):
    torch_module = require_torch()
    tensor = torch_module.from_numpy(np.ascontiguousarray(np.asarray(array)))
    if like is not None and torch_module.is_tensor(like):
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    else:
        tensor = tensor.to(dtype=torch_module.float32)
    return tensor
