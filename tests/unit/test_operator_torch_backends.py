from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
import pytest

from src.operators import get_operator_catalog
from src.operators import aggregate_ops, expand_ops, transform_ops


def _require_torch():
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit failure message
        pytest.fail(f"PyTorch is required in .venv for operator PT tests: {exc}")
    cuda_version = getattr(torch.version, "cuda", None)
    assert cuda_version is not None, "The installed torch wheel is not a CUDA build."
    assert str(cuda_version).startswith("11.8"), f"Expected torch+cu118, got torch built for CUDA {cuda_version}."
    return torch


def _signal_input() -> np.ndarray:
    time = np.linspace(0.0, 1.0, 256, endpoint=False)
    ch1 = np.sin(2 * np.pi * 20 * time) + 0.2 * np.cos(2 * np.pi * 60 * time)
    ch2 = np.cos(2 * np.pi * 15 * time) + 0.1 * np.sin(2 * np.pi * 40 * time)
    return np.stack([ch1, ch2], axis=0).astype(np.float32)


def _spectral_input() -> np.ndarray:
    signal = _signal_input()
    return np.abs(np.fft.rfft(signal, axis=-1)).astype(np.float32)


def _multi_signal_input() -> list[np.ndarray]:
    signal = _signal_input()
    return [signal[0], signal[1]]


def _feature_pair_input() -> list[np.ndarray]:
    return [np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([3.0, 4.0], dtype=np.float32)]


def _decision_input() -> np.ndarray:
    return np.asarray([0.25, 0.75, 1.0], dtype=np.float32)


def _case_for(op_uid: str) -> tuple[Callable[[], Any], dict[str, Any]]:
    mapping = {
        "signal.normalize": (_signal_input, {"eps": 1e-6}),
        "signal.fft_mag": (_signal_input, {}),
        "signal.stft": (_signal_input, {"nperseg": 64, "noverlap": 32}),
        "signal.patch": (_signal_input, {"patch_length": 32, "stride": 16}),
        "signal.filter": (_signal_input, {"fs": 1024.0, "low_cut_hz": 10.0, "high_cut_hz": 200.0, "order": 4}),
        "signal.hilbert_envelope": (_signal_input, {}),
        "signal.psd": (_signal_input, {"fs": 1024.0, "nperseg": 64}),
        "signal.wavefilters": (_signal_input, {"center_ratio": 0.12, "bandwidth_ratio": 0.06}),
        "signal.wavelet_ricker": (_signal_input, {"sigma": 0.2}),
        "signal.wavelet_chirplet": (_signal_input, {"sigma": 0.25, "omega": 16.0, "alpha": 3.0}),
        "signal.wavelet_laplace": (_signal_input, {"A": 1.0, "ep": 0.15, "tal": 0.1, "f": 5.0}),
        "signal.wavelet_morlet": (_signal_input, {"f_b": 2.0, "f_c": 6.0}),
        "feature.mean": (_signal_input, {}),
        "feature.std": (_signal_input, {}),
        "feature.rms": (_signal_input, {}),
        "feature.kurtosis": (_signal_input, {}),
        "feature.crest_factor": (_signal_input, {}),
        "feature.band_power": (_spectral_input, {"fs": 1024.0, "band_low_hz": 20.0, "band_high_hz": 150.0}),
        "feature.spectral_centroid": (_spectral_input, {"fs": 1024.0}),
        "multi.concatenate": (_feature_pair_input, {"axis": 0}),
        "multi.cross_correlation": (_multi_signal_input, {}),
        "decision.threshold": (_decision_input, {"threshold": 0.5}),
    }
    return mapping[op_uid]


def _to_torch(value: Any, torch):
    if isinstance(value, list):
        return [_to_torch(item, torch) for item in value]
    return torch.as_tensor(value, dtype=torch.float32)


TOLERANCE_BY_OP = {
    "signal.stft": (1e-4, 1e-4),
    "signal.filter": (1e-5, 1e-5),
    "signal.hilbert_envelope": (1e-4, 1e-4),
    "signal.psd": (2e-2, 2e-3),
    "signal.wavefilters": (1e-4, 1e-4),
    "signal.wavelet_ricker": (1e-4, 1e-4),
    "signal.wavelet_chirplet": (1e-4, 1e-4),
    "signal.wavelet_laplace": (1e-4, 1e-4),
    "signal.wavelet_morlet": (1e-4, 1e-4),
    "feature.kurtosis": (1e-5, 1e-5),
    "feature.band_power": (1e-4, 1e-4),
}


def test_torch_environment_uses_cu118_wheel():
    _require_torch()


@pytest.mark.parametrize("op_uid", sorted(get_operator_catalog().operators))
def test_operator_forward_pt_matches_forward_np_shape_and_values(op_uid: str):
    torch = _require_torch()
    catalog = get_operator_catalog()
    operator = catalog.get(op_uid)
    assert "pt" in operator.spec.backend_availability

    factory, params = _case_for(op_uid)
    np_input = factory()
    pt_input = _to_torch(factory(), torch)

    np_output = operator.forward_np(np_input, **params)
    pt_output = operator.forward_pt(pt_input, **params)

    if op_uid == "decision.threshold":
        assert isinstance(pt_output, dict)
        assert set(pt_output) == {"score", "decision", "threshold"}
        assert set(np_output) == {"score", "decision", "threshold"}
        assert pytest.approx(float(np_output["score"]), rel=1e-6, abs=1e-6) == float(pt_output["score"])
        assert bool(np_output["decision"]) is bool(pt_output["decision"])
        assert pytest.approx(float(np_output["threshold"]), rel=1e-8, abs=1e-8) == float(pt_output["threshold"])
        return

    assert torch.is_tensor(pt_output), f"{op_uid} forward_pt must return a torch.Tensor."
    np_arr = np.asarray(np_output)
    pt_arr = pt_output.detach().cpu().numpy()
    assert np_arr.shape == pt_arr.shape
    rtol, atol = TOLERANCE_BY_OP.get(op_uid, (1e-5, 1e-5))
    np.testing.assert_allclose(pt_arr, np_arr, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op_uid", ["signal.normalize", "signal.stft", "signal.psd", "feature.rms"])
def test_operator_forward_pt_rejects_non_finite_input(op_uid: str):
    torch = _require_torch()
    operator = get_operator_catalog().get(op_uid)
    bad = torch.tensor([[1.0, float("nan"), 2.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="non-finite"):
        operator.forward_pt(bad)


@pytest.mark.parametrize("op_uid", ["signal.normalize", "signal.stft", "signal.psd", "signal.hilbert_envelope"])
def test_signal_like_forward_pt_rejects_scalar_input(op_uid: str):
    torch = _require_torch()
    operator = get_operator_catalog().get(op_uid)
    with pytest.raises(ValueError, match="scalar input|rank"):
        operator.forward_pt(torch.tensor(1.0, dtype=torch.float32))


def test_bridge_based_pt_operators_use_unified_bridge_helper(monkeypatch):
    torch = _require_torch()
    counts = {"transform": 0, "aggregate": 0}
    original_transform_bridge = transform_ops.bridge_pt_via_numpy
    original_aggregate_bridge = aggregate_ops.bridge_pt_via_numpy

    def wrapped_transform_bridge(*args, **kwargs):
        counts["transform"] += 1
        return original_transform_bridge(*args, **kwargs)

    def wrapped_aggregate_bridge(*args, **kwargs):
        counts["aggregate"] += 1
        return original_aggregate_bridge(*args, **kwargs)

    monkeypatch.setattr(transform_ops, "bridge_pt_via_numpy", wrapped_transform_bridge)
    monkeypatch.setattr(aggregate_ops, "bridge_pt_via_numpy", wrapped_aggregate_bridge)

    catalog = get_operator_catalog()
    catalog.get("signal.filter").forward_pt(torch.as_tensor(_signal_input(), dtype=torch.float32), fs=1024.0)
    catalog.get("feature.kurtosis").forward_pt(torch.as_tensor(_signal_input(), dtype=torch.float32))

    assert counts == {"transform": 1, "aggregate": 1}


def test_native_pt_hotspots_do_not_use_numpy_bridge(monkeypatch):
    torch = _require_torch()

    def fail_bridge(*args, **kwargs):
        raise AssertionError("native PT hotspot unexpectedly used NumPy bridge")

    monkeypatch.setattr(transform_ops, "bridge_pt_via_numpy", fail_bridge)
    monkeypatch.setattr(expand_ops, "bridge_pt_via_numpy", fail_bridge, raising=False)

    signal = torch.as_tensor(_signal_input(), dtype=torch.float32)
    catalog = get_operator_catalog()
    stft = catalog.get("signal.stft").forward_pt(signal, nperseg=64, noverlap=32)
    hilbert = catalog.get("signal.hilbert_envelope").forward_pt(signal)
    psd = catalog.get("signal.psd").forward_pt(signal, fs=1024.0, nperseg=64)

    assert stft.shape == (2, 33, 7)
    assert hilbert.shape == signal.shape
    assert psd.shape == (2, 33)
