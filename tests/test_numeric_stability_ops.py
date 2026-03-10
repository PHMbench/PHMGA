import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.aggregate_schemas import HjorthParametersOp
from src.tools.transform_schemas import CepstrumOp, DenoiseWaveletOp


def test_hjorth_constant_signal_no_nan_inf():
    x = np.ones((2, 128, 3), dtype=np.float32)
    y = HjorthParametersOp().execute(x)
    assert y.shape == (2, 3, 3)
    assert np.isfinite(y).all()


def test_cepstrum_zero_signal_no_nan_inf():
    x = np.zeros((1, 64, 2), dtype=np.float32)
    y = CepstrumOp().execute(x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_denoise_wavelet_keeps_length():
    pytest.importorskip("pywt")
    x = np.random.randn(2, 257, 1).astype(np.float32)
    y = DenoiseWaveletOp(wavelet="db1", mode="soft").execute(x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
