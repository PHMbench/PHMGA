import numpy as np
from src.tools.functions import (
    patch_signal,
    copy_signal,
    fft_signal,
    mean_signal,
    kurtosis_signal,
    similarity,
)


def test_patch_signal():
    x = np.arange(8).reshape(1, 8, 1)
    result = patch_signal(x, 4, 2)
    assert result.shape == (1, 3, 4, 1)


def test_copy_signal():
    x = np.ones((1, 4, 1))
    result = copy_signal(x, 3)
    assert result.shape == (1, 3, 4, 1)
    assert np.all(result[0, 0] == 1)


def test_mean_kurtosis():
    x = np.array([[[1.0], [2.0], [3.0], [4.0]]])
    mean = mean_signal(x)
    kurt = kurtosis_signal(x)
    assert np.isclose(mean[0, 0], 2.5)
    assert kurt.shape == (1, 1)


def test_fft_similarity():
    x = np.linspace(0, 1, 8).reshape(1, 8, 1)
    fft = fft_signal(x)
    assert fft.shape[0] == 1
    ref = fft.copy()
    assert similarity(fft, ref, "cosine", 0.9)
