from __future__ import annotations

"""Low level signal-processing helpers used by the operators."""

from typing import Any

import numpy as np
from scipy.stats import kurtosis


def patch_signal(data: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """Split ``data`` into overlapping patches.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape ``(B, L, C)``.
    patch_size : int
        Length of each patch along the time dimension.
    stride : int
        Step size when sliding the patch window.

    Returns
    -------
    np.ndarray
        Array of shape ``(B, P, patch_size, C)`` where ``P`` is the number of
        generated patches.
    """
    b, l, c = data.shape
    patches = []
    for start in range(0, l - patch_size + 1, stride):
        patch = data[:, start : start + patch_size, :]
        patches.append(patch)
    result = np.stack(patches, axis=1)
    return result


def copy_signal(data: np.ndarray, num_copies: int) -> np.ndarray:
    """Repeat ``data`` ``num_copies`` times along a new dimension."""

    return np.repeat(data[:, None, :, :], num_copies, axis=1)


def fft_signal(data: np.ndarray) -> np.ndarray:
    """Compute the FFT along the time axis."""

    return np.fft.rfft(data, axis=-2)


def mean_signal(data: np.ndarray) -> np.ndarray:
    """Return the mean of ``data`` along the time axis."""

    return data.mean(axis=-2)


def kurtosis_signal(data: np.ndarray) -> np.ndarray:
    """Compute the kurtosis of ``data`` along the time axis."""

    return kurtosis(data, axis=-2)


def similarity(
    feat: np.ndarray,
    ref_feat: np.ndarray,
    method: str,
    threshold: float,
) -> bool:
    """Evaluate similarity between two feature vectors.

    Parameters
    ----------
    feat : np.ndarray
        Test feature vector.
    ref_feat : np.ndarray
        Reference feature vector.
    method : str
        Either ``"cosine"`` or ``"euclidean"``.
    threshold : float
        Decision threshold for similarity.

    Returns
    -------
    bool
        ``True`` if the similarity meets the threshold.
    """

    if method == "cosine":
        dot = np.vdot(feat, ref_feat)
        norm = np.linalg.norm(feat) * np.linalg.norm(ref_feat)
        score = float(abs(dot) / norm)
        return score >= threshold
    score = float(np.linalg.norm(feat - ref_feat))
    return score <= threshold
