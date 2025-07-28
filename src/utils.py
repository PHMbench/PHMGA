"""Utility helpers for the PHM system."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["assert_shape"]


def assert_shape(array: np.ndarray, expected: Sequence[Optional[int]]) -> None:
    """Assert that ``array`` has the given shape.

    ``None`` may be used in ``expected`` as a wildcard for any size.

    Args:
        array: Array whose shape should be validated.
        expected: Expected shape tuple where ``None`` means any value.

    Raises:
        AssertionError: If the shapes are incompatible.

    Examples:
        >>> import numpy as np
        >>> arr = np.zeros((2, 3, 4))
        >>> assert_shape(arr, (2, None, 4))
        >>> assert_shape(arr, (2, 3, 4))
        >>> assert_shape(arr, (1, 3, 4))
        Traceback (most recent call last):
        ...
        AssertionError: Expected shape (1, 3, 4), got (2, 3, 4)
    """

    if len(array.shape) != len(expected):
        raise AssertionError(f"Expected {len(expected)} dimensions, got {array.shape}")

    for idx, (dim, exp) in enumerate(zip(array.shape, expected)):
        if exp is not None and dim != exp:
            raise AssertionError(f"Expected shape {expected}, got {tuple(array.shape)}")
