from typing import List, Dict, Any, Optional, Sequence
import numpy as np

def assert_shape(array: np.ndarray, expected: Sequence[Optional[int]]) -> None:
    """Assert that a NumPy array matches an expected shape.

    ``None`` in the expected shape acts as a wildcard for that dimension.

    Args:
        array: Array whose shape will be validated.
        expected: Expected shape tuple where each element can be an ``int`` or
            ``None``.

    Raises:
        AssertionError: If the number of dimensions or any specified dimension
            does not match ``expected``.

    Examples:
        >>> import numpy as np
        >>> assert_shape(np.zeros((2, 3)), (2, 3))
        >>> assert_shape(np.zeros((2, 3)), (2, None))
        >>> assert_shape(np.zeros((2, 3)), (1, 3))
        Traceback (most recent call last):
        ...
        AssertionError: Expected shape (1, 3), got (2, 3)
    """

    actual = array.shape
    if len(actual) != len(expected):
        raise AssertionError(
            f"Expected {len(expected)} dimensions, got {len(actual)}"
        )
    for idx, (act, exp) in enumerate(zip(actual, expected)):
        if exp is not None and act != exp:
            raise AssertionError(f"Expected shape {tuple(expected)}, got {actual}")


if __name__ == "__main__":
    print("--- Testing utils.py ---")

    arr = np.zeros((2, 3))
    assert_shape(arr, (2, 3))

    try:
        assert_shape(arr, (3, 2))
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_shape should have raised an AssertionError")

    print("\n--- utils.py tests passed! ---")
