import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.multi_schemas import DistanceOp
from src.tools.transform_schemas import FilterOp, ResampleOp, SavitzkyGolayFilterOp


def test_filter_invalid_cutoff_raises():
    x = np.random.randn(1, 128, 1).astype(np.float32)
    op = FilterOp(filter_type="low", fs=1000.0, cutoff=800.0, order=4)
    with pytest.raises(ValueError):
        op.execute(x)


def test_resample_invalid_num_raises():
    x = np.random.randn(1, 128, 1).astype(np.float32)
    op = ResampleOp(num=0)
    with pytest.raises(ValueError):
        op.execute(x)


def test_savgol_even_window_raises():
    x = np.random.randn(1, 128, 1).astype(np.float32)
    op = SavitzkyGolayFilterOp(window_length=8, polyorder=3)
    with pytest.raises(ValueError):
        op.execute(x)


def test_patch_too_large_raises():
    pytest.importorskip("skimage")
    from src.tools.expand_schemas import PatchOp

    x = np.random.randn(2, 32, 1).astype(np.float32)
    op = PatchOp(patch_size=64, stride=8)
    with pytest.raises(ValueError):
        op.execute(x)


def test_distance_cosine_zero_vector_is_finite():
    v1 = np.zeros((4, 8), dtype=np.float32)
    v2 = np.zeros((4, 8), dtype=np.float32)
    op = DistanceOp(metric="cosine")
    out = op.execute({"a": v1, "b": v2})
    assert out.shape == (4,)
    assert np.isfinite(out).all()
