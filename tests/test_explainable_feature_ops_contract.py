import os

import pytest


@pytest.mark.skipif(
    os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() not in {"1", "true", "yes", "y"},
    reason="Torch contract tests are disabled by default (set PHM_ENABLE_TORCH_TESTS=1 to enable).",
)
def test_feature_ops_contract_shape_dtype():
    torch = pytest.importorskip("torch")

    from src.model.explainable.feature_ops import make_feature

    B, C, L = 2, 4, 4096
    x = torch.randn(B, C, L, dtype=torch.float32)

    tokens = [
        "Mean",
        "Std",
        "Var",
        "Entropy",
        "Max",
        "Min",
        "AbsMean",
        "Kurtosis",
        "RMS",
        "CrestFactor",
        "Skewness",
        "ClearanceFactor",
        "ShapeFactor",
        "SpectralKurtosis",
    ]

    for t in tokens:
        m = make_feature(t)
        y = m(x)
        assert tuple(y.shape) == (B, C, 1)
        assert y.dtype.is_floating_point
        assert not y.is_complex()

