import os

import pytest


@pytest.mark.skipif(
    os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() not in {"1", "true", "yes", "y"},
    reason="Torch contract tests are disabled by default (set PHM_ENABLE_TORCH_TESTS=1 to enable).",
)
def test_fft_wf_ht_i_contract_shape_dtype():
    # IMPORTANT:
    # This test is opt-in because importing a broken torch build can hard-crash the interpreter.
    torch = pytest.importorskip("torch")

    from src.model.explainable.ops import (
        Detrend,
        Differentiate,
        FFTMagnitude,
        HilbertEnvelope,
        Identity,
        Integrate,
        LogOperation,
        Normalize,
        STFTMagnitude,
        SinOperation,
        SquOperation,
        WaveFilters,
    )

    B, L, C = 2, 4096, 3
    x = torch.randn(B, L, C, dtype=torch.float32)

    for module in [
        Identity(channels=C),
        HilbertEnvelope(channels=C),
        WaveFilters(channels=C),
        FFTMagnitude(channels=C, align_strategy="interp"),
        FFTMagnitude(channels=C, align_strategy="mirror"),
        Normalize(channels=C, method="z_score"),
        Detrend(channels=C, type="linear"),
        Integrate(channels=C),
        Differentiate(channels=C),
        STFTMagnitude(channels=C, n_fft=64, hop_length=32),
        LogOperation(channels=C),
        SquOperation(channels=C),
        SinOperation(channels=C, frequency=1.0),
    ]:
        y = module(x)
        assert tuple(y.shape) == (B, L, C)
        assert y.dtype.is_floating_point
        assert not y.is_complex()
