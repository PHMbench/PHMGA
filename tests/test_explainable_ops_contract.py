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

    from src.model.explainable.ops import FFTMagnitude, HilbertEnvelope, Identity, WaveFilters

    B, L, C = 2, 4096, 3
    x = torch.randn(B, L, C, dtype=torch.float32)

    for module in [
        Identity(channels=C),
        HilbertEnvelope(channels=C),
        WaveFilters(channels=C),
        FFTMagnitude(channels=C, align_strategy="interp"),
        FFTMagnitude(channels=C, align_strategy="mirror"),
    ]:
        y = module(x)
        assert tuple(y.shape) == (B, L, C)
        assert y.dtype.is_floating_point
        assert not y.is_complex()

