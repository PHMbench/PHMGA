import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(
    not _enabled(),
    reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch builder autofix tests.",
)
def test_tspn_builder_autofixes_channel_divisibility():
    torch = pytest.importorskip("torch")

    from src.model.explainable.builder import build_tspn_from_config
    from src.model.explainable.config_schema import TSPNConfig

    cfg = TSPNConfig.model_validate(
        {
            "model": {
                "name": "tspn",
                "device": "cpu",
                "num_classes": 3,
                "in_dim": 128,
                "in_channels": 1,
                "out_channels": 4,
                "scale": 4,
                "skip_connection": True,
                "wf_init": {"f_c_mu": 0.0, "f_c_sigma": 0.1, "f_b_mu": 0.0, "f_b_sigma": 0.1},
                "layers": [
                    {
                        "gate_temperature": 1.0,
                        "ops": [
                            {"token": "I", "params": {}},
                            {"token": "WF", "params": {}},
                            {"token": "HT", "params": {}},
                            {"token": "FFT", "params": {}},
                            {"token": "NORM", "params": {}},
                            {"token": "SIN", "params": {}},
                        ],
                    }
                ],
                "features": ["Mean", "Std"],
                "disabled_ops": {},
            },
            "train": {
                "seed": 42,
                "epochs": 1,
                "batch_size": 4,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "val_ratio": 0.2,
                "patience": 1,
                "debug": True,
                "debug_max_samples": 8,
                "debug_epochs": 1,
            },
            "explain": {"topk_ops": 3, "save_wavefilters": True},
            "meta": {"test": True},
        }
    )

    model, manifest = build_tspn_from_config(cfg, device="cpu")
    adjustment = manifest.get("channel_adjustment")
    assert adjustment is not None
    assert adjustment["out_total_before"] == 16
    assert adjustment["out_total_after"] == 24
    assert adjustment["out_channels_before"] == 4
    assert adjustment["out_channels_after"] == 6

    x = torch.randn(2, 128, 1, dtype=torch.float32)
    logits = model(x)
    assert tuple(logits.shape) == (2, 3)

