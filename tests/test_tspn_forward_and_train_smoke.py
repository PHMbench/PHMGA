import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch training smoke tests.")
def test_tspn_forward_and_train_smoke():
    torch = pytest.importorskip("torch")

    from src.model.explainable.config_schema import TSPNConfig
    from src.model.explainable.builder import build_tspn_from_config

    L = 128
    in_channels = 1
    num_classes = 2

    cfg = TSPNConfig.model_validate(
        {
            "model": {
                "name": "tspn",
                "device": "cpu",
                "num_classes": num_classes,
                "in_dim": L,
                "in_channels": in_channels,
                "out_channels": 2,
                "scale": 2,
                "skip_connection": True,
                "wf_init": {"f_c_mu": 0.0, "f_c_sigma": 0.1, "f_b_mu": 0.0, "f_b_sigma": 0.1},
                "layers": [
                    {
                        "gate_temperature": 1.0,
                        "ops": [
                            {"token": "I", "params": {}},
                            {"token": "WF", "params": {}},
                        ],
                    }
                ],
                "features": ["Mean", "Std"],
                "disabled_ops": {},
            },
            "train": {
                "seed": 123,
                "epochs": 1,
                "batch_size": 4,
                "lr": 1e-2,
                "weight_decay": 0.0,
                "val_ratio": 0.2,
                "patience": 1,
                "debug": True,
                "debug_max_samples": 8,
                "debug_epochs": 1,
                "l1_gate": 0.0,
                "entropy_gate": 0.0,
            },
            "explain": {"topk_ops": 3, "save_wavefilters": True},
            "meta": {"test": True},
        }
    )

    model, _manifest = build_tspn_from_config(cfg, device="cpu")

    torch.manual_seed(0)
    x = torch.randn(4, L, in_channels, dtype=torch.float32)
    logits = model(x)
    assert tuple(logits.shape) == (4, num_classes)
    assert not logits.is_complex()

    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    w0 = model.classifier.clf[0].weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
    w1 = model.classifier.clf[0].weight.detach().clone()

    assert not torch.equal(w0, w1)
