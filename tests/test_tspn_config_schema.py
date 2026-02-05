import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.explainable import load_tspn_config


def test_model_tspn_basic_yaml_valid():
    path = Path("config") / "model_tspn_basic.yaml"
    cfg = load_tspn_config(path)
    assert cfg.model.name == "tspn"
    assert cfg.model.layers


def test_legacy_config_adapter(tmp_path):
    legacy = {
        "signal_processing_configs": {"layer1": ["I", "WF"], "layer2": ["I"]},
        "feature_extractor_configs": ["Mean", "RMS"],
        "args": {
            "device": "cpu",
            "num_classes": 3,
            "in_dim": 8,
            "in_channels": 2,
            "out_channels": 3,
            "scale": 4,
            "skip_connection": True,
            "learning_rate": 0.001,
            "batch_size": 4,
            "num_epochs": 2,
            "weight_decay": 0.0,
            "seed": 7,
            "patience": 2,
        },
    }
    p = tmp_path / "legacy.yaml"
    p.write_text(
        "signal_processing_configs:\n"
        "  layer1: [I, WF]\n"
        "  layer2: [I]\n"
        "feature_extractor_configs: [Mean, RMS]\n"
        "args:\n"
        "  device: cpu\n"
        "  num_classes: 3\n"
        "  in_dim: 8\n"
        "  in_channels: 2\n"
        "  out_channels: 3\n"
        "  scale: 4\n"
        "  skip_connection: true\n"
        "  learning_rate: 0.001\n"
        "  batch_size: 4\n"
        "  num_epochs: 2\n"
        "  weight_decay: 0.0\n"
        "  seed: 7\n"
        "  patience: 2\n"
    )
    cfg = load_tspn_config(p)
    assert cfg.model.in_dim == 8
    assert len(cfg.model.layers) == 2
    assert cfg.model.features == ["Mean", "RMS"]
