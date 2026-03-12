from __future__ import annotations

from pathlib import Path

from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_split_signals


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_uses_canonical_splits_for_both_datasets():
    for dataset_name in ("RM101", "Ottawa"):
        config = load_runtime_config(ROOT / "config/config.yaml", dataset_name=dataset_name)
        protocol = build_protocol_from_config(config)
        assert protocol.catalog == "PHM-Vibench"
        assert len(protocol.splits.train_ids) > 0
        assert len(protocol.splits.val_ids) > 0
        assert len(protocol.splits.test_ids) > 0
        assert protocol.window.window_size == 256


def test_materialized_windows_respect_split_before_windowing():
    config = load_runtime_config(ROOT / "config/config.yaml", dataset_name="RM101")
    protocol = build_protocol_from_config(config)
    records = materialize_split_signals(protocol)
    assert set(records) == {"train", "val", "test"}
    assert all(record.split == "train" for record in records["train"])
