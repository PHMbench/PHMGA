from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_split_signals


ROOT = Path(__file__).resolve().parents[2]
REAL_RM101_METADATA = Path("/home/user/data/PHMbenchdata/PHM-Vibench/gear_metadata.xlsx")
REAL_RM101_H5 = Path("/home/user/data/PHMbenchdata/PHM-Vibench/RM_101_THU_GEARBOX.h5")
REAL_OTTAWA_METADATA = Path("/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx")
REAL_OTTAWA_H5 = Path("/home/user/data/PHMbenchdata/PHM-Vibench/RM_017_Ottawa19.h5")


def test_real_protocol_uses_canonical_splits_for_both_datasets():
    for config_name, expected_dataset in (
        ("config/runs/rm101_dag.yaml", "RM_101_THU_GEARBOX"),
        ("config/runs/ottawa_dag.yaml", "RM_017_Ottawa19"),
    ):
        config = load_runtime_config(ROOT / config_name)
        protocol = build_protocol_from_config(config)
        assert protocol.catalog == "PHM-Vibench"
        assert protocol.dataset_name == expected_dataset
        assert len(protocol.splits.train_ids) > 0
        assert len(protocol.splits.val_ids) > 0
        assert len(protocol.splits.test_ids) > 0


def test_materialized_windows_respect_split_before_windowing_for_synthetic():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    records = materialize_split_signals(protocol)
    assert protocol.source_mode == "synthetic"
    assert set(records) == {"train", "val", "test"}
    assert all(record.split == "train" for record in records["train"])


def test_real_protocol_reads_metadata_columns_and_h5_shape():
    if not all(path.exists() for path in (REAL_RM101_METADATA, REAL_RM101_H5, REAL_OTTAWA_METADATA, REAL_OTTAWA_H5)):
        pytest.skip("Real PHM-Vibench files are not available in this environment.")

    rm101 = build_protocol_from_config(load_runtime_config(ROOT / "config/runs/rm101_dag.yaml"))
    ottawa = build_protocol_from_config(load_runtime_config(ROOT / "config/runs/ottawa_dag.yaml"))

    assert rm101.samples[0].metadata_length == 768000
    assert rm101.samples[0].observed_length == 767999
    assert rm101.samples[0].observed_channels == 8
    assert ottawa.samples[0].observed_length == 2000000
    assert ottawa.samples[0].observed_channels == 2
