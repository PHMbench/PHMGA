from __future__ import annotations

from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_preview_pair, materialize_split_signals


def test_protocol_reads_fixture_metadata_and_h5(make_dataset_fixture):
    fixture = make_dataset_fixture(dataset_name="RM_101_THU_GEARBOX", dataset_id=101, channels=3)
    config = load_runtime_config("rm101_ml_openrouter")
    config["data"]["metadata_path"] = fixture["metadata_path"]
    config["data"]["h5_path"] = fixture["h5_path"]

    protocol = build_protocol_from_config(config)

    assert protocol.dataset_name == "RM_101_THU_GEARBOX"
    assert len(protocol.samples) == 8
    assert protocol.samples[0].observed_channels == 3
    assert len(protocol.splits.train_ids) > 0
    assert len(protocol.splits.val_ids) > 0
    assert len(protocol.splits.test_ids) > 0


def test_protocol_materialization_respects_selected_channels_and_windowing(make_dataset_fixture):
    fixture = make_dataset_fixture(dataset_name="RM_101_THU_GEARBOX", dataset_id=101, channels=3, length=128)
    config = load_runtime_config("rm101_ml_openrouter")
    config["data"]["metadata_path"] = fixture["metadata_path"]
    config["data"]["h5_path"] = fixture["h5_path"]
    config["data"]["selected_channels"] = [0, 2]
    config["data"]["split"] = {
        "strategy": "stratified_fixed_per_class",
        "seed": 0,
        "train_per_class": 2,
        "val_per_class": 1,
        "test_per_class": 1,
    }
    config["data"]["window"] = {
        "window_size": 64,
        "stride": 32,
        "slice_mode": "sliding",
        "drop_last_window": False,
    }

    protocol = build_protocol_from_config(config)
    records = materialize_split_signals(protocol)
    preview_ref, preview_tst = materialize_preview_pair(protocol)

    assert protocol.selected_channels == [0, 2]
    assert preview_ref.window.shape == (2, 64)
    assert preview_tst.window.shape == (2, 64)
    assert records["train"][0].window.shape == (2, 64)
    assert records["train"][0].window_id.endswith("__w0000")
    assert all(record.split == "train" for record in records["train"])
