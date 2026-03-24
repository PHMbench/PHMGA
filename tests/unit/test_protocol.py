from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import pytest

from src.data.protocol import (
    build_protocol_from_config,
    export_split_manifest,
    materialize_split_signals,
    summarize_split_ids_by_label_domain,
    summarize_split_records,
)


def test_build_protocol_from_config_filters_domain_ids_and_materializes_splits(tmp_path: Path):
    metadata_path = tmp_path / "metadata.xlsx"
    h5_path = tmp_path / "signals.h5"

    rows = []
    for label in (0, 1):
        for domain_id in (0, 1, 2, 3):
            sample_id = f"s_{label}_{domain_id}"
            rows.append(
                {
                    "Id": sample_id,
                    "Name": "RM101",
                    "Dataset_id": 101,
                    "Domain_id": domain_id,
                    "Label": label,
                    "Sample_rate": 20480,
                    "Sample_lenth": 16,
                    "Channel": 1,
                    "Domain_description": f"domain-{domain_id}",
                    "Description": f"label-{label}",
                }
            )

    pd.DataFrame(rows).to_excel(metadata_path, index=False)
    with h5py.File(h5_path, "w") as handle:
        for row in rows:
            handle.create_dataset(row["Id"], data=np.arange(16, dtype=float).reshape(16, 1))

    protocol = build_protocol_from_config(
        {
            "data": {
                "dataset_name": "RM101",
                "metadata_path": str(metadata_path),
                "h5_path": str(h5_path),
                "selection": {
                    "dataset_id": 101,
                    "domain_ids": [0, 1, 2],
                    "drop_invalid_labels": True,
                },
                "split": {
                    "strategy": "stratified_fixed_per_class",
                    "train_per_class": 1,
                    "val_per_class": 1,
                    "test_per_class": 1,
                    "seed": 0,
                },
                "window": {
                    "window_size": 8,
                    "stride": 8,
                    "slice_mode": "centered",
                    "drop_last_window": False,
                },
            }
        }
    )

    assert len(protocol.samples) == 6
    assert all(sample.domain_id in {0, 1, 2} for sample in protocol.samples)
    assert len(protocol.splits.train_ids) == 2
    assert len(protocol.splits.val_ids) == 2
    assert len(protocol.splits.test_ids) == 2

    materialized = materialize_split_signals(protocol)
    assert len(materialized["train"]) == 2
    assert len(materialized["val"]) == 2
    assert len(materialized["test"]) == 2
    assert materialized["train"][0].window.shape == (1, 8)


def test_ratio_split_fails_fast_when_any_class_would_have_empty_val_or_test(tmp_path: Path):
    metadata_path = tmp_path / "metadata_ratio.xlsx"
    h5_path = tmp_path / "signals_ratio.h5"

    rows = []
    for label in (0, 1):
        for index in range(2):
            sample_id = f"s_{label}_{index}"
            rows.append(
                {
                    "Id": sample_id,
                    "Name": "RM101",
                    "Dataset_id": 101,
                    "Domain_id": index,
                    "Label": label,
                    "Sample_rate": 20480,
                    "Sample_lenth": 16,
                    "Channel": 1,
                    "Domain_description": f"domain-{index}",
                    "Description": f"label-{label}",
                }
            )

    pd.DataFrame(rows).to_excel(metadata_path, index=False)
    with h5py.File(h5_path, "w") as handle:
        for row in rows:
            handle.create_dataset(row["Id"], data=np.arange(16, dtype=float).reshape(16, 1))

    with pytest.raises(ValueError, match="every class must appear in train/val/test"):
        build_protocol_from_config(
            {
                "data": {
                    "dataset_name": "RM101",
                    "metadata_path": str(metadata_path),
                    "h5_path": str(h5_path),
                    "selection": {
                        "dataset_id": 101,
                        "domain_ids": [0, 1],
                        "drop_invalid_labels": True,
                    },
                    "split": {
                        "strategy": "stratified_ratio",
                        "train_ratio": 0.6,
                        "val_ratio": 0.2,
                        "test_ratio": 0.2,
                        "seed": 0,
                    },
                    "window": {
                        "window_size": 8,
                        "stride": 8,
                        "slice_mode": "sliding",
                        "drop_last_window": False,
                    },
                }
            }
        )


def test_sliding_windows_generate_multiple_windows_and_summary(tmp_path: Path):
    metadata_path = tmp_path / "metadata_sliding.xlsx"
    h5_path = tmp_path / "signals_sliding.h5"

    rows = []
    for label in (0, 1):
        for index in range(3):
            sample_id = f"s_{label}_{index}"
            rows.append(
                {
                    "Id": sample_id,
                    "Name": "RM101",
                    "Dataset_id": 101,
                    "Domain_id": index,
                    "Label": label,
                    "Sample_rate": 20480,
                    "Sample_lenth": 16384,
                    "Channel": 1,
                    "Domain_description": f"domain-{index}",
                    "Description": f"label-{label}",
                }
            )

    pd.DataFrame(rows).to_excel(metadata_path, index=False)
    with h5py.File(h5_path, "w") as handle:
        for row in rows:
            handle.create_dataset(row["Id"], data=np.arange(16384, dtype=float).reshape(16384, 1))

    protocol = build_protocol_from_config(
        {
            "data": {
                "dataset_name": "RM101",
                "metadata_path": str(metadata_path),
                "h5_path": str(h5_path),
                "selection": {
                    "dataset_id": 101,
                    "domain_ids": [0, 1, 2],
                    "drop_invalid_labels": True,
                },
                "split": {
                    "strategy": "stratified_fixed_per_class",
                    "train_per_class": 1,
                    "val_per_class": 1,
                    "test_per_class": 1,
                    "seed": 0,
                },
                "window": {
                    "window_size": 4096,
                    "stride": 4096,
                    "slice_mode": "sliding",
                    "drop_last_window": False,
                },
            }
        }
    )

    materialized = materialize_split_signals(protocol)
    summary = summarize_split_records(materialized)

    assert len(materialized["train"]) == 8
    assert len(materialized["val"]) == 8
    assert len(materialized["test"]) == 8
    assert summary["n_train_windows"] == 8
    assert summary["n_val_windows"] == 8
    assert summary["n_test_windows"] == 8
    assert summary["train_windows_by_class"] == {"0": 4, "1": 4}


def test_split_manifest_and_label_domain_summary(tmp_path: Path):
    metadata_path = tmp_path / "metadata_domains.xlsx"
    h5_path = tmp_path / "signals_domains.h5"

    rows = []
    for label in (0, 1):
        for domain_id in (0, 1, 2):
            for replica in range(2):
                sample_id = f"s_{label}_{domain_id}_{replica}"
                rows.append(
                    {
                        "Id": sample_id,
                        "Name": "RM101",
                        "Dataset_id": 101,
                        "Domain_id": domain_id,
                        "Label": label,
                        "Sample_rate": 20480,
                        "Sample_lenth": 16,
                        "Channel": 1,
                        "Domain_description": f"domain-{domain_id}",
                        "Description": f"label-{label}",
                    }
                )

    pd.DataFrame(rows).to_excel(metadata_path, index=False)
    with h5py.File(h5_path, "w") as handle:
        for row in rows:
            handle.create_dataset(row["Id"], data=np.arange(16, dtype=float).reshape(16, 1))

    protocol = build_protocol_from_config(
        {
            "data": {
                "dataset_name": "RM101",
                "metadata_path": str(metadata_path),
                "h5_path": str(h5_path),
                "selection": {
                    "dataset_id": 101,
                    "domain_ids": [0, 1, 2],
                    "drop_invalid_labels": True,
                },
                "split": {
                    "strategy": "stratified_ratio",
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2,
                    "seed": 0,
                },
                "window": {
                    "window_size": 8,
                    "stride": 8,
                    "slice_mode": "sliding",
                    "drop_last_window": False,
                },
            }
        }
    )

    manifest = export_split_manifest(protocol)
    rows = summarize_split_ids_by_label_domain(protocol)

    assert len(manifest["train_ids"]) == 8
    assert len(manifest["val_ids"]) == 2
    assert len(manifest["test_ids"]) == 2
    assert rows
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert {row["label"] for row in rows} == {0, 1}
    assert {row["domain_id"] for row in rows} <= {0, 1, 2}
