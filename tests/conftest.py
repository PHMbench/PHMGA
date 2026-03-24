from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_wave(label: int, length: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_index = np.linspace(0.0, 1.0, length, endpoint=False)
    frequency = 4.0 if label == 0 else 11.0
    amplitude = 1.0 if label == 0 else 1.8
    outputs = []
    for channel in range(channels):
        phase = channel * 0.2
        signal = amplitude * np.sin(2 * np.pi * (frequency + channel) * time_index + phase)
        signal += 0.05 * rng.standard_normal(length)
        outputs.append(signal)
    return np.stack(outputs, axis=1)


@pytest.fixture
def make_dataset_fixture(tmp_path: Path) -> Callable[..., dict[str, object]]:
    def factory(
        *,
        dataset_name: str,
        dataset_id: int,
        channels: int = 3,
        length: int = 128,
        sample_rate: int = 25600,
        samples_per_class: int = 4,
    ) -> dict[str, object]:
        metadata_path = tmp_path / f"{dataset_name}_metadata.xlsx"
        h5_path = tmp_path / f"{dataset_name}.h5"
        rows = []
        sample_id = 1000
        with h5py.File(h5_path, "w") as handle:
            for label in (0, 1):
                for offset in range(samples_per_class):
                    signal = _build_wave(label, length, channels, seed=sample_id + offset)
                    handle.create_dataset(str(sample_id), data=signal)
                    rows.append(
                        {
                            "Id": sample_id,
                            "Name": dataset_name,
                            "Dataset_id": dataset_id,
                            "Label": label,
                            "Sample_rate": sample_rate,
                            "Sample_lenth": length,
                            "Channel": channels,
                            "Domain_id": 1,
                            "Domain_description": "fixture",
                        }
                    )
                    sample_id += 1
        pd.DataFrame(rows).to_excel(metadata_path, index=False)
        return {
            "metadata_path": str(metadata_path),
            "h5_path": str(h5_path),
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "channels": channels,
            "length": length,
            "sample_ids": list(range(1000, sample_id)),
            "sample_ids_by_label": {
                0: list(range(1000, 1000 + samples_per_class)),
                1: list(range(1000 + samples_per_class, 1000 + samples_per_class * 2)),
            },
        }

    return factory


@pytest.fixture
def make_case_config(tmp_path: Path, make_dataset_fixture) -> Callable[..., dict[str, object]]:
    def factory(
        *,
        case_name: str,
        graph: str = "with_report",
        dataset_name: str = "RM_101_THU_GEARBOX",
        dataset_id: int = 101,
        channels: int = 2,
        length: int = 128,
        samples_per_class: int = 4,
    ) -> dict[str, object]:
        dataset = make_dataset_fixture(
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            channels=channels,
            length=length,
            samples_per_class=samples_per_class,
        )
        config_root = tmp_path / "config"
        config_root.mkdir(parents=True, exist_ok=True)
        artifact_root = tmp_path / "artifacts" / case_name
        state_save_path = artifact_root / f"{case_name}.pkl"
        report_path = artifact_root / f"{case_name}.md"
        ref_ids = [
            dataset["sample_ids_by_label"][0][0],
            dataset["sample_ids_by_label"][0][1],
            dataset["sample_ids_by_label"][1][0],
            dataset["sample_ids_by_label"][1][1],
        ]
        test_ids = [
            dataset["sample_ids_by_label"][0][2],
            dataset["sample_ids_by_label"][0][3],
            dataset["sample_ids_by_label"][1][2],
            dataset["sample_ids_by_label"][1][3],
        ]
        config = {
            "name": case_name,
            "save_dir": str(artifact_root),
            "metadata_path": dataset["metadata_path"],
            "h5_path": dataset["h5_path"],
            "state_save_path": str(state_save_path),
            "report_path": str(report_path),
            "ref_ids": ref_ids,
            "test_ids": test_ids,
            "builder": {
                "graph": graph,
                "min_depth": 3,
                "min_width": 2,
                "max_depth": 6,
            },
            "user_instruction": "Analyze the bearing signals and generate a final diagnosis report.",
        }
        config_path = config_root / f"{case_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return {
            "case_name": case_name,
            "config_root": str(config_root),
            "config_path": str(config_path),
            "state_save_path": str(state_save_path),
            "report_path": str(report_path),
            "dataset": dataset,
        }

    return factory
