from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import pytest


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
        }

    return factory
