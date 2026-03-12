from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SampleMeta(BaseModel):
    sample_id: str
    dataset: str
    label: int
    sampling_rate: int
    length: int
    channels: int
    operating_condition: str
    source_h5: str


class SplitManifest(BaseModel):
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


class WindowSpec(BaseModel):
    window_size: int = Field(gt=0)
    stride: int = Field(gt=0)
    slice_mode: Literal["sliding", "centered"] = "sliding"
    drop_last_window: bool = False


class DatasetProtocol(BaseModel):
    dataset_name: str
    catalog: str
    metadata_schema_version: str
    samples: List[SampleMeta]
    splits: SplitManifest
    window: WindowSpec
    leakage_boundary: str = "split_before_windowing"

    @model_validator(mode="after")
    def ensure_split_ids_exist(self) -> "DatasetProtocol":
        known = {sample.sample_id for sample in self.samples}
        for split_name in ("train_ids", "val_ids", "test_ids"):
            missing = [sample_id for sample_id in getattr(self.splits, split_name) if sample_id not in known]
            if missing:
                raise ValueError(f"Unknown ids in {split_name}: {missing}")
        return self


@dataclass
class SignalRecord:
    sample_id: str
    label: int
    split: str
    windows: list[np.ndarray]


def build_protocol_from_config(config: Dict[str, Any]) -> DatasetProtocol:
    data_cfg = dict(config.get("data", {}))
    labels = dict(data_cfg.get("labels", {}))
    split_cfg = dict(data_cfg.get("splits", {}))
    samples: list[SampleMeta] = []
    for split_name, ids in split_cfg.items():
        for sample_id in ids:
            samples.append(
                SampleMeta(
                    sample_id=sample_id,
                    dataset=str(data_cfg["dataset_name"]),
                    label=int(labels[sample_id]),
                    sampling_rate=int(data_cfg["sampling_rate"]),
                    length=int(data_cfg["length"]),
                    channels=int(data_cfg["channels"]),
                    operating_condition=str(data_cfg["operating_condition"]),
                    source_h5=f"{sample_id}.h5",
                )
            )
    return DatasetProtocol(
        dataset_name=str(data_cfg["dataset_name"]),
        catalog=str(data_cfg["catalog"]),
        metadata_schema_version=str(data_cfg["metadata_schema_version"]),
        samples=samples,
        splits=SplitManifest(**split_cfg),
        window=WindowSpec(**dict(data_cfg["window"])),
    )


def _seed(sample_id: str, dataset: str) -> int:
    seed_value = 0
    for token in (sample_id, dataset):
        for char in token:
            seed_value = (seed_value * 131 + ord(char)) % (2**32 - 1)
    return seed_value


def _generate_signal(sample: SampleMeta) -> np.ndarray:
    rng = np.random.default_rng(_seed(sample.sample_id, sample.dataset))
    time_index = np.linspace(0.0, 1.0, sample.length, endpoint=False)
    label_scale = 1.0 + 0.35 * sample.label
    base_freq = 5.0 if sample.dataset == "RM101" else 7.0
    channels: list[np.ndarray] = []
    for channel_idx in range(sample.channels):
        phase = channel_idx * 0.35
        waveform = label_scale * np.sin(2 * np.pi * (base_freq + channel_idx) * time_index + phase)
        modulation = 0.4 * np.cos(2 * np.pi * (base_freq * 0.5) * time_index)
        noise = 0.05 * rng.standard_normal(sample.length)
        channels.append(waveform + modulation + noise)
    return np.stack(channels, axis=0)


def _window_signal(signal: np.ndarray, window: WindowSpec) -> list[np.ndarray]:
    total_length = signal.shape[-1]
    if window.slice_mode == "centered":
        start = max(0, (total_length - window.window_size) // 2)
        return [signal[:, start : start + window.window_size]]

    windows: list[np.ndarray] = []
    start = 0
    while start < total_length:
        end = start + window.window_size
        if end > total_length:
            if window.drop_last_window:
                break
            start = max(0, total_length - window.window_size)
            end = total_length
        windows.append(signal[:, start:end])
        if end >= total_length:
            break
        start += window.stride
    return windows


def materialize_split_signals(protocol: DatasetProtocol) -> Dict[str, List[SignalRecord]]:
    split_lookup: Dict[str, set[str]] = {
        "train": set(protocol.splits.train_ids),
        "val": set(protocol.splits.val_ids),
        "test": set(protocol.splits.test_ids),
    }
    outputs: Dict[str, List[SignalRecord]] = {"train": [], "val": [], "test": []}
    for sample in protocol.samples:
        split_name = next(name for name, ids in split_lookup.items() if sample.sample_id in ids)
        signal = _generate_signal(sample)
        windows = _window_signal(signal, protocol.window)
        outputs[split_name].append(
            SignalRecord(
                sample_id=sample.sample_id,
                label=sample.label,
                split=split_name,
                windows=windows,
            )
        )
    return outputs
