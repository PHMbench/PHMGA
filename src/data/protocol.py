"""Canonical real-data protocol, split resolution, and window materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import h5py
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def _stringify_id(value: Any) -> str:
    if _is_missing(value):
        raise ValueError("Sample id cannot be missing.")
    if isinstance(value, str):
        return value.strip()
    return str(int(value))


def _clean_label(value: Any) -> int:
    if _is_missing(value):
        raise ValueError("Label cannot be missing.")
    return int(float(value))


def _clean_text(value: Any, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def _clean_optional_int(value: Any) -> Optional[int]:
    if _is_missing(value):
        return None
    return int(value)


def _normalize_h5_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 3 and int(shape[2]) == 1:
        return int(shape[0]), int(shape[1])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported H5 tensor shape: {shape!r}")


def _normalize_h5_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported H5 array shape: {array.shape!r}")
    return np.asarray(array, dtype=float).T


class SampleMeta(BaseModel):
    sample_id: str
    dataset: str
    label: int
    sampling_rate: int
    length: int
    channels: int
    operating_condition: str
    source_h5: str
    dataset_id: Optional[int] = None
    domain_id: Optional[int] = None
    domain_description: str = ""
    metadata_length: Optional[int] = None
    metadata_channels: Optional[int] = None
    observed_length: Optional[int] = None
    observed_channels: Optional[int] = None


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
    metadata_path: str
    h5_path: str
    selection: Dict[str, Any]
    split: Dict[str, Any]
    window: WindowSpec
    samples: List[SampleMeta]
    splits: SplitManifest
    selected_channels: Optional[List[int]] = None

    @model_validator(mode="after")
    def ensure_split_ids_exist(self) -> "DatasetProtocol":
        known = {sample.sample_id for sample in self.samples}
        for split_name in ("train_ids", "val_ids", "test_ids"):
            ids = getattr(self.splits, split_name)
            missing = [sample_id for sample_id in ids if sample_id not in known]
            if missing:
                raise ValueError(f"Unknown ids in {split_name}: {missing}")
        return self


class SignalRecord(BaseModel):
    source_sample_id: str
    window_id: str
    window_index: int
    split: Literal["train", "val", "test"]
    label: int
    window: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _resolve_selected_channels(data_cfg: Dict[str, Any], observed_channels: int) -> Optional[List[int]]:
    selected_channels = data_cfg.get("selected_channels")
    if selected_channels in (None, [], "null"):
        return None
    resolved = [int(index) for index in selected_channels]
    if any(index < 0 or index >= observed_channels for index in resolved):
        raise ValueError(f"selected_channels={resolved!r} exceed observed channel count {observed_channels}.")
    return resolved


def _resolve_ratio_splits(label_to_ids: Dict[int, List[str]], split_cfg: Dict[str, Any]) -> SplitManifest:
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    seed = int(split_cfg.get("seed", 0))
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for offset, label in enumerate(sorted(label_to_ids)):
        ids = list(label_to_ids[label])
        rng = np.random.default_rng(seed + offset)
        rng.shuffle(ids)
        total = len(ids)
        train_count = int(round(total * train_ratio))
        val_count = int(round(total * val_ratio))
        if total >= 3:
            train_count = max(1, min(train_count, total - 2))
            val_count = max(1, min(val_count, total - train_count - 1))
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                raise ValueError(f"Cannot resolve non-empty stratified splits for label {label}.")
        train_ids.extend(ids[:train_count])
        val_ids.extend(ids[train_count : train_count + val_count])
        test_ids.extend(ids[train_count + val_count : train_count + val_count + test_count])
    return SplitManifest(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def _resolve_fixed_splits(label_to_ids: Dict[int, List[str]], split_cfg: Dict[str, Any]) -> SplitManifest:
    train_per_class = int(split_cfg["train_per_class"])
    val_per_class = int(split_cfg["val_per_class"])
    test_per_class = int(split_cfg["test_per_class"])
    required = train_per_class + val_per_class + test_per_class
    seed = int(split_cfg.get("seed", 0))
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for offset, label in enumerate(sorted(label_to_ids)):
        ids = list(label_to_ids[label])
        if len(ids) < required:
            raise ValueError(f"Label {label} has only {len(ids)} ids; need {required}.")
        rng = np.random.default_rng(seed + offset)
        rng.shuffle(ids)
        train_ids.extend(ids[:train_per_class])
        val_ids.extend(ids[train_per_class : train_per_class + val_per_class])
        test_ids.extend(ids[train_per_class + val_per_class : required])
    return SplitManifest(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def _resolve_real_splits(filtered_df: pd.DataFrame, split_cfg: Dict[str, Any]) -> SplitManifest:
    strategy = str(split_cfg.get("strategy", "")).strip().lower()
    if not strategy:
        raise ValueError("Real-data configs require data.split.strategy.")

    label_to_ids: dict[int, list[str]] = {}
    for row in filtered_df.itertuples(index=False):
        label = _clean_label(getattr(row, "Label"))
        label_to_ids.setdefault(label, []).append(_stringify_id(getattr(row, "Id")))

    if strategy == "stratified_ratio":
        return _resolve_ratio_splits(label_to_ids, split_cfg)
    if strategy == "stratified_fixed_per_class":
        return _resolve_fixed_splits(label_to_ids, split_cfg)
    raise ValueError(f"Unsupported split strategy: {strategy}")


def build_protocol_from_config(config: Dict[str, Any]) -> DatasetProtocol:
    data_cfg = dict(config.get("data", {}))
    metadata_path = Path(str(data_cfg["metadata_path"])).expanduser().resolve()
    h5_path = Path(str(data_cfg["h5_path"])).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    metadata_df = pd.read_excel(metadata_path)
    selection = dict(data_cfg.get("selection", {}))
    filtered_df = metadata_df.copy()
    if selection.get("name"):
        filtered_df = filtered_df[filtered_df["Name"] == str(selection["name"])]
    if selection.get("dataset_id") is not None:
        dataset_id = int(selection["dataset_id"])
        numeric_dataset_ids = pd.to_numeric(filtered_df["Dataset_id"], errors="coerce")
        filtered_df = filtered_df[numeric_dataset_ids == dataset_id]
    if bool(selection.get("drop_invalid_labels", False)):
        numeric_labels = pd.to_numeric(filtered_df["Label"], errors="coerce")
        filtered_df = filtered_df[numeric_labels.notna()]
        filtered_df = filtered_df[numeric_labels != -1]
    filtered_df = filtered_df.reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"No metadata rows matched selection={selection!r}.")

    split_manifest = _resolve_real_splits(filtered_df, dict(data_cfg.get("split", {})))
    samples: list[SampleMeta] = []
    with h5py.File(h5_path, "r") as handle:
        for row in filtered_df.itertuples(index=False):
            sample_id = _stringify_id(getattr(row, "Id"))
            if sample_id not in handle:
                raise KeyError(f"H5 key {sample_id!r} missing from {h5_path}.")
            metadata_length = _clean_optional_int(getattr(row, "Sample_lenth", None))
            metadata_channels = _clean_optional_int(getattr(row, "Channel", None))
            observed_length, observed_channels = _normalize_h5_shape(handle[sample_id].shape)
            selected_channels = _resolve_selected_channels(data_cfg, observed_channels)
            effective_channels = len(selected_channels) if selected_channels else observed_channels
            operating_condition = _clean_text(getattr(row, "Domain_description", None))
            if not operating_condition:
                operating_condition = _clean_text(getattr(row, "Description", None), default="unknown")
            samples.append(
                SampleMeta(
                    sample_id=sample_id,
                    dataset=str(data_cfg["dataset_name"]),
                    label=_clean_label(getattr(row, "Label")),
                    sampling_rate=int(getattr(row, "Sample_rate")),
                    length=observed_length,
                    channels=effective_channels,
                    operating_condition=operating_condition,
                    source_h5=str(h5_path),
                    dataset_id=_clean_optional_int(getattr(row, "Dataset_id", None)),
                    domain_id=_clean_optional_int(getattr(row, "Domain_id", None)),
                    domain_description=operating_condition,
                    metadata_length=metadata_length,
                    metadata_channels=metadata_channels,
                    observed_length=observed_length,
                    observed_channels=observed_channels,
                )
            )

    selected_channels = data_cfg.get("selected_channels")
    return DatasetProtocol(
        dataset_name=str(data_cfg["dataset_name"]),
        metadata_path=str(metadata_path),
        h5_path=str(h5_path),
        selection=selection,
        split=dict(data_cfg.get("split", {})),
        window=WindowSpec(**dict(data_cfg["window"])),
        samples=samples,
        splits=split_manifest,
        selected_channels=[int(index) for index in selected_channels] if selected_channels else None,
    )


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


def _window_record_id(sample_id: str, window_index: int) -> str:
    return f"{sample_id}__w{window_index:04d}"


def _load_real_signal(handle: h5py.File, sample: SampleMeta, selected_channels: Optional[List[int]]) -> np.ndarray:
    raw = handle[sample.sample_id][()]
    signal = _normalize_h5_array(raw)
    if selected_channels is not None:
        signal = signal[selected_channels, :]
    return signal


def materialize_split_signals(protocol: DatasetProtocol) -> Dict[str, List[SignalRecord]]:
    split_lookup: Dict[str, set[str]] = {
        "train": set(protocol.splits.train_ids),
        "val": set(protocol.splits.val_ids),
        "test": set(protocol.splits.test_ids),
    }
    outputs: Dict[str, List[SignalRecord]] = {"train": [], "val": [], "test": []}

    with h5py.File(protocol.h5_path, "r") as handle:
        for sample in protocol.samples:
            split_name = None
            for current_split, ids in split_lookup.items():
                if sample.sample_id in ids:
                    split_name = current_split
                    break
            if split_name is None:
                continue
            signal = _load_real_signal(handle, sample, protocol.selected_channels)
            for window_index, window_value in enumerate(_window_signal(signal, protocol.window)):
                outputs[split_name].append(
                    SignalRecord(
                        source_sample_id=sample.sample_id,
                        window_id=_window_record_id(sample.sample_id, window_index),
                        window_index=window_index,
                        split=split_name,
                        label=sample.label,
                        window=np.asarray(window_value, dtype=float),
                    )
                )
    return outputs


def materialize_preview_pair(protocol: DatasetProtocol) -> tuple[SignalRecord, SignalRecord]:
    split_records = materialize_split_signals(protocol)
    if not split_records["train"]:
        raise ValueError("Protocol train split produced no windows.")
    if not split_records["test"]:
        raise ValueError("Protocol test split produced no windows.")
    return split_records["train"][0], split_records["test"][0]
