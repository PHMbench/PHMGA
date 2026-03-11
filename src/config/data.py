from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from src.schemas.config_schema import DataSelectionSpec


REQUIRED_METADATA_COLUMNS = [
    "Id",
    "Dataset_id",
    "Name",
    "Type",
    "File",
    "Label",
    "Label_Description",
    "Sample_Rate",
    "Length",
    "Channels",
]

LEGACY_FIXED_ID_KEYS = ("ref_ids", "val_ids", "test_ids", "train_ids")


def resolve_source_mode(config_like: Mapping[str, Any]) -> str:
    data_cfg = dict(config_like.get("data") or {})
    source_mode = str(data_cfg.get("source_mode") or "").strip().lower()
    if source_mode in {"fixed_ids", "vibench"}:
        return source_mode
    backend = str(data_cfg.get("backend") or "").strip().lower()
    if backend == "vibench":
        return "vibench"
    return "fixed_ids"


def resolve_data_selection(config_like: Mapping[str, Any]) -> DataSelectionSpec:
    data_cfg = dict(config_like.get("data") or {})
    selection = dict(data_cfg.get("selection") or {})
    mode = str(selection.get("mode") or data_cfg.get("selection_mode") or "").strip().lower()
    source_mode = resolve_source_mode(config_like)
    if source_mode == "fixed_ids" and not mode:
        mode = "fixed_ids"

    train_ids = selection.get("train_ids")
    if train_ids is None:
        train_ids = data_cfg.get("train_ids")
    if train_ids is None:
        train_ids = data_cfg.get("ref_ids")
    if train_ids is None:
        train_ids = config_like.get("train_ids")
    if train_ids is None:
        train_ids = config_like.get("ref_ids")

    val_ids = selection.get("val_ids")
    if val_ids is None:
        val_ids = data_cfg.get("val_ids")
    if val_ids is None:
        val_ids = config_like.get("val_ids")

    test_ids = selection.get("test_ids")
    if test_ids is None:
        test_ids = data_cfg.get("test_ids")
    if test_ids is None:
        test_ids = config_like.get("test_ids")

    payload = {
        "mode": mode or "fixed_ids",
        "train_ids": list(train_ids or []),
        "val_ids": list(val_ids or []),
        "test_ids": list(test_ids or []),
    }
    return DataSelectionSpec.model_validate(payload)


def normalize_runtime_config(config_like: Mapping[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(dict(config_like))
    data_cfg = dict(payload.get("data") or {})
    source_mode = resolve_source_mode(payload)
    data_cfg["source_mode"] = source_mode

    if source_mode == "fixed_ids":
        selection = resolve_data_selection(payload)
        data_cfg["selection"] = selection.model_dump()

    for key in ("metadata_path", "h5_path"):
        if payload.get(key) is not None and data_cfg.get(key) is None:
            data_cfg[key] = payload.get(key)

    for key in LEGACY_FIXED_ID_KEYS:
        payload.pop(key, None)
        data_cfg.pop(key, None)

    payload["data"] = data_cfg
    return payload


def validate_metadata_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] | None = None,
) -> Dict[str, Any]:
    required = list(required_columns or REQUIRED_METADATA_COLUMNS)
    present = list(frame.columns)
    missing = [name for name in required if name not in present]
    return {
        "ok": not missing,
        "required_columns": required,
        "present_columns": present,
        "missing_columns": missing,
    }


def _filter_ids(frame: pd.DataFrame, *, id_column: str, ids: Iterable[int]) -> pd.DataFrame:
    normalized = [int(value) for value in ids]
    return frame[frame[id_column].isin(normalized)].copy()


def select_fixed_ids(
    frame: pd.DataFrame,
    *,
    id_column: str = "Id",
    train_ids: Iterable[int] | None = None,
    val_ids: Iterable[int] | None = None,
    test_ids: Iterable[int] | None = None,
) -> Dict[str, pd.DataFrame]:
    return {
        "train": _filter_ids(frame, id_column=id_column, ids=list(train_ids or [])),
        "val": _filter_ids(frame, id_column=id_column, ids=list(val_ids or [])),
        "test": _filter_ids(frame, id_column=id_column, ids=list(test_ids or [])),
    }


def build_metadata_snapshot(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
    }
