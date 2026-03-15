"""Canonical data-protocol exports."""

from .dataset_preparer import DatasetView, build_dataset_views
from .protocol import (
    DatasetProtocol,
    SampleMeta,
    SignalRecord,
    SplitManifest,
    WindowSpec,
    build_protocol_from_config,
    materialize_preview_signal,
    materialize_split_signals,
)

__all__ = [
    "DatasetView",
    "DatasetProtocol",
    "SampleMeta",
    "SignalRecord",
    "SplitManifest",
    "WindowSpec",
    "build_dataset_views",
    "build_protocol_from_config",
    "materialize_preview_signal",
    "materialize_split_signals",
]
