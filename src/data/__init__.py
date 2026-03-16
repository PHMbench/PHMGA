"""Canonical data-protocol exports."""

from .dataset_preparer import (
    DatasetView,
    TorchDatasetView,
    build_dataset_views,
    build_dataset_views_np,
    build_dataset_views_pt,
)
from .protocol import (
    DatasetProtocol,
    SampleMeta,
    SignalRecord,
    SplitManifest,
    WindowSpec,
    build_protocol_from_config,
    materialize_proxy_split_signals,
    materialize_preview_signal,
    materialize_split_signals,
)

__all__ = [
    "DatasetView",
    "TorchDatasetView",
    "DatasetProtocol",
    "SampleMeta",
    "SignalRecord",
    "SplitManifest",
    "WindowSpec",
    "build_dataset_views",
    "build_dataset_views_np",
    "build_dataset_views_pt",
    "build_protocol_from_config",
    "materialize_proxy_split_signals",
    "materialize_preview_signal",
    "materialize_split_signals",
]
