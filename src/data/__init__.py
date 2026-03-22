"""Dataset protocol and materialization exports."""

from .protocol import (
    DatasetProtocol,
    SampleMeta,
    SignalRecord,
    SplitManifest,
    WindowSpec,
    build_protocol_from_config,
    materialize_preview_pair,
    materialize_split_signals,
)

__all__ = [
    "DatasetProtocol",
    "SampleMeta",
    "SignalRecord",
    "SplitManifest",
    "WindowSpec",
    "build_protocol_from_config",
    "materialize_preview_pair",
    "materialize_split_signals",
]
