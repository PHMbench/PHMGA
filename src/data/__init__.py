"""Dataset protocol and materialization exports."""

from .protocol import (
    DatasetProtocol,
    SampleMeta,
    SignalRecord,
    SplitManifest,
    WindowSpec,
    build_protocol_from_config,
    export_split_manifest,
    materialize_preview_pair,
    materialize_split_signals,
    summarize_protocol,
    summarize_split_ids_by_label_domain,
    summarize_split_records,
)

__all__ = [
    "DatasetProtocol",
    "SampleMeta",
    "SignalRecord",
    "SplitManifest",
    "WindowSpec",
    "build_protocol_from_config",
    "export_split_manifest",
    "materialize_preview_pair",
    "materialize_split_signals",
    "summarize_protocol",
    "summarize_split_ids_by_label_domain",
    "summarize_split_records",
]
