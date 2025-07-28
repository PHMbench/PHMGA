"""Configuration dataclass for the PHM system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["Config"]


@dataclass
class Config:
    use_patch: bool = False
    patch_size: int = 256
    signal_processing_methods: List[str] = field(default_factory=list)
    feature_methods: List[str] = field(default_factory=list)
    similarity_method: str = "euclidean"
    decision_model: str = "threshold"
    max_loops: int = 3
