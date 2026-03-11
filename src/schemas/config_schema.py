from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataSelectionSpec(BaseModel):
    mode: Literal["fixed_ids", "query_metadata", "split_from_metadata", "predefined_split"] = "fixed_ids"
    train_ids: List[int] = Field(default_factory=list)
    val_ids: List[int] = Field(default_factory=list)
    test_ids: List[int] = Field(default_factory=list)


class GraphSelectionSpec(BaseModel):
    selected: str = Field(min_length=1)


class ResolvedConfig(BaseModel):
    project: Dict[str, Any] = Field(default_factory=dict)
    system: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    graphs: Dict[str, Any] = Field(default_factory=dict)
    cases: Dict[str, Any] = Field(default_factory=dict)
    layers: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    experiments: Dict[str, Any] = Field(default_factory=dict)
    profiles: Dict[str, Any] = Field(default_factory=dict)
    hydra: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")
