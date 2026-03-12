from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.dag import DagJson


class WorkflowState(BaseModel):
    user_instruction: str
    dataset_name: str
    graph_path: Literal["dag_only", "ml", "torch"]
    data_context: Dict[str, Any] = Field(default_factory=dict)
    plan: List[str] = Field(default_factory=list)
    reflection_history: List[str] = Field(default_factory=list)
    dag: Optional[DagJson] = None
    artifact_index: Dict[str, str] = Field(default_factory=dict)
    status: str = "initialized"
