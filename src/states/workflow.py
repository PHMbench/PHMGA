"""Workflow contracts shared by the rebuilt front-end agents.

The paper-oriented workflow keeps a small but explicit state object so the
front-end contract is inspectable in tests and reports:

``signal_context -> step_plan -> execute results/gaps -> reflections -> report``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.dag import DagJson


GraphPath = Literal["dag_only", "ml", "torch"]
ReflectionDecision = Literal["finish", "need_patch", "need_replan", "halt"]


class SignalContext(BaseModel):
    """Prompt-safe summary of the current signal input space.

    The planner does not receive raw windows directly. It receives a compact
    summary that still exposes the PHM-relevant planning facts: channel count,
    window shape, sampling rate, and the available root identifiers.
    """

    dataset_name: str
    channel_count: int
    window_shape: List[int]
    sampling_rate: int
    source_mode: str
    root_node_ids: List[str] = Field(default_factory=list)
    representative_sample_id: Optional[str] = None
    available_splits: List[str] = Field(default_factory=lambda: ["train", "val", "test"])


class PlanStep(BaseModel):
    """NVTA-style execution step produced by the planner."""

    parent: str
    op_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StepPlan(BaseModel):
    """Structured planner output consumed by the execute agent."""

    plan: List[PlanStep] = Field(default_factory=list)


class ExecutionGap(BaseModel):
    """One recoverable or blocking issue encountered during plan execution."""

    step_index: int
    parent: str
    op_name: str
    message: str
    recoverable: bool = True


class ReflectionResult(BaseModel):
    """Structured reflection result aligned with the NVTA decision contract."""

    decision: ReflectionDecision
    reason: str
    missing_operators: List[str] = Field(default_factory=list)
    shape_risks: List[str] = Field(default_factory=list)
    structural_warnings: List[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    """Front-end state for plan, execute, reflect, and report stages."""

    user_instruction: str
    dataset_name: str
    graph_path: GraphPath
    data_context: Dict[str, Any] = Field(default_factory=dict)
    signal_context: Optional[SignalContext] = None
    step_plan: Optional[StepPlan] = None
    execution_results: Dict[str, Any] = Field(default_factory=dict)
    execution_gaps: List[ExecutionGap] = Field(default_factory=list)
    reflection_history: List[str] = Field(default_factory=list)
    reflection_results: List[ReflectionResult] = Field(default_factory=list)
    dag: Optional[DagJson] = None
    artifact_index: Dict[str, str] = Field(default_factory=dict)
    status: str = "initialized"
