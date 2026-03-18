"""PHMState-style workflow state for the LangGraph front-end runtime."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.dag import DAGTracker as CanonicalDAGTracker
from src.dag import DagJson, DagNode
from .workflow import ExecutionGap, ReflectionResult, RoundTrace, SignalContext, StepPlan


class DAGState(BaseModel):
    """Topology-oriented DAG holder exposed at the PHMState layer."""

    dag: Optional[DagJson] = None
    error_log: List[str] = Field(default_factory=list)
    graph_path: Optional[str] = None

    def node_map(self) -> Dict[str, DagNode]:
        if self.dag is None:
            return {}
        return {node.node_id: node for node in self.dag.nodes}

    def leaves(self) -> List[str]:
        if self.dag is None:
            return []
        source_ids = {edge.source for edge in self.dag.edges}
        return [node.node_id for node in self.dag.nodes if node.node_id not in source_ids]

    def export_json(self, max_nodes: int = 40) -> str:
        if self.dag is None:
            return json.dumps({"nodes": [], "edges": []}, ensure_ascii=False)
        nodes = [
            {
                "node_id": node.node_id,
                "parents": node.parents,
                "kind": node.kind,
                "operator_category": node.operator_category,
                "rank_class": node.rank_class,
                "in_shape": node.in_shape,
                "out_shape": node.out_shape,
            }
            for node in self.dag.nodes[-max_nodes:]
        ]
        visible_ids = {node["node_id"] for node in nodes}
        edges = [
            edge.model_dump()
            for edge in self.dag.edges
            if edge.source in visible_ids and edge.target in visible_ids
        ]
        return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)


class DAGTracker:
    """Adapter that keeps the canonical validated DAG JSON as the source of truth."""

    def __init__(self, dag_state: DAGState) -> None:
        self._dag_state = dag_state
        self._tracker = CanonicalDAGTracker()
        if dag_state.dag is not None:
            for node in dag_state.dag.nodes:
                self._tracker.add_node(node)

    def add_node(self, node: DagNode) -> str:
        self._tracker.add_node(node)
        self._dag_state.dag = self._tracker.export()
        return node.node_id

    def export(self) -> DagJson:
        self._dag_state.dag = self._tracker.export()
        return self._dag_state.dag

    def export_json(self, max_nodes: int = 40) -> str:
        self.export()
        return self._dag_state.export_json(max_nodes=max_nodes)

    def restore(self, dag: Optional[DagJson]) -> None:
        self._tracker = CanonicalDAGTracker()
        if dag is not None:
            for node in dag.nodes:
                self._tracker.add_node(node)
        self._dag_state.dag = dag.model_copy(deep=True) if dag else None


class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_instruction: str
    dataset_name: str
    graph_path: str
    runtime_config: Dict[str, Any] = Field(default_factory=dict)
    data_context: Dict[str, Any] = Field(default_factory=dict)
    signal_context: Optional[SignalContext] = None
    step_plan: Optional[StepPlan] = None
    execution_results: Dict[str, Any] = Field(default_factory=dict)
    execution_gaps: List[ExecutionGap] = Field(default_factory=list)
    reflection_history: List[str] = Field(default_factory=list)
    reflection_results: List[ReflectionResult] = Field(default_factory=list)
    dag_quality_summary: Dict[str, Any] = Field(default_factory=dict)
    iteration_index: int = 0
    max_iterations: int = 4
    round_history: List[RoundTrace] = Field(default_factory=list)
    last_stable_dag: Optional[DagJson] = None
    last_stable_execution_results: Dict[str, Any] = Field(default_factory=dict)
    artifact_index: Dict[str, str] = Field(default_factory=dict)
    status: str = "initialized"
    dag_state: DAGState = Field(default_factory=DAGState)
    compiled_bundle: Any = None
    compiled_manifest: Dict[str, Any] = Field(default_factory=dict)
    path_artifacts: Dict[str, Any] = Field(default_factory=dict)
    final_report: str = ""
    halt_reason: Optional[str] = None
    current_round_input_hash: str = ""
    current_round_previous_node_ids: List[str] = Field(default_factory=list)

    @property
    def dag(self) -> Optional[DagJson]:
        return self.dag_state.dag

    @dag.setter
    def dag(self, value: Optional[DagJson]) -> None:
        self.dag_state.dag = value

    def tracker(self) -> DAGTracker:
        return DAGTracker(self.dag_state)

    def stable_snapshot(self) -> None:
        self.last_stable_dag = self.dag.model_copy(deep=True) if self.dag else None
        self.last_stable_execution_results = deepcopy(self.execution_results)

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        payload = super().model_dump(*args, **kwargs)
        payload["dag"] = self.dag.model_dump() if self.dag else None
        return payload
