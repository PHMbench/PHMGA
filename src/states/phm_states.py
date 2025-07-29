from __future__ import annotations

from typing import List, Dict, Any, Annotated, Tuple
from pydantic import BaseModel, Field
import uuid
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from typing import Literal
import networkx as nx
from ..tools.signal_processing_schemas import PHMOperator

Shape = Tuple[int, ...]  # 支持多维形状

class _NodeBase(BaseModel):
    node_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    parents: List[str] | str     # 上游 node_id 列表（源节点为空）
    stage: Literal["input", "processed", "output"] = "input"  # 节点阶段
    shape: Shape
    kind: Literal["signal"] = "signal"

class InputData(_NodeBase):
    """Represents a batch of raw input signals."""

    stage: str = "input"
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary containing raw signal data, where keys are signal names and values are the corresponding signal data."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedData(_NodeBase):
    """Output of a single signal processing method."""
    stage: str = "processed"
    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    processed_data: Any


class FeatureData(_NodeBase):
    """Feature set extracted from a batch of signals."""
    stage: str = "output"
    feature_set_id: str = Field(default_factory=lambda: f"feat_{uuid.uuid4().hex[:8]}")
    source_processed_id: str
    features: List[Dict[str, float]]

# TODO
class AnalysisInsight(BaseModel):
    """Concrete insight produced by analysis or reflection."""

    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    content: str
    severity_score: float = Field(ge=0.0, le=1.0)
    supporting_feature_ids: List[str]

class DAGState(BaseModel):
    """只保存拓扑信息，不含业务数据"""
    user_instruction: str
    reference_root: str
    test_root: str
    nodes: Dict[str, Any] = {}
    leaves: List[str] = []           # 当前末端信号节点



class DAGTracker:
    """运行期辅助：把新执行写入 DAGState 并维护 networkx 图."""

    def __init__(self, dag_state: DAGState):
        self.state = dag_state
        self.g = nx.DiGraph()
        for n in dag_state.nodes.values():
            self.g.add_node(n.node_id)
            for p in n.parents:
                self.g.add_edge(p, n.node_id)

    # ---------- 写入一次执行 ---------- #
    def add_execution(
        self,
        op: _NodeBase | PHMOperator | List[_NodeBase],

    ) -> str:

        self._add_node(op)

        self.state.leaves = [op.node_id]
        return op.node_id  # 返回新信号 node_id

    # ---------- 导出给 LLM ---------- #
    def export_json(self, max_nodes: int = 40) -> str:
        """Serialize a trimmed version of the DAG for LLM consumption."""
        import json

        topo = list(nx.topological_sort(self.g))[-max_nodes:]
        mini = []
        for nid in topo:
            n = self.state.nodes[nid]
            mini.append(
                n.dict(
                    include={
                        "node_id",
                        "kind",
                        "stage",
                        "op_name",
                        "rank",
                        "shape",
                        "in_shape",
                        "out_shape",
                        "parents",
                    }
                )
            )
        return json.dumps({"graph": mini, "user_instruction": self.state.user_instruction})

    # ---------- 可视化 ---------- #
    def to_dot(self) -> "graphviz.Digraph":
        """Convert the internal graph into a ``graphviz`` object."""
        import graphviz

        dot = graphviz.Digraph()
        for nid in self.g.nodes:
            n = self.state.nodes[nid]
            if isinstance(n, PHMOperator):
                label = getattr(n, "op_name", nid)
                shape = "box"
                color = "lightblue"
            else:
                label = nid
                shape = "ellipse"
                color = "lightgray"
            dot.node(nid, label=label, shape=shape, style="filled", fillcolor=color)
        for u, v in self.g.edges:
            dot.edge(u, v)
        return dot

    def write_png(self, path: str) -> None:
        """Render the DAG to a PNG image on disk."""
        dot = self.to_dot()
        dot.render(filename=path, format="png", cleanup=True)

    # ---------- 内部 ---------- #
    def _add_node(self, n):
        self.state.nodes[n.node_id] = n
        self.g.add_node(n.node_id)
        for p in n.parents:
            self.g.add_edge(p, n.node_id)

# TODO
    def transfer_to_langgraph(self) -> nx.DiGraph:
        """将 DAGState 转换为 LangGraph 可用的 networkx 图."""
        return self.g



class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    user_instruction: str
    reference_signal: InputData
    test_signal: InputData

    high_level_plan: List[str] = Field(default_factory=list)
    needs_revision: bool = False


    reflection_history: List[str] = Field(default_factory=list)
    is_sufficient: bool = False
    iteration_count: int = 0

    processed_signals: Annotated[Dict[str, ProcessedData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )
    extracted_features: Annotated[Dict[str, FeatureData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )

    analysis_results: List[AnalysisInsight] = Field(default_factory=list)
    final_decision: str = ""

    final_report: str = ""

    # ---------- DAG ---------- #
    # ---------- DAG 状态 ---------- #
    dag_state: DAGState = Field(
        default_factory=lambda: DAGState(user_instruction="", reference_root="", test_root="")
    )

    dag_tracker: DAGTracker | None = None

    # ---- 快捷方法供 Agents 使用 ---- #
    def tracker(self) -> DAGTracker:
        if self.dag_tracker is None:
            self.dag_tracker = DAGTracker(self.dag_state)
        return self.dag_tracker

    def add_execution(self, *args, **kw) -> str:
        """代理给 DAGTracker.add_execution 并返回新 signal node_id."""
        return self.tracker().add_execution(*args, **kw)
    

