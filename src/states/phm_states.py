from __future__ import annotations

from typing import List, Dict, Any, Annotated, Tuple
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
import uuid
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from typing import Literal
import networkx as nx
from ..tools.signal_processing_schemas import PHMOperator
from ..schemas.insight_schema import AnalysisInsight
from ..schemas.plan_schema import AnalysisPlan

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
    metadata: Dict[str, Any] = Field(default_factory=dict) # see metadata 


class ProcessedData(_NodeBase):
    """Output of a single signal processing method."""
    stage: str = "processed"
    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    processed_data: Any


# class FeatureData(_NodeBase):
#     """Feature set extracted from a batch of signals."""
#     stage: str = "output"
#     feature_set_id: str = Field(default_factory=lambda: f"feat_{uuid.uuid4().hex[:8]}")
#     source_processed_id: str
#     features: List[Dict[str, float]]

# TODO

class Result(BaseModel):
    """
    Represents the final result of a PHM analysis, structured to constrain LLM output based on a predefined schema.
    """

    dataset: str | None = Field(None, description="Identifier for the dataset used.")
    Description: str | None = Field(None, description="A brief description of the analysis performed.")
    Label: int | None = Field(None, description="The primary label assigned to the result (e.g., fault type).")
    Label_Description: str | None = Field(None, description="Description of the assigned label.")
    Fault_level: float | None = Field(None, description="Severity level of the detected fault (e.g., 'Normal', 'Warning', 'Critical').")
    RUL_label: float | None = Field(None, description="Categorical label for Remaining Useful Life.")
    RUL_label_description: str | None = Field(None, description="Description of the RUL label.")
    Domain_id: int | None = Field(None, description="Identifier for the operational domain.")
    Domain_description: str | None = Field(None, description="Description of the operational domain.")
    Sample_rate: int | None = Field(None, description="The sample rate of the signal data in Hz.")
    Sample_length: int | None = Field(None, description="The length of the data sample used.")
    Channel: int | None = Field(None, description="The specific data channel or sensor analyzed.")
    Fault_Diagnosis: str = Field(..., description="The conclusive diagnosis of the fault. This field is mandatory.")
    Anomaly_Detection: str = Field(..., description="Results of the anomaly detection process. This field is mandatory.")
    Remaining_Life: str | None = Field(None, description="Predicted Remaining Useful Life in appropriate units (e.g., cycles, hours).")

class DAGState(BaseModel):
    """只保存拓扑信息，不含业务数据"""
    user_instruction: str
    reference_root: str
    test_root: str
    nodes: Dict[str, Any] = Field(default_factory=dict)
    leaves: List[str] = Field(default_factory=list)           # 当前末端信号节点

    
    def __init__(self, **data):
        super().__init__(**data)
        # 初始化时确保至少有一个叶子节点
        if not self.leaves:
            self.leaves = [self.reference_root, self.test_root]
        # 确保根节点存在
        if not self.nodes:
            self.nodes[self.reference_root] = InputData(node_id=self.reference_root, parents=[], shape=(0,), stage="input")
            self.nodes[self.test_root] = InputData(node_id=self.test_root, parents=[], shape=(0,), stage="input")



class DAGTracker:
    """运行期辅助：把新执行写入 DAGState 并维护 networkx 图."""

    def __init__(self, dag_state: DAGState):
        self.state = dag_state
        self.g = nx.DiGraph()
        if dag_state.nodes:
            for node_id, node in dag_state.nodes.items():
                self.g.add_node(node_id)
                # Ensure parents is a list before iterating
                parents = node.parents if isinstance(node.parents, list) else [node.parents]
                for p in parents:
                    if p: # Avoid adding edges for empty parent lists
                        self.g.add_edge(p, node_id)

    # ---------- 写入一次执行 ---------- #
    # Let's rename add_execution to add_node for clarity. Its job is to add a node to the graph structure.
    def add_node(self, node: _NodeBase) -> str:
        """
        Adds a new node to the state, updates the networkx graph, and correctly updates leaves.
        """
        if node.node_id in self.state.nodes:
            # Avoid adding duplicate nodes
            return node.node_id

        self.state.nodes[node.node_id] = node
        self.g.add_node(node.node_id)
        
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        
        for p in parents:
            if p and p in self.g:
                self.g.add_edge(p, node.node_id)

        # --- CRITICAL FIX FOR LEAVES ---
        # 1. Start with the existing leaves.
        # 2. Remove any parents of the new node from the leaves list.
        # 3. Add the new node to the leaves list.
        # This correctly handles branching and merging.
        current_leaves = self.state.leaves[:]
        new_leaves = [leaf for leaf in current_leaves if leaf not in parents]
        new_leaves.append(node.node_id)
        self.state.leaves = new_leaves

        return node.node_id

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
        # return json.dumps({"graph": mini, "user_instruction": self.state.user_instruction})
        return json.dumps({"graph": mini})

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


def get_node_data(state: "PHMState", node_id: str):
    """Utility to fetch raw array data from a node."""
    node = state.dag_state.nodes.get(node_id)
    if isinstance(node, InputData):
        return np.asarray(node.data.get("signal", []))
    if isinstance(node, ProcessedData):
        return np.asarray(node.processed_data)
    return None

# TODO
    def transfer_to_langgraph(self) -> nx.DiGraph:
        """将 DAGState 转换为 LangGraph 可用的 networkx 图."""
        return self.g
    def save(self, path: str) -> None:
        """将 DAG 状态保存到指定路径."""
        import json
        with open(path, 'w') as f:
            json.dump(self.state.dict(), f, indent=4)
    def load(self, path: str) -> None:
        """从指定路径加载 DAG 状态."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.state = DAGState(**data)
            self.g = nx.DiGraph()
            for n in self.state.nodes.values():
                self._add_node(n)
            self.state.leaves = [self.state.reference_root, self.state.test_root]



class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    user_instruction: str
    reference_signal: InputData
    test_signal: InputData

    high_level_plan: List[str] = Field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None
    needs_revision: bool = False

    detailed_plan: List[dict] = Field(default_factory=list)
    error_logs: List[str] = Field(default_factory=list)


    reflection_history: List[str] = Field(default_factory=list)
    is_sufficient: bool = False
    iteration_count: int = 0

    processed_reference_signals: Annotated[Dict[str, ProcessedData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )
    processed_test_signals: Annotated[Dict[str, ProcessedData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )
    extracted_reference_features: Annotated[Dict[str, ProcessedData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )
    extracted_test_features: Annotated[Dict[str, ProcessedData], lambda x, y: {**x, **y}] = Field(
        default_factory=dict
    )

    insights: List[AnalysisInsight] = Field(
        default_factory=list, description="Insights generated by the inquirer"
    )
    final_decision: str = ""

    final_report: str = ""

    # ---------- DAG ---------- #
    # ---------- DAG 状态 ---------- #
    dag_state: DAGState = Field(
        default_factory=lambda: DAGState(user_instruction="", reference_root="", test_root="")
    )

    error_logs: List[str] = Field(default_factory=list)
    # _dag_tracker: DAGTracker | None = PrivateAttr(default=None) # 1. Exclude from serialization

    class Config: # 2. Allow arbitrary types
        arbitrary_types_allowed = True

    # ---- 快捷方法供 Agents 使用 ---- #
    def tracker(self) -> DAGTracker:
        if not hasattr(self, "_dag_tracker") or self._dag_tracker is None:
            self._dag_tracker = DAGTracker(self.dag_state)
        return self._dag_tracker

    def add_execution(self, *args, **kw) -> str:
        """代理给 DAGTracker.add_execution 并返回新 signal node_id."""
        return self.tracker().add_node(*args, **kw)
    

