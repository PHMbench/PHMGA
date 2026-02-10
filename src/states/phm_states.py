from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
import uuid
from typing_extensions import Annotated
import operator
from typing import Literal
import networkx as nx
from ..tools.signal_processing_schemas import PHMOperator
from ..schemas.insight_schema import AnalysisInsight
from ..schemas.plan_schema import AnalysisPlan

Shape = Tuple[int, ...]  # 支持多维形状


class TrainReport(BaseModel):
    """Structured inner-loop training report (aligns with doc/plan/*/AGENT_IO.md)."""

    run_id: str = Field(..., description="Unique run id for this training job.")
    dataset_id: str = Field(default="", description="Dataset identifier (e.g. case/dataset name).")
    task_id: str = Field(default="", description="Task identifier (optional).")
    split_protocol: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: Dict[str, Any] = Field(default_factory=dict)
    explain_summary: Dict[str, Any] = Field(default_factory=dict)
    error_modes: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class ConfigPatch(BaseModel):
    """Outer-loop patch output (white-listed config changes only)."""

    diff_summary: str = ""
    reason_codes: List[str] = Field(default_factory=list)
    config_patch: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class _NodeBase(BaseModel):
    node_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    parents: List[str] | str     # 上游 node_id 列表（源节点为空）
    stage: Literal["input", "processed", "similarity", "dataset", "output"] = "input"  # 节点阶段
    shape: Shape
    kind: Literal["signal"] = "signal"
    sim: Dict[str, Any] = Field(default_factory=dict, description="Similarity metrics")


class InputData(_NodeBase):
    """Represents a batch of raw input signals."""

    stage: str = "input"
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary containing raw signal data, where keys are signal names and values are the corresponding signal data."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)  # see metadata
    results: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict) # 添加 meta 字段


class ProcessedData(_NodeBase):
    """Output of a single signal processing method."""
    stage: str = "processed"
    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    # processed_data: Any
    results: Any = None
    meta: Dict[str, Any] = Field(default_factory=dict)

# ---------- Dataset Node ---------- #
class DataSetNode(_NodeBase):
    """Represents a dataset derived from a processed node."""

    stage: str = "dataset"
    meta: Dict[str, Any] = Field(default_factory=dict)


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
    channels: List[str]
    nodes: Dict[str, Any] = Field(default_factory=dict)
    leaves: List[str] = Field(default_factory=list)           # 当前末端信号节点
    error_log: List[str] = Field(default_factory=list)
    graph_path: str | None = None

    
    def __init__(self, **data):
        super().__init__(**data)
        # 初始化时确保至少有一个叶子节点
        if not self.leaves:
            self.leaves = list(self.channels)
        # 确保根节点存在
        if not self.nodes:
            for ch in self.channels:
                self.nodes[ch] = InputData(node_id=ch, parents=[], shape=(0,), stage="input")



class DAGTracker:
    """运行期辅助：把新执行写入 DAGState 并维护 networkx 图."""

    def __init__(self, dag_state: DAGState):
        self.update(dag_state)
    def update(self, dag_state: DAGState):
        """Update the tracker with a new DAGState."""
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

    def _build_dot_source(self) -> str:
        """Build DOT source without requiring ``python-graphviz``."""
        def _escape(text: str) -> str:
            return text.replace("\\", "\\\\").replace('"', '\\"')

        lines = ["digraph G {"]
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
            lines.append(
                f'  "{_escape(str(nid))}" [label="{_escape(str(label))}", shape="{shape}", style="filled", fillcolor="{color}"];'
            )
        for u, v in self.g.edges:
            lines.append(f'  "{_escape(str(u))}" -> "{_escape(str(v))}";')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_png(self, path: str) -> bool:
        """Render the DAG to a PNG image on disk; return success flag."""
        base = path[:-4] if path.endswith(".png") else path
        png_path = f"{base}.png"
        dot_path = f"{base}.dot"
        dot = None
        try:
            dot = self.to_dot()
            dot.render(filename=base, format="png", cleanup=True)
            if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
                raise RuntimeError("Graph render produced empty PNG output.")
            return True
        except Exception as exc:
            dot_source: str
            try:
                dot_source = getattr(dot, "source", "") if dot is not None else ""
            except Exception:
                dot_source = ""
            if not dot_source:
                dot_source = self._build_dot_source()
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(dot_source)
            try:
                if os.path.exists(png_path) and os.path.getsize(png_path) == 0:
                    os.remove(png_path)
            except Exception:
                pass
            self.state.error_log.append(
                f"Graph PNG export failed: {type(exc).__name__}: {exc}. DOT fallback saved to: {dot_path}"
            )
            return False

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
            self.state.leaves = list(self.state.channels)



class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    case_name: str = "" # Add case_name to the state
    user_instruction: str = Field(default="", description="User's instruction for the PHM analysis.")
    reference_signal: InputData
    test_signal: InputData
    dag_state: DAGState
    min_depth: int = 4
    min_width: int = 4
    max_depth: int = 8
    fs: float | None = Field(default=None, description="Sampling frequency of the signals in Hz.")

    # high_level_plan: List[str] = Field(default_factory=list)
    # analysis_plan: AnalysisPlan | None = None
    needs_revision: bool = False

    detailed_plan: List[dict] = Field(default_factory=list)
    executed_steps: int = Field(default=0, description="Number of steps executed in the last ExecuteAgent run.")
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
    datasets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    dataset_path: str | None = None
    model_path: str | None = None
    accuracy: Optional[float] = None
    ml_results: Dict[str, Any] = Field(default_factory=dict)

    # --- New: inner-loop training report history (for reflection/outer-loop) ---
    train_history: List[TrainReport] = Field(default_factory=list)

    # --- New: current immutable TSPN config snapshot (dict) ---
    current_model_config: Dict[str, Any] = Field(default_factory=dict)

    # --- New: real-data backend configuration (e.g., PHM-Vibench data_factory) ---
    data_cfg: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data backend configuration (e.g., {backend: vibench, data_dir, metadata_file, dataset_name, ...}).",
    )

    # --- Data boundary / training control (SPEC redlines) ---
    labels_ref: Dict[str, Any] = Field(default_factory=dict, description="Train/val-visible labels.")
    labels_tst: Dict[str, Any] = Field(default_factory=dict, description="Test labels (default not visible).")
    allow_test_labels_for_reporting: bool = Field(
        default=False, description="If true, allow using labels_tst for reporting-only metrics."
    )

    # --- Backend selection / model config ---
    task_type: Literal["signal_processing_dag", "neuro_symbolic_train"] = Field(
        default="signal_processing_dag",
        description="Execution mode: classic signal-processing DAG build, or neuro-symbolic (TSPN) training.",
    )
    train_backend: str = Field(default="shallow", description="Training backend: shallow|tspn|both.")
    model_config_path: str | None = Field(default=None, description="Path to model_config.yaml for TSPN.")
    save_dir: str | None = Field(default=None, description="Base directory to save artifacts.")
    run_dir: str | None = Field(default=None, description="Resolved run directory for this execution.")

    _tracker_instance: Optional[Any] = PrivateAttr(default=None)

    def tracker(self) -> "DAGTracker":
        if self._tracker_instance is None:
            self._tracker_instance = DAGTracker(self.dag_state)
        return self._tracker_instance

    class Config:
        arbitrary_types_allowed = True
