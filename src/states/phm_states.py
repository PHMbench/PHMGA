from __future__ import annotations

from typing import List, Dict, Any, Annotated, Tuple, Optional, TypedDict, Union
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, validator
import uuid
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from typing import Literal
import networkx as nx
from ..tools.signal_processing_schemas import PHMOperator
from ..schemas.insight_schema import AnalysisInsight
from ..schemas.plan_schema import AnalysisPlan

# Load environment variables
load_dotenv()


class UnifiedStateManager(BaseModel):
    """
    Unified state management system that consolidates configuration objects,
    state variables, and environment variables with type safety and validation.
    """

    # Core configuration
    _config_data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _env_data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _state_data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # Deprecation tracking
    _deprecated_access: Dict[str, int] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        self._load_environment_variables()
        self._load_default_configuration()

    def _load_environment_variables(self):
        """Load and categorize environment variables."""
        env_mapping = {
            # LLM Configuration
            'GEMINI_API_KEY': ('llm', 'gemini_api_key'),
            'OPENAI_API_KEY': ('llm', 'openai_api_key'),
            'LLM_PROVIDER': ('llm', 'provider'),
            'LLM_MODEL': ('llm', 'model'),
            'QUERY_GENERATOR_MODEL': ('llm', 'query_generator_model'),
            'REFLECTION_MODEL': ('llm', 'reflection_model'),
            'ANSWER_MODEL': ('llm', 'answer_model'),

            # Data Paths
            'PHM_DATA_DIR': ('paths', 'data_dir'),
            'PHM_SAVE_DIR': ('paths', 'save_dir'),
            'PHM_CACHE_DIR': ('paths', 'cache_dir'),

            # Processing Configuration
            'PHM_MAX_DEPTH': ('processing', 'max_depth'),
            'PHM_MIN_DEPTH': ('processing', 'min_depth'),
            'PHM_MAX_STEPS': ('processing', 'max_steps'),

            # System Configuration
            'LANGCHAIN_TRACING_V2': ('system', 'langchain_tracing'),
            'LANGCHAIN_ENDPOINT': ('system', 'langchain_endpoint'),
            'LANGCHAIN_API_KEY': ('system', 'langchain_api_key'),
            'LANGCHAIN_PROJECT': ('system', 'langchain_project'),

            # Debug and Testing
            'FAKE_LLM': ('debug', 'fake_llm'),
            'DEBUG_MODE': ('debug', 'debug_mode'),
            'LOG_LEVEL': ('debug', 'log_level'),
        }

        for env_var, (category, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if category not in self._env_data:
                    self._env_data[category] = {}

                # Type conversion
                if key in ['max_depth', 'min_depth', 'max_steps']:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif key in ['fake_llm', 'debug_mode', 'langchain_tracing']:
                    value = value.lower() in ('true', '1', 'yes', 'on')

                self._env_data[category][key] = value

    def _load_default_configuration(self):
        """Load default configuration values."""
        defaults = {
            'llm': {
                'provider': 'google',
                'model': 'gemini-2.5-pro',
                'query_generator_model': 'gemini-2.5-pro',
                'reflection_model': 'gemini-2.5-pro',
                'answer_model': 'gemini-2.5-pro',
                'temperature': 1.0,
                'max_retries': 2,
            },
            'paths': {
                'data_dir': str(Path.home() / 'phmga_data'),
                'save_dir': str(Path.home() / 'phmga_save'),
                'cache_dir': str(Path.home() / 'phmga_cache'),
            },
            'processing': {
                'min_depth': 4,
                'max_depth': 8,
                'min_width': 4,
                'max_steps': 20,
                'default_fs': 1000.0,
            },
            'system': {
                'langchain_tracing': False,
                'langchain_endpoint': '',
                'langchain_api_key': '',
                'langchain_project': '',
            },
            'debug': {
                'fake_llm': False,
                'debug_mode': False,
                'log_level': 'INFO',
            }
        }

        # Merge with environment variables (env takes precedence)
        for category, config in defaults.items():
            if category not in self._config_data:
                self._config_data[category] = {}

            for key, default_value in config.items():
                env_value = self._env_data.get(category, {}).get(key)
                self._config_data[category][key] = env_value if env_value is not None else default_value

    def get(self, path: str, default: Any = None, warn_deprecated: bool = True) -> Any:
        """
        Get configuration value using dot notation path.

        Parameters
        ----------
        path : str
            Dot-separated path (e.g., 'llm.model', 'processing.max_depth')
        default : Any
            Default value if path not found
        warn_deprecated : bool
            Whether to warn about deprecated access patterns

        Returns
        -------
        Any
            Configuration value
        """
        parts = path.split('.')

        # Check for deprecated patterns and warn
        if warn_deprecated and self._is_deprecated_pattern(path):
            self._warn_deprecated_access(path)

        # Navigate through nested structure
        current = self._config_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, path: str, value: Any, category: str = 'state') -> None:
        """
        Set configuration value using dot notation path.

        Parameters
        ----------
        path : str
            Dot-separated path
        value : Any
            Value to set
        category : str
            Category for the value ('config', 'env', 'state')
        """
        parts = path.split('.')

        # Choose target dictionary
        if category == 'config':
            target = self._config_data
        elif category == 'env':
            target = self._env_data
        else:
            target = self._state_data

        # Navigate and create nested structure
        current = target
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _is_deprecated_pattern(self, path: str) -> bool:
        """Check if access pattern is deprecated."""
        deprecated_patterns = [
            'phm_model',  # Use llm.model instead
            'query_generator_model',  # Use llm.query_generator_model
            'reflection_model',  # Use llm.reflection_model
            'answer_model',  # Use llm.answer_model
        ]
        return any(pattern in path for pattern in deprecated_patterns)

    def _warn_deprecated_access(self, path: str):
        """Issue deprecation warning for old access patterns."""
        self._deprecated_access[path] = self._deprecated_access.get(path, 0) + 1

        # Only warn on first access to avoid spam
        if self._deprecated_access[path] == 1:
            migration_map = {
                'phm_model': 'llm.model',
                'query_generator_model': 'llm.query_generator_model',
                'reflection_model': 'llm.reflection_model',
                'answer_model': 'llm.answer_model',
            }

            new_path = migration_map.get(path, f"unified_state.get('{path}')")
            warnings.warn(
                f"Accessing '{path}' is deprecated. Use '{new_path}' instead. "
                f"This will be removed in a future version.",
                DeprecationWarning,
                stacklevel=3
            )

    def get_llm_config(self) -> Dict[str, Any]:
        """Get complete LLM configuration."""
        return self._config_data.get('llm', {})

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self._config_data.get('processing', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config_data.get('paths', {})

    def update_from_yaml(self, yaml_path: str) -> None:
        """Update configuration from YAML file."""
        import yaml

        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Merge YAML config into state data
        self._merge_config(yaml_config, self._state_data)

    def _merge_config(self, source: Dict[str, Any], target: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_config(value, target[key])
            else:
                target[key] = value

    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration for debugging."""
        return {
            'config': self._config_data,
            'environment': self._env_data,
            'state': self._state_data,
            'deprecated_access': dict(self._deprecated_access)
        }

    def validate_required_config(self) -> List[str]:
        """Validate that required configuration is present."""
        errors = []

        # Check required LLM configuration
        llm_config = self.get_llm_config()
        if not llm_config.get('gemini_api_key') and not llm_config.get('openai_api_key'):
            errors.append("No LLM API key found. Set GEMINI_API_KEY or OPENAI_API_KEY")

        # Check required paths exist
        paths_config = self.get_paths_config()
        for path_name, path_value in paths_config.items():
            if path_name.endswith('_dir'):
                path_obj = Path(path_value)
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Cannot create directory {path_value}: {e}")

        return errors


# Global unified state manager instance
_unified_state_manager = None


def get_unified_state() -> UnifiedStateManager:
    """Get the global unified state manager instance."""
    global _unified_state_manager
    if _unified_state_manager is None:
        _unified_state_manager = UnifiedStateManager()
    return _unified_state_manager


def reset_unified_state():
    """Reset the global unified state manager (for testing)."""
    global _unified_state_manager
    _unified_state_manager = None

Shape = Tuple[int, ...]  # 支持多维形状

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

    def write_png(self, path: str) -> None:
        """Render the DAG to a PNG image on disk."""
        try:
            dot = self.to_dot()
            base = path[:-4] if path.endswith(".png") else path
            dot.render(filename=base, format="png", cleanup=True)
        except Exception:
            fname = path if path.endswith(".png") else f"{path}.png"
            with open(fname, "wb") as f:
                f.write(b"")

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
    """
    Central state for the PHM LangGraph pipeline with unified state management.

    This class integrates with the UnifiedStateManager to provide centralized
    configuration and state management while maintaining backward compatibility.
    """

    # Core identification
    case_name: str = Field(default="", description="Unique case identifier")
    user_instruction: str = Field(default="", description="User's instruction for the PHM analysis")

    # Signal data
    reference_signal: InputData
    test_signal: InputData
    dag_state: DAGState

    # Processing constraints (with unified state integration)
    min_depth: int = Field(default_factory=lambda: get_unified_state().get('processing.min_depth', 4))
    min_width: int = Field(default_factory=lambda: get_unified_state().get('processing.min_width', 4))
    max_depth: int = Field(default_factory=lambda: get_unified_state().get('processing.max_depth', 8))
    fs: float | None = Field(default=None, description="Sampling frequency of the signals in Hz")

    # Planning and execution
    detailed_plan: List[dict] = Field(default_factory=list, description="Structured processing plan")
    error_logs: List[str] = Field(default_factory=list, description="System-wide error tracking")
    needs_revision: bool = Field(default=False, description="Whether the analysis needs revision")

    # Reflection and iteration
    reflection_history: List[str] = Field(default_factory=list, description="Feedback from reflect agent")
    is_sufficient: bool = Field(default=False, description="Whether analysis is complete")
    iteration_count: int = Field(default=0, description="Current iteration number")

    # Legacy processed signals (for backward compatibility)
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

    # Analysis results
    insights: List[AnalysisInsight] = Field(default_factory=list, description="Structured analysis insights")
    # datasets: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="ML-ready datasets")
    ml_results: Dict[str, Any] = Field(default_factory=dict, description="Machine learning results")
    final_report: str = Field(default="", description="Generated analysis report")
    final_decision: str = Field(default="", description="Final analysis decision")

    # Legacy fields (for backward compatibility)
    dataset_path: str | None = Field(default=None, description="Path to saved dataset")
    model_path: str | None = Field(default=None, description="Path to saved model")
    accuracy: Optional[float] = Field(default=None, description="Overall accuracy score")

    # Private attributes
    _tracker_instance: Optional[Any] = PrivateAttr(default=None)
    _unified_state: Optional[UnifiedStateManager] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        self._unified_state = get_unified_state()

        # Apply configuration overrides from unified state
        self._apply_unified_config()

    def _apply_unified_config(self):
        """Apply configuration from unified state manager."""
        # Update processing constraints if not explicitly set
        if self.fs is None:
            self.fs = self._unified_state.get('processing.default_fs', 1000.0)

    @property
    def unified_state(self) -> UnifiedStateManager:
        """Access to unified state manager."""
        if self._unified_state is None:
            self._unified_state = get_unified_state()
        return self._unified_state

    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value from unified state.

        Parameters
        ----------
        path : str
            Dot-separated configuration path
        default : Any
            Default value if not found

        Returns
        -------
        Any
            Configuration value
        """
        return self.unified_state.get(path, default)

    def set_config(self, path: str, value: Any) -> None:
        """
        Set configuration value in unified state.

        Parameters
        ----------
        path : str
            Dot-separated configuration path
        value : Any
            Value to set
        """
        self.unified_state.set(path, value, category='state')

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration from unified state."""
        return self.unified_state.get_llm_config()

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration from unified state."""
        return self.unified_state.get_processing_config()

    def get_data_paths(self) -> Dict[str, str]:
        """Get data paths configuration from unified state."""
        return self.unified_state.get_paths_config()

    def validate_configuration(self) -> List[str]:
        """Validate that required configuration is present."""
        return self.unified_state.validate_required_config()

    def update_from_yaml(self, yaml_path: str) -> None:
        """Update state configuration from YAML file."""
        self.unified_state.update_from_yaml(yaml_path)
        self._apply_unified_config()

    # Backward compatibility methods with deprecation warnings
    @property
    def phm_model(self) -> str:
        """Deprecated: Use get_config('llm.model') instead."""
        warnings.warn(
            "phm_model is deprecated. Use get_config('llm.model') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.unified_state.get('llm.model', 'gemini-2.5-pro')

    @property
    def query_generator_model(self) -> str:
        """Deprecated: Use get_config('llm.query_generator_model') instead."""
        warnings.warn(
            "query_generator_model is deprecated. Use get_config('llm.query_generator_model') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.unified_state.get('llm.query_generator_model', 'gemini-2.5-pro')

    def tracker(self) -> "DAGTracker":
        if self._tracker_instance is None:
            self._tracker_instance = DAGTracker(self.dag_state)
        return self._tracker_instance

    class Config:
        arbitrary_types_allowed = True


