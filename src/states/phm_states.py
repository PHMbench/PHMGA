from __future__ import annotations

from typing import List, Dict, Any, Annotated, Tuple, Optional, TypedDict, Union
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, validator
import uuid
import os
import warnings
import yaml
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


# PHMConfig is no longer needed since configuration is embedded directly in PHMState


# Deprecated: UnifiedStateManager is no longer used in PHMState
# Kept for backward compatibility only
class UnifiedStateManager(BaseModel):
    """
    Deprecated: This class is kept for backward compatibility only.

    PHMState now manages configuration directly without this intermediate layer.
    """

    def __init__(self, **data):
        super().__init__(**data)
        warnings.warn(
            "UnifiedStateManager is deprecated. Use PHMState directly.",
            DeprecationWarning,
            stacklevel=2
        )

    def get(self, path: str, default: Any = None) -> Any:
        """Deprecated: Use PHMState fields directly."""
        return default

    def set(self, path: str, value: Any, category: str = 'config') -> None:
        """Deprecated: Use PHMState fields directly."""
        pass

    def get_llm_config(self) -> Dict[str, Any]:
        """Deprecated: Use PHMState.get_llm_config() instead."""
        return {}

    def get_processing_config(self) -> Dict[str, Any]:
        """Deprecated: Use PHMState.get_processing_config() instead."""
        return {}

    def get_paths_config(self) -> Dict[str, Any]:
        """Deprecated: Use PHMState.get_data_paths() instead."""
        return {}

    def validate_required_config(self) -> List[str]:
        """Deprecated: Use PHMState.validate() instead."""
        return []

    def update_from_yaml(self, yaml_path: str) -> None:
        """Deprecated: Use PHMState.load_config() instead."""
        pass


# Global unified state instance
_unified_state_instance: Optional[UnifiedStateManager] = None


def get_unified_state() -> UnifiedStateManager:
    """
    Get the global unified state manager instance.
    
    Returns
    -------
    UnifiedStateManager
        Global unified state manager
    """
    global _unified_state_instance
    if _unified_state_instance is None:
        _unified_state_instance = UnifiedStateManager()
    return _unified_state_instance


def reset_unified_state() -> None:
    """Reset the global unified state manager (for testing)."""
    global _unified_state_instance
    _unified_state_instance = None


# Data structures for PHM processing
Shape = Tuple[int, ...]


class _NodeBase(BaseModel):
    """Base class for all DAG nodes."""
    node_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    parents: List[str] = Field(default_factory=list, description="Parent node IDs")
    stage: Literal["input", "processed", "similarity", "dataset", "output"] = "input"
    shape: Shape
    kind: Literal["signal"] = "signal"
    sim: Dict[str, Any] = Field(default_factory=dict, description="Similarity metrics")


class InputData(_NodeBase):
    """Represents a batch of raw input signals."""
    stage: str = "input"
    data: Dict[str, Any] = Field(default_factory=dict, description="Raw data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    results: Dict[str, Any] = Field(default_factory=dict, description="Processing results")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProcessedData(_NodeBase):
    """Output of a single signal processing method."""
    stage: str = "processed"
    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    results: Any = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class DataSetNode(_NodeBase):
    """Represents a dataset derived from a processed node."""
    stage: str = "dataset"
    meta: Dict[str, Any] = Field(default_factory=dict)


class DAGState(BaseModel):
    """
    Simplified DAG state for signal processing workflow.
    """
    user_instruction: str = Field(default="", description="User's analysis instruction")
    channels: List[str] = Field(default_factory=list, description="Signal channel names")
    nodes: Dict[str, Union[InputData, ProcessedData, DataSetNode]] = Field(
        default_factory=dict, description="All nodes in the DAG"
    )
    leaves: List[str] = Field(default_factory=list, description="Current leaf node IDs")
    error_log: List[str] = Field(default_factory=list, description="Processing errors")

    class Config:
        arbitrary_types_allowed = True


class PHMState(BaseModel):
    """
    Direct PHM state for the LangGraph pipeline.

    This simplified version contains all state parameters directly as fields,
    eliminating the need for complex unified state management while maintaining
    full backward compatibility.
    """

    # Core identification
    case_name: str = Field(default="", description="Unique case identifier")
    user_instruction: str = Field(default="", description="User's instruction for analysis")

    # Signal data (required)
    reference_signal: InputData
    test_signal: InputData
    dag_state: DAGState

    # Configuration parameters (directly embedded)
    # LLM Configuration
    llm_provider: str = Field(default="google", description="LLM provider (google, openai)")
    llm_model: str = Field(default="gemini-2.5-pro", description="LLM model name")
    llm_temperature: float = Field(default=1.0, description="LLM sampling temperature")
    llm_max_retries: int = Field(default=2, description="Maximum LLM API retries")
    # gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    # openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Processing Configuration
    min_depth: int = Field(default=4, description="Minimum DAG depth")
    max_depth: int = Field(default=8, description="Maximum DAG depth")
    min_width: int = Field(default=4, description="Minimum DAG width")
    max_steps: int = Field(default=100, description="Maximum processing steps")
    fs: Optional[float] = Field(default=1000.0, description="Sampling frequency in Hz")

    # Paths Configuration
    data_dir: Optional[str] = Field(default=None, description="Data directory")
    save_dir: Optional[str] = Field(default=None, description="Save directory")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory")

    # Processing state
    detailed_plan: List[dict] = Field(default_factory=list, description="Processing plan")
    error_logs: List[str] = Field(default_factory=list, description="Error tracking")
    needs_revision: bool = Field(default=False, description="Whether analysis needs revision")

    # Reflection and iteration
    reflection_history: List[str] = Field(default_factory=list, description="Reflection feedback")
    is_sufficient: bool = Field(default=False, description="Whether analysis is complete")
    iteration_count: int = Field(default=0, description="Current iteration number")

    # Analysis results
    insights: List[AnalysisInsight] = Field(default_factory=list, description="Analysis insights")
    datasets: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="ML datasets")
    ml_results: Dict[str, Any] = Field(default_factory=dict, description="ML results")
    final_report: str = Field(default="", description="Generated analysis report")
    final_decision: str = Field(default="", description="Final analysis decision")

    # Legacy fields for backward compatibility
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

    # Additional legacy fields
    dataset_path: Optional[str] = Field(default=None, description="Path to saved dataset")
    model_path: Optional[str] = Field(default=None, description="Path to saved model")
    accuracy: Optional[float] = Field(default=None, description="Overall accuracy score")

    # Additional configuration storage for YAML extras
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        # Load environment variables directly
        self._load_environment_variables()

    def _load_environment_variables(self):
        """Load environment variables directly into state fields."""
        # LLM environment variables
        if api_key := os.getenv("GEMINI_API_KEY"):
            self.gemini_api_key = api_key
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = api_key
        if provider := os.getenv("LLM_PROVIDER"):
            self.llm_provider = provider
        if model := os.getenv("LLM_MODEL"):
            self.llm_model = model

        # Processing environment variables
        if max_depth := os.getenv("PHM_MAX_DEPTH"):
            try:
                self.max_depth = int(max_depth)
            except ValueError:
                pass
        if min_depth := os.getenv("PHM_MIN_DEPTH"):
            try:
                self.min_depth = int(min_depth)
            except ValueError:
                pass

        # Paths environment variables
        if data_dir := os.getenv("PHM_DATA_DIR"):
            self.data_dir = data_dir
        if save_dir := os.getenv("PHM_SAVE_DIR"):
            self.save_dir = save_dir
        if cache_dir := os.getenv("PHM_CACHE_DIR"):
            self.cache_dir = cache_dir

    # Factory method for simplified initialization
    @classmethod
    def from_case_config(
        cls,
        config_path: str,
        case_name: str,
        user_instruction: str,
        metadata_path: str,
        h5_path: str,
        ref_ids: List[int],
        test_ids: List[int],
        fs: Optional[float] = None
    ) -> "PHMState":
        """
        Create PHMState from case configuration.

        This factory method loads YAML configuration and signal data to create
        a fully initialized PHMState instance.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        case_name : str
            Case identifier
        user_instruction : str
            User's analysis instruction
        metadata_path : str
            Path to metadata file
        h5_path : str
            Path to HDF5 data file
        ref_ids : List[int]
            Reference signal IDs
        test_ids : List[int]
            Test signal IDs
        fs : Optional[float]
            Sampling frequency

        Returns
        -------
        PHMState
            Initialized PHM state with configuration loaded
        """
        # Load signal data using existing utility
        from ..utils import load_signal_data

        ref_signals, ref_labels = load_signal_data(metadata_path, h5_path, ref_ids)
        test_signals, test_labels = load_signal_data(metadata_path, h5_path, test_ids)

        if not ref_signals or not test_signals:
            raise ValueError("Failed to load reference or test signals.")

        all_labels = {**ref_labels, **test_labels}

        # Determine channels from first signal
        first_sig_array = next(iter(ref_signals.values()))
        num_channels = first_sig_array.shape[2]  # Shape is (B, L, C)
        channel_names = [f"ch{i+1}" for i in range(num_channels)]

        nodes = {}
        leaves = []

        for i, channel_name in enumerate(channel_names):
            # Extract signals for current channel
            channel_ref_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in ref_signals.items()}
            channel_test_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in test_signals.items()}

            first_sig_shape = next(iter(channel_ref_signals.values())).shape

            meta = {
                "channel": channel_name,
                "labels": all_labels,
            }
            if fs is not None:
                meta["fs"] = fs

            node = InputData(
                node_id=channel_name,
                data={},
                results={"ref": channel_ref_signals, "tst": channel_test_signals},
                parents=[],
                shape=first_sig_shape,
                meta=meta,
            )
            nodes[channel_name] = node
            leaves.append(channel_name)

        if not nodes:
            raise ValueError("No valid nodes could be created from the provided data.")

        # Create DAG state
        dag_state = DAGState(
            user_instruction=user_instruction,
            nodes=nodes,
            leaves=leaves,
            channels=channel_names
        )

        # Create PHM state instance
        state = cls(
            case_name=case_name,
            user_instruction=user_instruction,
            reference_signal=next(iter(nodes.values())),
            test_signal=next(iter(nodes.values())),
            dag_state=dag_state,
            fs=fs,
        )

        # Load YAML configuration directly into state
        state.load_config(config_path)

        return state

    def load_config(self, yaml_path: str) -> None:
        """
        Load configuration from YAML file directly into state fields.

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            warnings.warn(f"YAML configuration file not found: {yaml_path}")
            return

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Update LLM configuration
        if 'llm' in yaml_data:
            llm_config = yaml_data['llm']
            if 'provider' in llm_config:
                self.llm_provider = llm_config['provider']
            if 'model' in llm_config:
                self.llm_model = llm_config['model']
            if 'temperature' in llm_config:
                self.llm_temperature = llm_config['temperature']
            if 'max_retries' in llm_config:
                self.llm_max_retries = llm_config['max_retries']
            if 'gemini_api_key' in llm_config:
                self.gemini_api_key = llm_config['gemini_api_key']
            if 'openai_api_key' in llm_config:
                self.openai_api_key = llm_config['openai_api_key']

        # Update processing configuration
        if 'processing' in yaml_data:
            proc_config = yaml_data['processing']
            if 'min_depth' in proc_config:
                self.min_depth = proc_config['min_depth']
            if 'max_depth' in proc_config:
                self.max_depth = proc_config['max_depth']
            if 'min_width' in proc_config:
                self.min_width = proc_config['min_width']
            if 'max_steps' in proc_config:
                self.max_steps = proc_config['max_steps']
            if 'default_fs' in proc_config and self.fs is None:
                self.fs = proc_config['default_fs']

        # Update paths configuration
        if 'paths' in yaml_data:
            paths_config = yaml_data['paths']
            if 'data_dir' in paths_config:
                self.data_dir = paths_config['data_dir']
            if 'save_dir' in paths_config:
                self.save_dir = paths_config['save_dir']
            if 'cache_dir' in paths_config:
                self.cache_dir = paths_config['cache_dir']

        # Store any additional configuration
        for key, value in yaml_data.items():
            if key not in ['llm', 'processing', 'paths']:
                self.extra_config[key] = value

    def validate(self) -> List[str]:
        """
        Validate state configuration.

        Returns
        -------
        List[str]
            List of validation error messages
        """
        errors = []

        # Validate LLM configuration
        if self.llm_provider == "google" and not self.gemini_api_key:
            if not os.getenv("GEMINI_API_KEY"):
                errors.append("Google provider requires GEMINI_API_KEY")

        if self.llm_provider == "openai" and not self.openai_api_key:
            if not os.getenv("OPENAI_API_KEY"):
                errors.append("OpenAI provider requires OPENAI_API_KEY")

        # Validate processing constraints
        if self.min_depth > self.max_depth:
            errors.append("min_depth cannot be greater than max_depth")

        if self.min_depth < 1:
            errors.append("min_depth must be at least 1")

        return errors

    # Backward compatibility methods
    def get_config(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (backward compatibility)."""
        try:
            parts = path.split('.')
            if len(parts) == 2:
                section, key = parts
                if section == 'llm':
                    return getattr(self, f'llm_{key}', default)
                elif section == 'processing':
                    if key == 'default_fs':
                        return self.fs
                    return getattr(self, key, default)
                elif section == 'paths':
                    return getattr(self, f'{key}_dir' if not key.endswith('_dir') else key, default)
            return default
        except (AttributeError, IndexError):
            return default

    def set_config(self, path: str, value: Any) -> None:
        """Set configuration value using dot notation (backward compatibility)."""
        try:
            parts = path.split('.')
            if len(parts) == 2:
                section, key = parts
                if section == 'llm':
                    setattr(self, f'llm_{key}', value)
                elif section == 'processing':
                    if key == 'default_fs':
                        self.fs = value
                    else:
                        setattr(self, key, value)
                elif section == 'paths':
                    attr_name = f'{key}_dir' if not key.endswith('_dir') else key
                    setattr(self, attr_name, value)
        except (AttributeError, IndexError):
            pass

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration (backward compatibility)."""
        return {
            'provider': self.llm_provider,
            'model': self.llm_model,
            'temperature': self.llm_temperature,
            'max_retries': self.llm_max_retries,
            'gemini_api_key': self.gemini_api_key,
            'openai_api_key': self.openai_api_key,
        }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration (backward compatibility)."""
        return {
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'min_width': self.min_width,
            'max_steps': self.max_steps,
            'default_fs': self.fs,
        }

    def get_data_paths(self) -> Dict[str, Any]:
        """Get data paths configuration (backward compatibility)."""
        return {
            'data_dir': self.data_dir,
            'save_dir': self.save_dir,
            'cache_dir': self.cache_dir,
        }

    def update_from_yaml(self, yaml_path: str) -> None:
        """Update from YAML file (backward compatibility)."""
        self.load_config(yaml_path)

    def validate_configuration(self) -> List[str]:
        """Validate configuration (backward compatibility)."""
        return self.validate()

    # Legacy property accessors with deprecation warnings
    @property
    def phm_model(self) -> str:
        """Deprecated: Use llm_model instead."""
        warnings.warn(
            "phm_model is deprecated. Use llm_model instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.llm_model

    @property
    def query_generator_model(self) -> str:
        """Deprecated: Use llm_model instead."""
        warnings.warn(
            "query_generator_model is deprecated. Use llm_model instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.llm_model
