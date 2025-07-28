from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field
import uuid
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator

class SignalData(BaseModel):
    """Represents a batch of raw input signals."""

    # 为信号数据生成一个唯一的ID。
    # 如果在创建实例时未提供，则会自动生成一个默认ID，
    # 格式为 "sig_" 加上一个8位的随机十六进制字符串。
    # 例如: "sig_a1b2c3d4"
    signal_id: str = Field(default_factory=lambda: f"sig_{uuid.uuid4().hex[:8]}")
    data: List[List[List[float]]]
    sampling_rate: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedSignal(BaseModel):
    """Output of a single signal processing method."""

    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    processed_data: Any


class ExtractedFeatures(BaseModel):
    """Feature set extracted from a batch of signals."""

    feature_set_id: str = Field(default_factory=lambda: f"feat_{uuid.uuid4().hex[:8]}")
    source_processed_id: str
    features: List[Dict[str, float]]


class AnalysisInsight(BaseModel):
    """Concrete insight produced by analysis or reflection."""

    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    content: str
    severity_score: float = Field(ge=0.0, le=1.0)
    supporting_feature_ids: List[str]


class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    user_instruction: str
    reference_signal: SignalData
    test_signal: SignalData

    plan: Dict[str, Any]
    reflection_history: List[str]
    is_sufficient: bool
    iteration_count: int

    processed_signals: Annotated[List[ProcessedSignal], operator.add]
    extracted_features: Annotated[List[ExtractedFeatures], operator.add]

    analysis_results: List[AnalysisInsight]
    final_decision: str

    final_report: str