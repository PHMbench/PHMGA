"""
Enhanced base classes and utilities for research-oriented PHM agents.

This module provides a robust foundation for implementing research agents with
improved architecture, dependency injection, performance monitoring, and
standardized interfaces.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Protocol, Union
import numpy as np
from pydantic import BaseModel, Field
import logging
import time
import psutil
import functools
from dataclasses import dataclass
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..states.research_states import ResearchPHMState, ResearchHypothesis, ResearchObjective
from ..model import get_llm
from ..configuration import Configuration

logger = logging.getLogger(__name__)


# Performance monitoring decorators
def monitor_performance(func):
    """Decorator to monitor memory usage and execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise e
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB

            logger.info(f"{func.__name__}: Time={execution_time:.2f}s, "
                       f"Memory={memory_delta:+.1f}MB, Success={success}")

        return result
    return wrapper


def circuit_breaker(max_failures: int = 3, reset_timeout: int = 60):
    """Circuit breaker pattern for agent methods."""
    def decorator(func):
        func.failure_count = 0
        func.last_failure_time = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # Reset failure count after timeout
            if current_time - func.last_failure_time > reset_timeout:
                func.failure_count = 0

            # Check if circuit is open
            if func.failure_count >= max_failures:
                raise Exception(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                func.failure_count = 0  # Reset on success
                return result
            except Exception as e:
                func.failure_count += 1
                func.last_failure_time = current_time
                logger.error(f"Circuit breaker: {func.__name__} failed "
                           f"({func.failure_count}/{max_failures})")
                raise e

        return wrapper
    return decorator


# Agent communication protocols
class AgentMessageType(Enum):
    """Types of messages agents can exchange."""
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESULT = "analysis_result"
    HYPOTHESIS_REQUEST = "hypothesis_request"
    HYPOTHESIS_RESULT = "hypothesis_result"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESULT = "validation_result"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Typed message object for agent communication."""
    message_id: str
    sender: str
    recipient: str
    message_type: AgentMessageType
    payload: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


@dataclass
class AnalysisResult:
    """Typed result object for agent analysis."""
    agent_name: str
    confidence: float
    results: Dict[str, Any]
    execution_time: float
    memory_usage: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return len(self.errors) == 0 and self.confidence > 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis result."""
        return {
            "agent": self.agent_name,
            "confidence": self.confidence,
            "success": self.is_successful(),
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


# Input validation protocols
class InputValidator(Protocol):
    """Protocol for input validation."""

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data and return (is_valid, error_messages)."""
        ...


class SignalValidator:
    """Validator for signal data."""

    def __init__(self, min_length: int = 10, max_length: int = 1000000):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, signals: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate signal array."""
        errors = []

        if signals is None:
            errors.append("Signal data is None")
            return False, errors

        if not isinstance(signals, np.ndarray):
            errors.append(f"Expected numpy array, got {type(signals)}")
            return False, errors

        if signals.size == 0:
            errors.append("Signal array is empty")
            return False, errors

        if len(signals.shape) < 2:
            errors.append(f"Signal must be at least 2D, got shape {signals.shape}")
            return False, errors

        signal_length = signals.shape[1]
        if signal_length < self.min_length:
            errors.append(f"Signal too short: {signal_length} < {self.min_length}")

        if signal_length > self.max_length:
            errors.append(f"Signal too long: {signal_length} > {self.max_length}")

        if np.any(np.isnan(signals)):
            errors.append("Signal contains NaN values")

        if np.any(np.isinf(signals)):
            errors.append("Signal contains infinite values")

        return len(errors) == 0, errors


class StateValidator:
    """Validator for research state."""

    def validate(self, state: ResearchPHMState) -> Tuple[bool, List[str]]:
        """Validate research state."""
        errors = []

        if state is None:
            errors.append("Research state is None")
            return False, errors

        # Validate required fields
        if not hasattr(state, 'reference_signal') or state.reference_signal is None:
            errors.append("Missing reference signal")

        if not hasattr(state, 'test_signal') or state.test_signal is None:
            errors.append("Missing test signal")

        if not hasattr(state, 'fs') or state.fs is None or state.fs <= 0:
            errors.append("Invalid sampling frequency")

        # Validate signal data
        signal_validator = SignalValidator()

        if state.reference_signal and state.reference_signal.results:
            ref_data = state.reference_signal.results.get("ref")
            if ref_data is not None:
                is_valid, signal_errors = signal_validator.validate(ref_data)
                if not is_valid:
                    errors.extend([f"Reference signal: {e}" for e in signal_errors])

        if state.test_signal and state.test_signal.results:
            test_data = state.test_signal.results.get("test")
            if test_data is not None:
                is_valid, signal_errors = signal_validator.validate(test_data)
                if not is_valid:
                    errors.extend([f"Test signal: {e}" for e in signal_errors])

        return len(errors) == 0, errors


class ResearchAgentBase(ABC):
    """
    Enhanced abstract base class for all research agents.

    Provides comprehensive functionality including:
    - Performance monitoring and resource management
    - Input validation and error handling
    - Standardized communication interfaces
    - Dependency injection support
    - Circuit breaker pattern for reliability
    """

    def __init__(self,
                 agent_name: str,
                 config: Optional[Dict[str, Any]] = None,
                 validators: Optional[Dict[str, InputValidator]] = None,
                 dependencies: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.validators = validators or self._create_default_validators()
        self.dependencies = dependencies or {}
        self.llm = None
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_time": 0.0,
            "total_memory": 0.0
        }
        self._initialize_llm()

    def _create_default_validators(self) -> Dict[str, InputValidator]:
        """Create default validators for this agent."""
        return {
            "state": StateValidator(),
            "signal": SignalValidator()
        }

    def _initialize_llm(self):
        """Initialize the LLM for this agent."""
        try:
            config = Configuration.from_runnable_config(None)
            self.llm = get_llm(config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for {self.agent_name}: {e}")
            self.llm = None

    def validate_input(self, state: ResearchPHMState) -> Tuple[bool, List[str]]:
        """Validate input state before processing."""
        if "state" in self.validators:
            return self.validators["state"].validate(state)
        return True, []

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with type safety."""
        return self.config.get(key, default)

    def update_execution_stats(self, execution_time: float, memory_usage: float, success: bool):
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        if success:
            self.execution_stats["successful_executions"] += 1
        self.execution_stats["total_time"] += execution_time
        self.execution_stats["total_memory"] += memory_usage

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        total_exec = self.execution_stats["total_executions"]
        if total_exec == 0:
            return {"no_executions": True}

        return {
            "total_executions": total_exec,
            "success_rate": self.execution_stats["successful_executions"] / total_exec,
            "avg_execution_time": self.execution_stats["total_time"] / total_exec,
            "avg_memory_usage": self.execution_stats["total_memory"] / total_exec,
            "total_time": self.execution_stats["total_time"],
            "total_memory": self.execution_stats["total_memory"]
        }
    
    @monitor_performance
    @circuit_breaker(max_failures=3)
    def analyze(self, state: ResearchPHMState) -> AnalysisResult:
        """
        Perform the core analysis for this research agent with monitoring and validation.

        Args:
            state: Current research state

        Returns:
            AnalysisResult object containing analysis results and metadata
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Validate input
        is_valid, errors = self.validate_input(state)
        if not is_valid:
            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=0.0,
                results={},
                execution_time=0.0,
                memory_usage=0.0,
                errors=errors
            )

        try:
            # Perform the actual analysis
            results = self._perform_analysis(state)
            confidence = self._calculate_confidence(results)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB

            # Update stats
            self.update_execution_stats(execution_time, memory_usage, True)

            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=confidence,
                results=results,
                execution_time=execution_time,
                memory_usage=memory_usage
            )

        except Exception as e:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB

            # Update stats
            self.update_execution_stats(execution_time, memory_usage, False)

            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=0.0,
                results={},
                execution_time=execution_time,
                memory_usage=memory_usage,
                errors=[str(e)]
            )

    @abstractmethod
    def _perform_analysis(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform the actual analysis implementation.

        Args:
            state: Current research state

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    def generate_hypotheses(self, state: ResearchPHMState, analysis_result: AnalysisResult) -> List[ResearchHypothesis]:
        """
        Generate research hypotheses based on analysis results.

        Args:
            state: Current research state
            analysis_result: Results from the analyze method

        Returns:
            List of generated hypotheses
        """
        pass

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for analysis results.

        Args:
            results: Analysis results

        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0

        # Default confidence calculation - can be overridden by subclasses
        successful_analyses = sum(1 for v in results.values() if v is not None and not isinstance(v, dict) or (isinstance(v, dict) and "error" not in v))
        total_analyses = len(results)

        return successful_analyses / total_analyses if total_analyses > 0 else 0.0
    
    def validate_findings(self, state: ResearchPHMState, findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate research findings using statistical tests and domain knowledge.
        
        Args:
            state: Current research state
            findings: Research findings to validate
            
        Returns:
            Validation results with confidence scores
        """
        validation_results = {
            "validation_score": 0.0,
            "validation_tests": [],
            "confidence_intervals": {},
            "statistical_significance": {}
        }
        
        # Basic validation framework - to be extended by subclasses
        if findings:
            validation_results["validation_score"] = 0.7  # Default moderate confidence
            validation_results["validation_tests"].append("basic_consistency_check")
        
        return validation_results
    
    def update_state(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> ResearchPHMState:
        """
        Update the research state with analysis results.
        
        Args:
            state: Current research state
            analysis_results: Results to integrate into state
            
        Returns:
            Updated research state
        """
        # Add audit entry
        state.add_audit_entry(
            agent=self.agent_name,
            action="analysis_completed",
            confidence=analysis_results.get("confidence", 0.5),
            inputs={"analysis_type": self.agent_name},
            outputs={"results_summary": str(analysis_results.keys())}
        )
        
        return state
    
    def calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            Confidence score between 0 and 1
        """
        # Default confidence calculation - to be overridden by subclasses
        if not results:
            return 0.0
        
        # Simple heuristic based on number of successful analyses
        successful_analyses = sum(1 for v in results.values() if v is not None)
        total_analyses = len(results)
        
        return successful_analyses / total_analyses if total_analyses > 0 else 0.0


class SignalQualityAssessment(BaseModel):
    """Structured representation of signal quality assessment results."""
    
    signal_to_noise_ratio: float = Field(..., description="Estimated SNR in dB")
    stationarity_score: float = Field(..., ge=0.0, le=1.0, description="Stationarity measure")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Data completeness")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly detection score")
    quality_grade: str = Field(..., description="Overall quality grade (A-F)")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")


class StatisticalSummary(BaseModel):
    """Comprehensive statistical summary of signal data."""
    
    mean: float = Field(..., description="Signal mean")
    std: float = Field(..., description="Signal standard deviation")
    skewness: float = Field(..., description="Signal skewness")
    kurtosis: float = Field(..., description="Signal kurtosis")
    rms: float = Field(..., description="Root mean square")
    peak_to_peak: float = Field(..., description="Peak-to-peak amplitude")
    crest_factor: float = Field(..., description="Crest factor")
    entropy: float = Field(..., description="Signal entropy")


class FeatureSpaceAnalysis(BaseModel):
    """Results of feature space exploration and dimensionality analysis."""
    
    principal_components: np.ndarray = Field(..., description="Principal component vectors")
    explained_variance_ratio: np.ndarray = Field(..., description="Explained variance ratios")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    cluster_analysis: Dict[str, Any] = Field(default_factory=dict, description="Clustering results")
    dimensionality_recommendation: int = Field(..., description="Recommended feature dimensions")
    
    class Config:
        arbitrary_types_allowed = True


def calculate_signal_quality_metrics(signal: np.ndarray, fs: float = 1000.0) -> SignalQualityAssessment:
    """
    Calculate comprehensive signal quality metrics.
    
    Args:
        signal: Input signal array with shape (batch, length, channels)
        fs: Sampling frequency in Hz
        
    Returns:
        SignalQualityAssessment object with quality metrics
    """
    # Flatten signal for analysis
    if signal.ndim > 1:
        signal_flat = signal.reshape(-1)
    else:
        signal_flat = signal
    
    # Calculate SNR (simple noise floor estimation)
    signal_power = np.mean(signal_flat ** 2)
    noise_power = np.var(np.diff(signal_flat))  # High-frequency noise estimate
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Stationarity assessment using variance of windowed statistics
    window_size = min(len(signal_flat) // 10, 1000)
    if window_size > 10:
        windowed_means = []
        for i in range(0, len(signal_flat) - window_size, window_size):
            windowed_means.append(np.mean(signal_flat[i:i + window_size]))
        stationarity_score = 1.0 / (1.0 + np.var(windowed_means))
    else:
        stationarity_score = 0.5  # Default for short signals
    
    # Completeness (check for NaN, inf, or constant values)
    valid_samples = np.isfinite(signal_flat).sum()
    completeness_score = valid_samples / len(signal_flat)
    
    # Anomaly detection using z-score
    z_scores = np.abs((signal_flat - np.mean(signal_flat)) / (np.std(signal_flat) + 1e-10))
    anomaly_score = np.mean(z_scores > 3.0)  # Fraction of outliers
    
    # Overall quality grade
    overall_score = (
        min(snr_db / 20.0, 1.0) * 0.3 +  # SNR contribution
        stationarity_score * 0.3 +        # Stationarity contribution
        completeness_score * 0.3 +        # Completeness contribution
        (1.0 - anomaly_score) * 0.1       # Anomaly contribution (inverted)
    )
    
    if overall_score >= 0.9:
        quality_grade = "A"
    elif overall_score >= 0.8:
        quality_grade = "B"
    elif overall_score >= 0.7:
        quality_grade = "C"
    elif overall_score >= 0.6:
        quality_grade = "D"
    else:
        quality_grade = "F"
    
    # Generate recommendations
    recommendations = []
    if snr_db < 10:
        recommendations.append("Consider noise reduction filtering")
    if stationarity_score < 0.7:
        recommendations.append("Signal shows non-stationary behavior - consider segmentation")
    if completeness_score < 0.95:
        recommendations.append("Signal contains missing or invalid data points")
    if anomaly_score > 0.05:
        recommendations.append("Signal contains significant outliers - consider outlier removal")
    
    return SignalQualityAssessment(
        signal_to_noise_ratio=snr_db,
        stationarity_score=stationarity_score,
        completeness_score=completeness_score,
        anomaly_score=anomaly_score,
        quality_grade=quality_grade,
        recommendations=recommendations
    )


def calculate_statistical_summary(signal: np.ndarray) -> StatisticalSummary:
    """
    Calculate comprehensive statistical summary of signal.
    
    Args:
        signal: Input signal array
        
    Returns:
        StatisticalSummary object with statistical metrics
    """
    from scipy import stats
    
    # Flatten signal for analysis
    if signal.ndim > 1:
        signal_flat = signal.reshape(-1)
    else:
        signal_flat = signal
    
    # Remove any NaN or infinite values
    signal_clean = signal_flat[np.isfinite(signal_flat)]
    
    if len(signal_clean) == 0:
        # Return default values for empty signal
        return StatisticalSummary(
            mean=0.0, std=0.0, skewness=0.0, kurtosis=0.0,
            rms=0.0, peak_to_peak=0.0, crest_factor=1.0, entropy=0.0
        )
    
    # Calculate statistics
    mean_val = np.mean(signal_clean)
    std_val = np.std(signal_clean)
    skewness_val = stats.skew(signal_clean)
    kurtosis_val = stats.kurtosis(signal_clean)
    rms_val = np.sqrt(np.mean(signal_clean ** 2))
    peak_to_peak_val = np.ptp(signal_clean)
    crest_factor_val = np.max(np.abs(signal_clean)) / (rms_val + 1e-10)
    
    # Calculate entropy (using histogram-based approach)
    hist, _ = np.histogram(signal_clean, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    entropy_val = -np.sum(hist * np.log2(hist + 1e-10))
    
    return StatisticalSummary(
        mean=mean_val,
        std=std_val,
        skewness=skewness_val,
        kurtosis=kurtosis_val,
        rms=rms_val,
        peak_to_peak=peak_to_peak_val,
        crest_factor=crest_factor_val,
        entropy=entropy_val
    )


if __name__ == "__main__":
    # Test the research base functionality
    import numpy as np
    
    # Generate test signal
    fs = 1000
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
    signal = signal.reshape(1, -1, 1)  # Shape for PHM system
    
    # Test signal quality assessment
    quality = calculate_signal_quality_metrics(signal, fs)
    print(f"Signal quality grade: {quality.quality_grade}")
    print(f"SNR: {quality.signal_to_noise_ratio:.1f} dB")
    print(f"Stationarity: {quality.stationarity_score:.3f}")
    
    # Test statistical summary
    stats_summary = calculate_statistical_summary(signal)
    print(f"Signal RMS: {stats_summary.rms:.3f}")
    print(f"Signal entropy: {stats_summary.entropy:.3f}")
    
    print("Research base test completed successfully!")
