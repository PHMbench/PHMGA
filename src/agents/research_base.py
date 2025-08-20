"""
Base classes and utilities for research-oriented PHM agents.

This module provides the foundation for implementing research agents that can
autonomously investigate signal processing techniques, generate hypotheses,
and validate findings in the PHM domain.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
import logging

from ..states.research_states import ResearchPHMState, ResearchHypothesis, ResearchObjective
from ..model import get_llm
from ..configuration import Configuration

logger = logging.getLogger(__name__)


class ResearchAgentBase(ABC):
    """
    Abstract base class for all research agents.
    
    Provides common functionality for research workflow integration,
    hypothesis generation, and result validation.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM for this agent."""
        try:
            config = Configuration.from_runnable_config(None)
            self.llm = get_llm(config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for {self.agent_name}: {e}")
            self.llm = None
    
    @abstractmethod
    def analyze(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform the core analysis for this research agent.
        
        Args:
            state: Current research state
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def generate_hypotheses(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """
        Generate research hypotheses based on analysis results.
        
        Args:
            state: Current research state
            analysis_results: Results from the analyze method
            
        Returns:
            List of generated hypotheses
        """
        pass
    
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
