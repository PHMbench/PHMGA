"""
Data Analyst Agent for PHM research workflows.

This agent performs exploratory data analysis, signal quality assessment,
and feature space exploration to provide insights for research planning.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from .research_base import (
    ResearchAgentBase, 
    SignalQualityAssessment, 
    StatisticalSummary,
    FeatureSpaceAnalysis,
    calculate_signal_quality_metrics,
    calculate_statistical_summary
)
from ..states.research_states import ResearchPHMState, ResearchHypothesis
from ..tools.aggregate_schemas import (
    MeanOp, StdOp, RMSOp, KurtosisOp, SkewnessOp, 
    EntropyOp, CrestFactorOp, PeakToPeakOp
)
from ..tools.transform_schemas import FFTOp

logger = logging.getLogger(__name__)


class DataAnalystAgent(ResearchAgentBase):
    """
    Data Analyst Agent for comprehensive signal analysis and exploration.
    
    Responsibilities:
    - Signal quality assessment and anomaly detection
    - Statistical characterization of signal properties
    - Feature space exploration using dimensionality reduction
    - Data preprocessing recommendations
    - Uncertainty quantification for measurement quality
    """
    
    def __init__(self):
        super().__init__("data_analyst")
        self.feature_extractors = self._initialize_feature_extractors()
    
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize signal processing operators for feature extraction."""
        return {
            "mean": MeanOp(axis=-2),
            "std": StdOp(axis=-2),
            "rms": RMSOp(axis=-2),
            "kurtosis": KurtosisOp(axis=-2),
            "skewness": SkewnessOp(axis=-2),
            "entropy": EntropyOp(axis=-2),
            "crest_factor": CrestFactorOp(axis=-2),
            "peak_to_peak": PeakToPeakOp(axis=-2),
            "fft": FFTOp()
        }
    
    def analyze(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis on reference and test signals.
        
        Args:
            state: Current research state containing signals
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"{self.agent_name}: Starting comprehensive data analysis")
        
        analysis_results = {
            "reference_analysis": {},
            "test_analysis": {},
            "comparative_analysis": {},
            "preprocessing_recommendations": [],
            "confidence": 0.0
        }
        
        try:
            # Analyze reference signals
            ref_data = state.reference_signal.results.get("ref")
            if ref_data is not None:
                analysis_results["reference_analysis"] = self._analyze_signal_set(
                    ref_data, "reference", state.fs or 1000.0
                )
            
            # Analyze test signals
            test_data = state.test_signal.results.get("test")
            if test_data is not None:
                analysis_results["test_analysis"] = self._analyze_signal_set(
                    test_data, "test", state.fs or 1000.0
                )
            
            # Comparative analysis
            if ref_data is not None and test_data is not None:
                analysis_results["comparative_analysis"] = self._comparative_analysis(
                    ref_data, test_data
                )
            
            # Generate preprocessing recommendations
            analysis_results["preprocessing_recommendations"] = self._generate_preprocessing_recommendations(
                analysis_results
            )
            
            # Calculate overall confidence
            analysis_results["confidence"] = self.calculate_confidence(analysis_results)
            
            logger.info(f"{self.agent_name}: Analysis completed with confidence {analysis_results['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["confidence"] = 0.0
        
        return analysis_results
    
    def _analyze_signal_set(self, signals: np.ndarray, signal_type: str, fs: float) -> Dict[str, Any]:
        """
        Analyze a set of signals (reference or test).
        
        Args:
            signals: Signal array with shape (n_signals, length, channels)
            signal_type: Type identifier ("reference" or "test")
            fs: Sampling frequency
            
        Returns:
            Analysis results for the signal set
        """
        results = {
            "signal_quality": {},
            "statistical_summary": {},
            "feature_analysis": {},
            "frequency_analysis": {},
            "anomaly_detection": {}
        }
        
        # Ensure proper signal shape
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]  # Add channel dimension
        
        n_signals, length, n_channels = signals.shape
        
        # Signal quality assessment for each signal
        quality_assessments = []
        for i in range(n_signals):
            for ch in range(n_channels):
                signal = signals[i, :, ch]
                quality = calculate_signal_quality_metrics(signal, fs)
                quality_assessments.append(quality)
        
        # Aggregate quality metrics
        results["signal_quality"] = self._aggregate_quality_assessments(quality_assessments)
        
        # Statistical analysis
        results["statistical_summary"] = self._extract_statistical_features(signals)
        
        # Feature space analysis
        results["feature_analysis"] = self._perform_feature_space_analysis(signals)
        
        # Frequency domain analysis
        results["frequency_analysis"] = self._analyze_frequency_domain(signals, fs)
        
        # Anomaly detection
        results["anomaly_detection"] = self._detect_anomalies(signals)
        
        return results
    
    def _aggregate_quality_assessments(self, assessments: List[SignalQualityAssessment]) -> Dict[str, Any]:
        """Aggregate quality assessments across multiple signals."""
        if not assessments:
            return {}
        
        return {
            "mean_snr": np.mean([a.signal_to_noise_ratio for a in assessments]),
            "mean_stationarity": np.mean([a.stationarity_score for a in assessments]),
            "mean_completeness": np.mean([a.completeness_score for a in assessments]),
            "mean_anomaly_score": np.mean([a.anomaly_score for a in assessments]),
            "quality_grades": [a.quality_grade for a in assessments],
            "common_recommendations": self._find_common_recommendations(assessments)
        }
    
    def _find_common_recommendations(self, assessments: List[SignalQualityAssessment]) -> List[str]:
        """Find recommendations that appear frequently across assessments."""
        all_recommendations = []
        for assessment in assessments:
            all_recommendations.extend(assessment.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Return recommendations that appear in >50% of assessments
        threshold = len(assessments) * 0.5
        return [rec for rec, count in recommendation_counts.items() if count >= threshold]
    
    def _extract_statistical_features(self, signals: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive statistical features from signals."""
        features = {}
        
        # Extract features using operators
        for name, extractor in self.feature_extractors.items():
            if name == "fft":
                continue  # Handle separately
            try:
                feature_values = extractor.execute(signals)
                features[name] = {
                    "values": feature_values,
                    "mean": np.mean(feature_values),
                    "std": np.std(feature_values),
                    "range": np.ptp(feature_values)
                }
            except Exception as e:
                logger.warning(f"Failed to extract {name} features: {e}")
        
        return features
    
    def _perform_feature_space_analysis(self, signals: np.ndarray) -> Dict[str, Any]:
        """Perform dimensionality reduction and clustering analysis."""
        try:
            # Extract features for PCA
            feature_matrix = []
            for name, extractor in self.feature_extractors.items():
                if name == "fft":
                    continue
                try:
                    features = extractor.execute(signals)
                    if features.ndim > 1:
                        features = features.reshape(features.shape[0], -1)
                    feature_matrix.append(features)
                except Exception:
                    continue
            
            if not feature_matrix:
                return {"error": "No features extracted for analysis"}
            
            # Combine features
            feature_matrix = np.column_stack(feature_matrix)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            
            # PCA analysis
            pca = PCA()
            pca_features = pca.fit_transform(features_scaled)
            
            # Determine optimal number of components (95% variance)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
            
            # Clustering analysis
            if len(features_scaled) > 1:
                n_clusters = min(len(features_scaled), 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)
            else:
                cluster_labels = np.array([0])
            
            return {
                "pca_components": pca.components_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "n_components_95_variance": n_components_95,
                "cluster_labels": cluster_labels,
                "feature_importance": dict(zip(
                    [f"feature_{i}" for i in range(feature_matrix.shape[1])],
                    np.abs(pca.components_[0])  # First PC loadings as importance
                ))
            }
            
        except Exception as e:
            logger.error(f"Feature space analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_frequency_domain(self, signals: np.ndarray, fs: float) -> Dict[str, Any]:
        """Analyze frequency domain characteristics."""
        try:
            # Apply FFT
            fft_op = self.feature_extractors["fft"]
            fft_results = fft_op.execute(signals)
            
            # Calculate frequency bins
            n_freq_bins = fft_results.shape[-2]
            freqs = np.fft.rfftfreq(signals.shape[-2], 1/fs)
            
            # Find dominant frequencies
            mean_spectrum = np.mean(fft_results, axis=(0, -1))
            dominant_freq_idx = np.argmax(mean_spectrum)
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate spectral centroid
            spectral_centroid = np.sum(freqs * mean_spectrum) / np.sum(mean_spectrum)
            
            # Calculate spectral bandwidth
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * mean_spectrum) / np.sum(mean_spectrum)
            )
            
            return {
                "dominant_frequency": dominant_freq,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "frequency_bins": freqs,
                "mean_spectrum": mean_spectrum
            }
            
        except Exception as e:
            logger.error(f"Frequency domain analysis failed: {e}")
            return {"error": str(e)}
    
    def _detect_anomalies(self, signals: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in the signal set."""
        try:
            # Simple outlier detection using z-scores
            signal_energies = np.mean(signals ** 2, axis=(1, 2))  # Energy per signal
            mean_energy = np.mean(signal_energies)
            std_energy = np.std(signal_energies)
            
            z_scores = np.abs((signal_energies - mean_energy) / (std_energy + 1e-10))
            outlier_threshold = 2.0
            outlier_indices = np.where(z_scores > outlier_threshold)[0]
            
            return {
                "outlier_indices": outlier_indices.tolist(),
                "outlier_scores": z_scores[outlier_indices].tolist(),
                "outlier_threshold": outlier_threshold,
                "n_outliers": len(outlier_indices)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": str(e)}
    
    def _comparative_analysis(self, ref_signals: np.ndarray, test_signals: np.ndarray) -> Dict[str, Any]:
        """Compare reference and test signal characteristics."""
        try:
            # Calculate basic statistics for comparison
            ref_stats = {
                "mean_energy": np.mean(ref_signals ** 2),
                "mean_amplitude": np.mean(np.abs(ref_signals)),
                "std_amplitude": np.std(ref_signals)
            }
            
            test_stats = {
                "mean_energy": np.mean(test_signals ** 2),
                "mean_amplitude": np.mean(np.abs(test_signals)),
                "std_amplitude": np.std(test_signals)
            }
            
            # Calculate relative differences
            energy_ratio = test_stats["mean_energy"] / (ref_stats["mean_energy"] + 1e-10)
            amplitude_ratio = test_stats["mean_amplitude"] / (ref_stats["mean_amplitude"] + 1e-10)
            
            return {
                "reference_stats": ref_stats,
                "test_stats": test_stats,
                "energy_ratio": energy_ratio,
                "amplitude_ratio": amplitude_ratio,
                "significant_difference": abs(energy_ratio - 1.0) > 0.2
            }
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_preprocessing_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate preprocessing recommendations based on analysis results."""
        recommendations = []
        
        # Check reference analysis
        ref_quality = analysis_results.get("reference_analysis", {}).get("signal_quality", {})
        if ref_quality.get("mean_snr", 0) < 10:
            recommendations.append("Apply noise reduction filtering to improve SNR")
        
        if ref_quality.get("mean_stationarity", 1) < 0.7:
            recommendations.append("Consider signal segmentation due to non-stationarity")
        
        # Check comparative analysis
        comp_analysis = analysis_results.get("comparative_analysis", {})
        if comp_analysis.get("significant_difference", False):
            recommendations.append("Normalize signal amplitudes before comparison")
        
        # Check for outliers
        ref_anomalies = analysis_results.get("reference_analysis", {}).get("anomaly_detection", {})
        if ref_anomalies.get("n_outliers", 0) > 0:
            recommendations.append("Remove or investigate outlier signals")
        
        return recommendations
    
    def generate_hypotheses(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate research hypotheses based on data analysis results."""
        hypotheses = []
        
        # Hypothesis based on signal quality
        ref_quality = analysis_results.get("reference_analysis", {}).get("signal_quality", {})
        mean_snr = ref_quality.get("mean_snr", 0)
        
        if mean_snr > 15:
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_quality_{len(hypotheses)}",
                statement="High signal quality enables reliable fault detection with standard methods",
                confidence=0.8,
                generated_by=self.agent_name,
                evidence=[f"Mean SNR: {mean_snr:.1f} dB"],
                test_methods=["cross_validation", "noise_robustness_test"]
            ))
        elif mean_snr < 10:
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_noise_{len(hypotheses)}",
                statement="Low signal quality requires advanced denoising for accurate diagnosis",
                confidence=0.7,
                generated_by=self.agent_name,
                evidence=[f"Mean SNR: {mean_snr:.1f} dB"],
                test_methods=["denoising_comparison", "robustness_analysis"]
            ))
        
        # Hypothesis based on comparative analysis
        comp_analysis = analysis_results.get("comparative_analysis", {})
        if comp_analysis.get("significant_difference", False):
            energy_ratio = comp_analysis.get("energy_ratio", 1.0)
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_difference_{len(hypotheses)}",
                statement="Significant energy differences indicate potential fault conditions",
                confidence=0.6,
                generated_by=self.agent_name,
                evidence=[f"Energy ratio: {energy_ratio:.2f}"],
                test_methods=["statistical_significance_test", "fault_correlation_analysis"]
            ))
        
        return hypotheses


def data_analyst_agent(state: ResearchPHMState) -> Dict[str, Any]:
    """
    LangGraph node function for the Data Analyst Agent.
    
    Args:
        state: Current research state
        
    Returns:
        State updates from data analysis
    """
    agent = DataAnalystAgent()
    
    # Perform analysis
    analysis_results = agent.analyze(state)
    
    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_results)
    
    # Update state
    state_updates = {
        "data_analysis_state": analysis_results,
        "research_hypotheses": state.research_hypotheses + hypotheses
    }
    
    # Add audit entry
    state.add_audit_entry(
        agent="data_analyst",
        action="comprehensive_analysis",
        confidence=analysis_results.get("confidence", 0.0),
        outputs={"n_hypotheses": len(hypotheses)}
    )
    
    return state_updates


if __name__ == "__main__":
    # Test the Data Analyst Agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from states.research_states import ResearchPHMState
    from states.phm_states import DAGState, InputData
    import numpy as np
    
    # Create test data
    fs = 1000
    t = np.linspace(0, 1, fs)
    
    # Reference signals (healthy)
    ref_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t)),
        np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Test signals (with fault)
    test_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Create research state
    dag_state = DAGState(user_instruction="Test data analysis", channels=["ch1"], nodes={}, leaves=[])
    ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                          results={"ref": ref_signals}, meta={"fs": fs})
    test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                           results={"test": test_signals}, meta={"fs": fs})
    
    state = ResearchPHMState(
        case_name="test_data_analysis",
        user_instruction="Analyze bearing signals for research insights",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=fs
    )
    
    # Test the agent
    agent = DataAnalystAgent()
    results = agent.analyze(state)
    hypotheses = agent.generate_hypotheses(state, results)
    
    print(f"Data analysis completed with confidence: {results['confidence']:.3f}")
    print(f"Generated {len(hypotheses)} hypotheses")
    print("Data Analyst Agent test completed successfully!")
