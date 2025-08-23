"""
Refactored Data Analyst Agent for PHM research workflows.

This agent performs exploratory data analysis, signal quality assessment,
and feature space exploration using the enhanced service-oriented architecture.
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
    AnalysisResult,
    SignalQualityAssessment,
    StatisticalSummary,
    FeatureSpaceAnalysis,
    calculate_signal_quality_metrics,
    calculate_statistical_summary,
    monitor_performance,
    circuit_breaker
)
from .services import service_registry
from ..states.research_states import ResearchPHMState, ResearchHypothesis

logger = logging.getLogger(__name__)


class DataAnalystAgent(ResearchAgentBase):
    """
    Refactored Data Analyst Agent for comprehensive signal analysis and exploration.

    Enhanced with:
    - Service-oriented architecture for better modularity
    - Improved error handling and performance monitoring
    - Configurable feature extraction and analysis methods
    - Parallel processing capabilities
    - Caching for improved performance

    Responsibilities:
    - Signal quality assessment and anomaly detection
    - Statistical characterization of signal properties
    - Feature space exploration using dimensionality reduction
    - Data preprocessing recommendations
    - Uncertainty quantification for measurement quality
    """

    def __init__(self,
                 agent_name: str = "data_analyst",
                 config: Optional[Dict[str, Any]] = None,
                 validators: Optional[Dict[str, Any]] = None,
                 dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(agent_name, config, validators, dependencies)

        # Initialize services
        self.feature_service = service_registry.get("feature_extraction")
        self.stats_service = service_registry.get("statistical_analysis")
        self.cache_service = service_registry.get("cache")

        # Configuration
        self.enable_advanced_features = self.get_config_value("enable_advanced_features", True)
        self.enable_pca_analysis = self.get_config_value("enable_pca_analysis", True)
        self.enable_clustering = self.get_config_value("enable_clustering", True)
        self.max_features = self.get_config_value("max_features", None)
        self.quick_mode = self.get_config_value("quick_mode", False)
        self.use_parallel = self.get_config_value("use_parallel", True)
        self.use_cache = self.get_config_value("use_cache", True)

        # Feature selection based on configuration
        self.selected_features = self._select_features()

    def _select_features(self) -> List[str]:
        """Select features based on configuration."""
        available_features = self.feature_service.get_available_features()

        if self.quick_mode:
            # Quick mode: use only essential features
            essential_features = ["mean", "std", "rms", "kurtosis", "crest_factor"]
            selected = [f for f in essential_features if f in available_features]
        else:
            # Full mode: use all or configured features
            selected = available_features

        # Apply max_features limit
        if self.max_features and len(selected) > self.max_features:
            selected = selected[:self.max_features]

        return selected
    
    def _perform_analysis(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform the actual comprehensive data analysis implementation.

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
            "feature_extraction_summary": {},
            "performance_metrics": {}
        }

        # Get signal data
        ref_data = self._extract_signal_data(state.reference_signal, "ref")
        test_data = self._extract_signal_data(state.test_signal, "test")

        if ref_data is None and test_data is None:
            raise ValueError("No valid signal data found")

        # Analyze reference signals
        if ref_data is not None:
            analysis_results["reference_analysis"] = self._analyze_signal_set(
                ref_data, "reference", state.fs or 1000.0
            )

        # Analyze test signals
        if test_data is not None:
            analysis_results["test_analysis"] = self._analyze_signal_set(
                test_data, "test", state.fs or 1000.0
            )

        # Comparative analysis
        if ref_data is not None and test_data is not None:
            analysis_results["comparative_analysis"] = self._perform_comparative_analysis(
                ref_data, test_data
            )

        # Generate preprocessing recommendations
        analysis_results["preprocessing_recommendations"] = self._generate_preprocessing_recommendations(
            analysis_results
        )

        # Add performance metrics
        analysis_results["performance_metrics"] = self.get_performance_metrics()

        return analysis_results

    def _extract_signal_data(self, signal_container: Any, key: str) -> Optional[np.ndarray]:
        """Safely extract signal data from container."""
        if signal_container is None:
            return None

        if hasattr(signal_container, 'results') and signal_container.results:
            data = signal_container.results.get(key)
            if data is not None and isinstance(data, np.ndarray) and data.size > 0:
                return data

        return None
    
    @monitor_performance
    def _analyze_signal_set(self, signals: np.ndarray, signal_type: str, fs: float) -> Dict[str, Any]:
        """
        Analyze a set of signals using the enhanced service architecture.

        Args:
            signals: Signal array with shape (n_signals, length, channels)
            signal_type: Type identifier ("reference" or "test")
            fs: Sampling frequency

        Returns:
            Analysis results for the signal set
        """
        logger.debug(f"Analyzing {signal_type} signals with shape {signals.shape}")

        results = {
            "signal_quality": {},
            "statistical_summary": {},
            "feature_analysis": {},
            "frequency_analysis": {},
            "anomaly_detection": {},
            "signal_metadata": {
                "type": signal_type,
                "shape": signals.shape,
                "sampling_frequency": fs
            }
        }

        # Ensure proper signal shape
        signals = self._normalize_signal_shape(signals)

        # Signal quality assessment
        results["signal_quality"] = self._assess_signal_quality(signals, fs)

        # Statistical analysis using service
        results["statistical_summary"] = self._perform_statistical_analysis(signals)

        # Feature extraction using service
        results["feature_analysis"] = self._extract_and_analyze_features(signals)

        # Frequency domain analysis
        if "fft" in self.selected_features:
            results["frequency_analysis"] = self._analyze_frequency_domain(signals, fs)

        # Anomaly detection
        results["anomaly_detection"] = self._detect_signal_anomalies(signals)

        return results

    def _normalize_signal_shape(self, signals: np.ndarray) -> np.ndarray:
        """Normalize signal shape to (n_signals, length, channels)."""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1, 1)
        elif signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        elif signals.ndim > 3:
            # Flatten extra dimensions
            signals = signals.reshape(signals.shape[0], signals.shape[1], -1)

        return signals

    def _assess_signal_quality(self, signals: np.ndarray, fs: float) -> Dict[str, Any]:
        """Assess signal quality for all signals."""
        n_signals, length, n_channels = signals.shape
        quality_assessments = []

        for i in range(n_signals):
            for ch in range(n_channels):
                signal = signals[i, :, ch]
                quality = calculate_signal_quality_metrics(signal, fs)
                quality_assessments.append(quality)

        return self._aggregate_quality_assessments(quality_assessments)

    def _perform_statistical_analysis(self, signals: np.ndarray) -> Dict[str, Any]:
        """Perform statistical analysis using the statistical service."""
        try:
            # Use statistical analysis service
            stats_results = self.stats_service.analyze(signals, methods=["basic_stats", "distribution_analysis", "outlier_detection"])

            # Add signal-specific statistics
            stats_results["signal_specific"] = self._calculate_signal_specific_stats(signals)

            return stats_results
        except Exception as e:
            logger.warning(f"Statistical analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_signal_specific_stats(self, signals: np.ndarray) -> Dict[str, Any]:
        """Calculate signal-specific statistical measures."""
        # Calculate per-signal statistics
        signal_stats = []
        for i in range(signals.shape[0]):
            signal = signals[i, :, :]
            signal_flat = signal.reshape(-1)

            stats = calculate_statistical_summary(signal_flat)
            signal_stats.append({
                "rms": stats.rms,
                "crest_factor": stats.crest_factor,
                "kurtosis": stats.kurtosis,
                "skewness": stats.skewness,
                "entropy": stats.entropy
            })

        # Aggregate across signals
        if signal_stats:
            aggregated = {}
            for key in signal_stats[0].keys():
                values = [s[key] for s in signal_stats]
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values
                }
            return aggregated

        return {}
    
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
    
    @monitor_performance
    def _extract_and_analyze_features(self, signals: np.ndarray) -> Dict[str, Any]:
        """Extract and analyze features using the enhanced feature service."""
        try:
            # Extract features using the service
            features = self.feature_service.extract_features(
                signals,
                feature_names=self.selected_features,
                use_cache=self.use_cache,
                use_parallel=self.use_parallel
            )

            # Analyze extracted features
            feature_analysis = {}
            for name, feature_values in features.items():
                if feature_values is not None:
                    feature_analysis[name] = self._analyze_feature_values(feature_values, name)

            # Perform feature space analysis if enabled
            if self.enable_advanced_features:
                feature_analysis["feature_space"] = self._perform_feature_space_analysis(features)

            return {
                "extracted_features": feature_analysis,
                "feature_count": len(features),
                "successful_extractions": len([f for f in features.values() if f is not None])
            }

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {"error": str(e)}

    def _analyze_feature_values(self, feature_values: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """Analyze individual feature values."""
        try:
            # Flatten feature values for analysis
            if feature_values.ndim > 1:
                flat_values = feature_values.reshape(feature_values.shape[0], -1)
            else:
                flat_values = feature_values.reshape(-1, 1)

            # Calculate statistics
            analysis = {
                "shape": feature_values.shape,
                "mean": float(np.mean(flat_values)),
                "std": float(np.std(flat_values)),
                "min": float(np.min(flat_values)),
                "max": float(np.max(flat_values)),
                "range": float(np.ptp(flat_values))
            }

            # Add feature-specific analysis
            if feature_name in ["kurtosis", "skewness"]:
                analysis["distribution_info"] = {
                    "is_normal": abs(analysis["mean"]) < 0.5,  # Rough normality check
                    "outlier_threshold": analysis["mean"] + 3 * analysis["std"]
                }
            elif feature_name in ["rms", "crest_factor"]:
                analysis["signal_health"] = {
                    "health_indicator": "good" if analysis["mean"] < 3.0 else "concerning",
                    "variability": "low" if analysis["std"] < 0.5 else "high"
                }

            return analysis

        except Exception as e:
            logger.warning(f"Feature analysis failed for {feature_name}: {e}")
            return {"error": str(e)}
    
    def _perform_feature_space_analysis(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform dimensionality reduction and clustering analysis on extracted features."""
        try:
            # Prepare feature matrix
            feature_matrix = []
            feature_names = []

            for name, feature_values in features.items():
                if feature_values is not None and name != "fft":  # Skip FFT for PCA
                    if feature_values.ndim > 1:
                        flat_features = feature_values.reshape(feature_values.shape[0], -1)
                    else:
                        flat_features = feature_values.reshape(-1, 1)

                    feature_matrix.append(flat_features)
                    feature_names.extend([f"{name}_{i}" for i in range(flat_features.shape[1])])

            if not feature_matrix:
                return {"error": "No valid features for space analysis"}

            # Combine features
            combined_features = np.column_stack(feature_matrix)

            if combined_features.shape[0] < 2:
                return {"error": "Insufficient samples for space analysis"}

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(combined_features)

            analysis_results = {}

            # PCA analysis if enabled
            if self.enable_pca_analysis:
                analysis_results["pca"] = self._perform_pca_analysis(features_scaled, feature_names)

            # Clustering analysis if enabled
            if self.enable_clustering:
                analysis_results["clustering"] = self._perform_clustering_analysis(features_scaled)

            # Feature correlation analysis
            analysis_results["correlation"] = self._analyze_feature_correlations(combined_features, feature_names)

            return analysis_results

        except Exception as e:
            logger.error(f"Feature space analysis failed: {e}")
            return {"error": str(e)}

    def _perform_pca_analysis(self, features_scaled: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Perform PCA analysis on scaled features."""
        try:
            pca = PCA()
            pca_features = pca.fit_transform(features_scaled)

            # Determine optimal number of components
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
            n_components_99 = np.argmax(cumsum_variance >= 0.99) + 1

            # Feature importance from first principal component
            feature_importance = {}
            if len(feature_names) == len(pca.components_[0]):
                feature_importance = dict(zip(feature_names, np.abs(pca.components_[0])))

            return {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumsum_variance.tolist(),
                "n_components_95": int(n_components_95),
                "n_components_99": int(n_components_99),
                "feature_importance": feature_importance,
                "total_variance_explained": float(np.sum(pca.explained_variance_ratio_))
            }

        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            return {"error": str(e)}

    def _perform_clustering_analysis(self, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on scaled features."""
        try:
            n_samples = features_scaled.shape[0]
            if n_samples < 2:
                return {"error": "Insufficient samples for clustering"}

            # Determine optimal number of clusters
            max_clusters = min(n_samples, 5)
            cluster_results = {}

            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)

                # Calculate silhouette score if possible
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(features_scaled, cluster_labels)
                except ImportError:
                    silhouette = 0.0

                cluster_results[f"k_{n_clusters}"] = {
                    "labels": cluster_labels.tolist(),
                    "silhouette_score": float(silhouette),
                    "inertia": float(kmeans.inertia_)
                }

            # Find best clustering
            best_k = 2
            best_score = -1
            for k, result in cluster_results.items():
                if result["silhouette_score"] > best_score:
                    best_score = result["silhouette_score"]
                    best_k = int(k.split("_")[1])

            return {
                "cluster_results": cluster_results,
                "best_k": best_k,
                "best_silhouette_score": best_score
            }

        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_feature_correlations(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze correlations between features."""
        try:
            if features.shape[1] < 2:
                return {"error": "Need at least 2 features for correlation analysis"}

            correlation_matrix = np.corrcoef(features.T)

            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_value = correlation_matrix[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            "feature1": feature_names[i],
                            "feature2": feature_names[j],
                            "correlation": float(corr_value)
                        })

            return {
                "correlation_matrix": correlation_matrix.tolist(),
                "feature_names": feature_names,
                "high_correlations": high_corr_pairs,
                "max_correlation": float(np.max(np.abs(correlation_matrix - np.eye(len(feature_names))))),
                "mean_correlation": float(np.mean(np.abs(correlation_matrix - np.eye(len(feature_names)))))
            }

        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
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
    
    def _detect_signal_anomalies(self, signals: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in the signal set using multiple methods."""
        try:
            anomaly_results = {}

            # Energy-based anomaly detection
            anomaly_results["energy_based"] = self._detect_energy_anomalies(signals)

            # Statistical anomaly detection
            anomaly_results["statistical"] = self._detect_statistical_anomalies(signals)

            # Shape-based anomaly detection
            if signals.shape[0] > 1:
                anomaly_results["shape_based"] = self._detect_shape_anomalies(signals)

            # Combine results
            anomaly_results["summary"] = self._summarize_anomaly_detection(anomaly_results)

            return anomaly_results

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": str(e)}

    def _detect_energy_anomalies(self, signals: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies based on signal energy."""
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
            "n_outliers": len(outlier_indices),
            "energy_statistics": {
                "mean": float(mean_energy),
                "std": float(std_energy),
                "all_energies": signal_energies.tolist()
            }
        }

    def _detect_statistical_anomalies(self, signals: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies based on statistical properties."""
        try:
            # Calculate statistical features for each signal
            signal_stats = []
            for i in range(signals.shape[0]):
                signal = signals[i, :, :].reshape(-1)
                stats = calculate_statistical_summary(signal)
                signal_stats.append([stats.rms, stats.kurtosis, stats.crest_factor])

            signal_stats = np.array(signal_stats)

            # Detect outliers in statistical space
            scaler = StandardScaler()
            stats_scaled = scaler.fit_transform(signal_stats)

            # Mahalanobis distance-based detection
            mean_stats = np.mean(stats_scaled, axis=0)
            distances = np.sqrt(np.sum((stats_scaled - mean_stats) ** 2, axis=1))

            threshold = np.mean(distances) + 2 * np.std(distances)
            outlier_indices = np.where(distances > threshold)[0]

            return {
                "outlier_indices": outlier_indices.tolist(),
                "distances": distances.tolist(),
                "threshold": float(threshold),
                "n_outliers": len(outlier_indices)
            }

        except Exception as e:
            logger.warning(f"Statistical anomaly detection failed: {e}")
            return {"error": str(e)}

    def _detect_shape_anomalies(self, signals: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies based on signal shape similarity."""
        try:
            # Calculate pairwise correlations
            n_signals = signals.shape[0]
            correlations = np.zeros((n_signals, n_signals))

            for i in range(n_signals):
                for j in range(i, n_signals):
                    signal_i = signals[i, :, :].reshape(-1)
                    signal_j = signals[j, :, :].reshape(-1)

                    # Normalize signals
                    signal_i = (signal_i - np.mean(signal_i)) / (np.std(signal_i) + 1e-10)
                    signal_j = (signal_j - np.mean(signal_j)) / (np.std(signal_j) + 1e-10)

                    corr = np.corrcoef(signal_i, signal_j)[0, 1]
                    correlations[i, j] = correlations[j, i] = corr if not np.isnan(corr) else 0

            # Find signals with low average correlation to others
            avg_correlations = np.mean(correlations, axis=1)
            threshold = np.mean(avg_correlations) - 2 * np.std(avg_correlations)
            outlier_indices = np.where(avg_correlations < threshold)[0]

            return {
                "outlier_indices": outlier_indices.tolist(),
                "avg_correlations": avg_correlations.tolist(),
                "threshold": float(threshold),
                "n_outliers": len(outlier_indices)
            }

        except Exception as e:
            logger.warning(f"Shape anomaly detection failed: {e}")
            return {"error": str(e)}

    def _summarize_anomaly_detection(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize anomaly detection results across methods."""
        all_outliers = set()
        method_counts = {}

        for method, results in anomaly_results.items():
            if isinstance(results, dict) and "outlier_indices" in results:
                outliers = set(results["outlier_indices"])
                all_outliers.update(outliers)
                method_counts[method] = len(outliers)

        # Find consensus outliers (detected by multiple methods)
        consensus_outliers = set()
        for outlier in all_outliers:
            detection_count = sum(1 for method, results in anomaly_results.items()
                                if isinstance(results, dict) and
                                "outlier_indices" in results and
                                outlier in results["outlier_indices"])
            if detection_count >= 2:  # Detected by at least 2 methods
                consensus_outliers.add(outlier)

        return {
            "total_unique_outliers": len(all_outliers),
            "consensus_outliers": list(consensus_outliers),
            "method_counts": method_counts,
            "all_outliers": list(all_outliers)
        }
    
    def _perform_comparative_analysis(self, ref_signals: np.ndarray, test_signals: np.ndarray) -> Dict[str, Any]:
        """Enhanced comparative analysis between reference and test signals."""
        try:
            comparative_results = {}

            # Basic statistical comparison
            comparative_results["statistical_comparison"] = self._compare_signal_statistics(ref_signals, test_signals)

            # Feature-based comparison
            comparative_results["feature_comparison"] = self._compare_signal_features(ref_signals, test_signals)

            # Distribution comparison
            comparative_results["distribution_comparison"] = self._compare_signal_distributions(ref_signals, test_signals)

            # Overall assessment
            comparative_results["overall_assessment"] = self._assess_signal_differences(comparative_results)

            return comparative_results

        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return {"error": str(e)}

    def _compare_signal_statistics(self, ref_signals: np.ndarray, test_signals: np.ndarray) -> Dict[str, Any]:
        """Compare basic statistical properties."""
        ref_stats = {
            "mean_energy": float(np.mean(ref_signals ** 2)),
            "mean_amplitude": float(np.mean(np.abs(ref_signals))),
            "std_amplitude": float(np.std(ref_signals)),
            "rms": float(np.sqrt(np.mean(ref_signals ** 2))),
            "peak_to_peak": float(np.ptp(ref_signals))
        }

        test_stats = {
            "mean_energy": float(np.mean(test_signals ** 2)),
            "mean_amplitude": float(np.mean(np.abs(test_signals))),
            "std_amplitude": float(np.std(test_signals)),
            "rms": float(np.sqrt(np.mean(test_signals ** 2))),
            "peak_to_peak": float(np.ptp(test_signals))
        }

        # Calculate ratios and differences
        ratios = {}
        differences = {}
        for key in ref_stats.keys():
            ref_val = ref_stats[key]
            test_val = test_stats[key]
            ratios[key] = test_val / (ref_val + 1e-10)
            differences[key] = abs(test_val - ref_val) / (ref_val + 1e-10)

        return {
            "reference_stats": ref_stats,
            "test_stats": test_stats,
            "ratios": ratios,
            "relative_differences": differences,
            "significant_changes": {k: v > 0.2 for k, v in differences.items()}
        }

    def _compare_signal_features(self, ref_signals: np.ndarray, test_signals: np.ndarray) -> Dict[str, Any]:
        """Compare extracted features between signal sets."""
        try:
            # Extract features for both signal sets
            ref_features = self.feature_service.extract_features(
                ref_signals, self.selected_features, use_cache=self.use_cache
            )
            test_features = self.feature_service.extract_features(
                test_signals, self.selected_features, use_cache=self.use_cache
            )

            feature_comparisons = {}

            for feature_name in self.selected_features:
                if feature_name in ref_features and feature_name in test_features:
                    ref_feat = ref_features[feature_name]
                    test_feat = test_features[feature_name]

                    if ref_feat is not None and test_feat is not None:
                        feature_comparisons[feature_name] = self._compare_feature_arrays(ref_feat, test_feat)

            return {
                "feature_comparisons": feature_comparisons,
                "compared_features": list(feature_comparisons.keys()),
                "comparison_count": len(feature_comparisons)
            }

        except Exception as e:
            logger.warning(f"Feature comparison failed: {e}")
            return {"error": str(e)}

    def _compare_feature_arrays(self, ref_array: np.ndarray, test_array: np.ndarray) -> Dict[str, Any]:
        """Compare two feature arrays."""
        # Flatten arrays for comparison
        ref_flat = ref_array.reshape(-1)
        test_flat = test_array.reshape(-1)

        ref_mean = np.mean(ref_flat)
        test_mean = np.mean(test_flat)

        return {
            "ref_mean": float(ref_mean),
            "test_mean": float(test_mean),
            "ratio": float(test_mean / (ref_mean + 1e-10)),
            "relative_difference": float(abs(test_mean - ref_mean) / (ref_mean + 1e-10)),
            "ref_std": float(np.std(ref_flat)),
            "test_std": float(np.std(test_flat))
        }

    def _compare_signal_distributions(self, ref_signals: np.ndarray, test_signals: np.ndarray) -> Dict[str, Any]:
        """Compare signal distributions using statistical tests."""
        try:
            ref_flat = ref_signals.reshape(-1)
            test_flat = test_signals.reshape(-1)

            # Remove invalid values
            ref_valid = ref_flat[np.isfinite(ref_flat)]
            test_valid = test_flat[np.isfinite(test_flat)]

            if len(ref_valid) == 0 or len(test_valid) == 0:
                return {"error": "No valid data for distribution comparison"}

            # Basic distribution comparison
            ref_hist, bin_edges = np.histogram(ref_valid, bins=50, density=True)
            test_hist, _ = np.histogram(test_valid, bins=bin_edges, density=True)

            # Calculate histogram difference
            hist_difference = np.sum(np.abs(ref_hist - test_hist))

            return {
                "histogram_difference": float(hist_difference),
                "ref_range": float(np.ptp(ref_valid)),
                "test_range": float(np.ptp(test_valid)),
                "ref_skewness": float(self._calculate_skewness(ref_valid)),
                "test_skewness": float(self._calculate_skewness(test_valid)),
                "ref_kurtosis": float(self._calculate_kurtosis(ref_valid)),
                "test_kurtosis": float(self._calculate_kurtosis(test_valid))
            }

        except Exception as e:
            logger.warning(f"Distribution comparison failed: {e}")
            return {"error": str(e)}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _assess_signal_differences(self, comparative_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall differences between signal sets."""
        assessment = {
            "overall_similarity": 0.0,
            "significant_differences": [],
            "confidence": 0.0
        }

        try:
            # Analyze statistical comparison
            stat_comp = comparative_results.get("statistical_comparison", {})
            if "significant_changes" in stat_comp:
                significant_stats = [k for k, v in stat_comp["significant_changes"].items() if v]
                assessment["significant_differences"].extend(significant_stats)

            # Analyze feature comparison
            feat_comp = comparative_results.get("feature_comparison", {})
            if "feature_comparisons" in feat_comp:
                for feat_name, feat_data in feat_comp["feature_comparisons"].items():
                    if feat_data.get("relative_difference", 0) > 0.3:
                        assessment["significant_differences"].append(f"feature_{feat_name}")

            # Calculate overall similarity score
            if stat_comp.get("relative_differences"):
                avg_difference = np.mean(list(stat_comp["relative_differences"].values()))
                assessment["overall_similarity"] = max(0.0, 1.0 - avg_difference)

            # Calculate confidence based on available data
            data_quality_factors = [
                len(comparative_results.get("statistical_comparison", {})) > 0,
                len(comparative_results.get("feature_comparison", {})) > 0,
                len(comparative_results.get("distribution_comparison", {})) > 0
            ]
            assessment["confidence"] = sum(data_quality_factors) / len(data_quality_factors)

        except Exception as e:
            logger.warning(f"Assessment calculation failed: {e}")
            assessment["error"] = str(e)

        return assessment
    
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
    
    def generate_hypotheses(self, state: ResearchPHMState, analysis_result: AnalysisResult) -> List[ResearchHypothesis]:
        """Generate research hypotheses based on enhanced data analysis results."""
        hypotheses = []
        analysis_results = analysis_result.results

        # Hypothesis about signal quality
        hypotheses.extend(self._generate_signal_quality_hypotheses(analysis_results))

        # Hypothesis about feature separability
        hypotheses.extend(self._generate_separability_hypotheses(analysis_results))

        # Hypothesis about anomalies
        hypotheses.extend(self._generate_anomaly_hypotheses(analysis_results))

        # Hypothesis about feature space structure
        hypotheses.extend(self._generate_feature_space_hypotheses(analysis_results))

        # Hypothesis about preprocessing needs
        hypotheses.extend(self._generate_preprocessing_hypotheses_from_analysis(analysis_results))

        return hypotheses

    def _generate_signal_quality_hypotheses(self, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses about signal quality."""
        hypotheses = []

        ref_analysis = analysis_results.get("reference_analysis", {})
        signal_quality = ref_analysis.get("signal_quality", {})

        if signal_quality:
            mean_snr = signal_quality.get("mean_snr", 0)
            quality_grades = signal_quality.get("quality_grades", [])

            if mean_snr > 15:
                hypotheses.append(ResearchHypothesis(
                    hypothesis_id=f"hyp_quality_high_{len(hypotheses)}",
                    statement="High signal quality enables reliable fault detection with standard methods",
                    confidence=min(mean_snr / 20.0, 1.0),
                    generated_by=self.agent_name,
                    evidence=[f"Mean SNR: {mean_snr:.1f} dB", f"Quality grades: {quality_grades}"],
                    test_methods=["signal_quality_validation", "noise_analysis"]
                ))
            elif mean_snr < 10:
                hypotheses.append(ResearchHypothesis(
                    hypothesis_id=f"hyp_quality_low_{len(hypotheses)}",
                    statement="Low signal quality requires advanced preprocessing and robust algorithms",
                    confidence=max(0.3, 1.0 - mean_snr / 10.0),
                    generated_by=self.agent_name,
                    evidence=[f"Mean SNR: {mean_snr:.1f} dB", f"Quality grades: {quality_grades}"],
                    test_methods=["preprocessing_validation", "denoising_analysis"]
                ))

        return hypotheses

    def _generate_separability_hypotheses(self, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses about signal separability."""
        hypotheses = []

        comparative_analysis = analysis_results.get("comparative_analysis", {})
        overall_assessment = comparative_analysis.get("overall_assessment", {})

        if overall_assessment:
            similarity = overall_assessment.get("overall_similarity", 0.5)
            significant_diffs = overall_assessment.get("significant_differences", [])

            if similarity < 0.7 and len(significant_diffs) > 0:
                hypotheses.append(ResearchHypothesis(
                    hypothesis_id=f"hyp_separability_{len(hypotheses)}",
                    statement="Significant differences detected between reference and test signals enable classification",
                    confidence=1.0 - similarity,
                    generated_by=self.agent_name,
                    evidence=[f"Similarity score: {similarity:.2f}", f"Significant differences: {len(significant_diffs)}"],
                    test_methods=["statistical_significance_test", "classification_validation"]
                ))
            elif similarity > 0.9:
                hypotheses.append(ResearchHypothesis(
                    hypothesis_id=f"hyp_similarity_{len(hypotheses)}",
                    statement="High similarity between signal sets suggests minimal fault progression",
                    confidence=similarity,
                    generated_by=self.agent_name,
                    evidence=[f"Similarity score: {similarity:.2f}"],
                    test_methods=["similarity_validation", "trend_analysis"]
                ))

        return hypotheses

    def _generate_anomaly_hypotheses(self, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses about detected anomalies."""
        hypotheses = []

        for signal_type in ["reference_analysis", "test_analysis"]:
            signal_analysis = analysis_results.get(signal_type, {})
            anomaly_detection = signal_analysis.get("anomaly_detection", {})

            if "summary" in anomaly_detection:
                summary = anomaly_detection["summary"]
                consensus_outliers = summary.get("consensus_outliers", [])
                total_outliers = summary.get("total_unique_outliers", 0)

                if len(consensus_outliers) > 0:
                    signal_name = signal_type.replace("_analysis", "")
                    hypotheses.append(ResearchHypothesis(
                        hypothesis_id=f"hyp_anomalies_{signal_name}_{len(hypotheses)}",
                        statement=f"Consensus anomalies in {signal_name} signals indicate potential measurement issues or fault conditions",
                        confidence=min(len(consensus_outliers) / 3.0, 1.0),
                        generated_by=self.agent_name,
                        evidence=[f"Consensus outliers: {len(consensus_outliers)}", f"Total outliers: {total_outliers}"],
                        test_methods=["outlier_validation", "anomaly_characterization"]
                    ))

        return hypotheses

    def _generate_feature_space_hypotheses(self, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses about feature space structure."""
        hypotheses = []

        for signal_type in ["reference_analysis", "test_analysis"]:
            signal_analysis = analysis_results.get(signal_type, {})
            feature_analysis = signal_analysis.get("feature_analysis", {})

            if "extracted_features" in feature_analysis and "feature_space" in feature_analysis["extracted_features"]:
                feature_space = feature_analysis["extracted_features"]["feature_space"]

                # PCA-based hypothesis
                if "pca" in feature_space:
                    pca_results = feature_space["pca"]
                    n_components_95 = pca_results.get("n_components_95", 0)
                    total_features = len(pca_results.get("feature_importance", {}))

                    if n_components_95 < total_features * 0.5:
                        hypotheses.append(ResearchHypothesis(
                            hypothesis_id=f"hyp_dimensionality_{signal_type}_{len(hypotheses)}",
                            statement=f"Feature space in {signal_type.replace('_analysis', '')} signals shows strong dimensionality reduction potential",
                            confidence=1.0 - (n_components_95 / total_features) if total_features > 0 else 0.5,
                            generated_by=self.agent_name,
                            evidence=[f"95% variance in {n_components_95}/{total_features} components"],
                            test_methods=["dimensionality_validation", "feature_selection"]
                        ))

                # Clustering-based hypothesis
                if "clustering" in feature_space:
                    clustering_results = feature_space["clustering"]
                    best_silhouette = clustering_results.get("best_silhouette_score", 0)

                    if best_silhouette > 0.5:
                        hypotheses.append(ResearchHypothesis(
                            hypothesis_id=f"hyp_clustering_{signal_type}_{len(hypotheses)}",
                            statement=f"Clear clustering structure in {signal_type.replace('_analysis', '')} feature space suggests distinct signal patterns",
                            confidence=best_silhouette,
                            generated_by=self.agent_name,
                            evidence=[f"Best silhouette score: {best_silhouette:.2f}"],
                            test_methods=["cluster_validation", "pattern_analysis"]
                        ))

        return hypotheses

    def _generate_preprocessing_hypotheses_from_analysis(self, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses about preprocessing needs."""
        hypotheses = []

        preprocessing_recs = analysis_results.get("preprocessing_recommendations", [])

        if len(preprocessing_recs) > 0:
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_preprocessing_{len(hypotheses)}",
                statement="Signal preprocessing is recommended to improve analysis quality",
                confidence=min(len(preprocessing_recs) / 3.0, 1.0),
                generated_by=self.agent_name,
                evidence=[f"Recommendations: {preprocessing_recs}"],
                test_methods=["preprocessing_validation", "quality_improvement_test"]
            ))

        return hypotheses

    def _analyze_frequency_domain(self, signals: np.ndarray, fs: float) -> Dict[str, Any]:
        """Analyze frequency domain characteristics of signals."""
        try:
            # Extract FFT features if available
            if "fft" in self.selected_features:
                fft_features = self.feature_service.extract_features(signals, ["fft"], use_cache=self.use_cache)
                fft_data = fft_features.get("fft")

                if fft_data is not None:
                    # Analyze frequency content
                    freq_analysis = {
                        "dominant_frequencies": self._find_dominant_frequencies(fft_data, fs),
                        "spectral_centroid": self._calculate_spectral_centroid(fft_data, fs),
                        "spectral_bandwidth": self._calculate_spectral_bandwidth(fft_data, fs),
                        "spectral_rolloff": self._calculate_spectral_rolloff(fft_data, fs)
                    }
                    return freq_analysis

            return {"info": "FFT analysis not enabled or failed"}

        except Exception as e:
            logger.warning(f"Frequency domain analysis failed: {e}")
            return {"error": str(e)}

    def _find_dominant_frequencies(self, fft_data: np.ndarray, fs: float) -> List[float]:
        """Find dominant frequencies in the FFT data."""
        # Simple peak finding in frequency domain
        freq_bins = np.fft.fftfreq(fft_data.shape[-1], 1/fs)
        magnitude = np.abs(fft_data)

        # Find peaks (simplified)
        dominant_freqs = []
        for i in range(magnitude.shape[0]):  # For each signal
            signal_mag = magnitude[i, :, 0] if magnitude.ndim > 2 else magnitude[i, :]
            # Find top 3 peaks
            peak_indices = np.argsort(signal_mag)[-3:]
            peak_freqs = [abs(freq_bins[idx]) for idx in peak_indices if abs(freq_bins[idx]) > 0]
            dominant_freqs.extend(peak_freqs)

        return sorted(list(set(dominant_freqs)))[:5]  # Return top 5 unique frequencies

    def _calculate_spectral_centroid(self, fft_data: np.ndarray, fs: float) -> float:
        """Calculate spectral centroid."""
        freq_bins = np.fft.fftfreq(fft_data.shape[-1], 1/fs)
        magnitude = np.abs(fft_data)

        # Calculate weighted average frequency
        numerator = np.sum(freq_bins * np.mean(magnitude, axis=(0, 2)))
        denominator = np.sum(np.mean(magnitude, axis=(0, 2)))

        return float(numerator / (denominator + 1e-10))

    def _calculate_spectral_bandwidth(self, fft_data: np.ndarray, fs: float) -> float:
        """Calculate spectral bandwidth."""
        freq_bins = np.fft.fftfreq(fft_data.shape[-1], 1/fs)
        magnitude = np.abs(fft_data)

        centroid = self._calculate_spectral_centroid(fft_data, fs)

        # Calculate bandwidth as weighted standard deviation
        numerator = np.sum((freq_bins - centroid) ** 2 * np.mean(magnitude, axis=(0, 2)))
        denominator = np.sum(np.mean(magnitude, axis=(0, 2)))

        return float(np.sqrt(numerator / (denominator + 1e-10)))

    def _calculate_spectral_rolloff(self, fft_data: np.ndarray, fs: float, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        magnitude = np.abs(fft_data)
        freq_bins = np.fft.fftfreq(fft_data.shape[-1], 1/fs)

        # Calculate cumulative energy
        energy = np.mean(magnitude ** 2, axis=(0, 2))
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]

        # Find rolloff frequency
        rolloff_threshold = rolloff_percent * total_energy
        rolloff_index = np.where(cumulative_energy >= rolloff_threshold)[0]

        if len(rolloff_index) > 0:
            return float(abs(freq_bins[rolloff_index[0]]))
        else:
            return float(fs / 2)  # Nyquist frequency as fallback


def data_analyst_agent(state: ResearchPHMState) -> Dict[str, Any]:
    """
    Enhanced LangGraph node function for the Data Analyst Agent.

    Args:
        state: Current research state

    Returns:
        State updates from data analysis
    """
    # Create agent with configuration from state if available
    config = getattr(state, 'agent_configs', {}).get('data_analyst', {})
    agent = DataAnalystAgent(config=config)

    # Perform analysis
    analysis_result = agent.analyze(state)

    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_result)

    # Update state
    state_updates = {
        "data_analysis_state": analysis_result.results,
        "research_hypotheses": state.research_hypotheses + hypotheses
    }

    # Add audit entry with enhanced information
    state.add_audit_entry(
        agent="data_analyst",
        action="comprehensive_analysis",
        confidence=analysis_result.confidence,
        outputs={
            "n_hypotheses": len(hypotheses),
            "execution_time": analysis_result.execution_time,
            "memory_usage": analysis_result.memory_usage,
            "success": analysis_result.is_successful(),
            "performance_summary": agent.get_performance_metrics()
        }
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
    
    # Test the enhanced agent
    agent = DataAnalystAgent(config={"quick_mode": False, "enable_advanced_features": True})

    # Perform analysis
    analysis_result = agent.analyze(state)

    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_result)

    # Display results
    print(f"Data analysis completed:")
    print(f"  - Confidence: {analysis_result.confidence:.3f}")
    print(f"  - Execution time: {analysis_result.execution_time:.2f}s")
    print(f"  - Memory usage: {analysis_result.memory_usage:.1f}MB")
    print(f"  - Success: {analysis_result.is_successful()}")
    print(f"  - Generated {len(hypotheses)} hypotheses")

    # Display performance metrics
    perf_metrics = agent.get_performance_metrics()
    if not perf_metrics.get("no_executions"):
        print(f"  - Success rate: {perf_metrics['success_rate']:.1%}")
        print(f"  - Avg execution time: {perf_metrics['avg_execution_time']:.2f}s")

    print("Enhanced Data Analyst Agent test completed successfully!")
