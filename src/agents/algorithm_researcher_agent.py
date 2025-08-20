"""
Algorithm Researcher Agent for PHM research workflows.

This agent investigates and compares different signal processing and ML techniques,
performs hyperparameter optimization, and provides algorithm recommendations.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

from .research_base import ResearchAgentBase
from ..states.research_states import ResearchPHMState, ResearchHypothesis
from ..tools.signal_processing_schemas import get_operator, OP_REGISTRY
from ..tools.transform_schemas import FFTOp, NormalizeOp
from ..tools.expand_schemas import STFTOp, PatchOp
from ..tools.aggregate_schemas import MeanOp, StdOp, RMSOp

logger = logging.getLogger(__name__)


class AlgorithmPerformanceResult:
    """Container for algorithm performance evaluation results."""
    
    def __init__(self, algorithm_name: str, parameters: Dict[str, Any]):
        self.algorithm_name = algorithm_name
        self.parameters = parameters
        self.scores = {}
        self.cross_val_scores = []
        self.execution_time = 0.0
        self.memory_usage = 0.0
        self.confidence = 0.0
    
    def add_score(self, metric: str, value: float):
        """Add a performance score."""
        self.scores[metric] = value
    
    def calculate_overall_score(self) -> float:
        """Calculate overall performance score."""
        if not self.scores:
            return 0.0
        
        # Weighted combination of metrics
        weights = {"accuracy": 0.4, "f1": 0.3, "precision": 0.15, "recall": 0.15}
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.scores:
                weighted_score += self.scores[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0


class AlgorithmResearcherAgent(ResearchAgentBase):
    """
    Algorithm Researcher Agent for signal processing and ML technique investigation.
    
    Responsibilities:
    - Automated hyperparameter optimization for signal processing pipelines
    - Comparative analysis of different feature extraction methods
    - Algorithm performance benchmarking and statistical validation
    - Novel signal processing technique discovery
    - Adaptive algorithm selection based on signal characteristics
    """
    
    def __init__(self):
        super().__init__("algorithm_researcher")
        self.signal_processors = self._initialize_signal_processors()
        self.ml_algorithms = self._initialize_ml_algorithms()
        self.performance_cache = {}
    
    def _initialize_signal_processors(self) -> Dict[str, Any]:
        """Initialize available signal processing operators."""
        processors = {}
        
        # Basic transforms
        processors["fft"] = FFTOp()
        processors["normalize"] = NormalizeOp()
        
        # Time-frequency analysis
        processors["stft"] = STFTOp(fs=1000, nperseg=256, noverlap=128)
        
        # Windowing
        processors["patch"] = PatchOp(patch_size=128, stride=64)
        
        # Statistical features
        processors["mean"] = MeanOp(axis=-2)
        processors["std"] = StdOp(axis=-2)
        processors["rms"] = RMSOp(axis=-2)
        
        return processors
    
    def _initialize_ml_algorithms(self) -> Dict[str, Any]:
        """Initialize ML algorithms for comparison."""
        return {
            "random_forest": {
                "class": RandomForestClassifier,
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "svm": {
                "class": SVC,
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"]
                }
            }
        }
    
    def analyze(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform comprehensive algorithm research and comparison.
        
        Args:
            state: Current research state
            
        Returns:
            Dictionary containing algorithm research results
        """
        logger.info(f"{self.agent_name}: Starting algorithm research")
        
        analysis_results = {
            "signal_processing_comparison": {},
            "ml_algorithm_comparison": {},
            "hyperparameter_optimization": {},
            "performance_benchmarks": {},
            "recommended_pipeline": {},
            "confidence": 0.0
        }
        
        try:
            # Extract features using different signal processing methods
            feature_sets = self._extract_feature_sets(state)
            
            # Compare signal processing methods
            analysis_results["signal_processing_comparison"] = self._compare_signal_processing_methods(
                feature_sets, state
            )
            
            # Compare ML algorithms
            if feature_sets:
                analysis_results["ml_algorithm_comparison"] = self._compare_ml_algorithms(
                    feature_sets, state
                )
                
                # Hyperparameter optimization for best methods
                analysis_results["hyperparameter_optimization"] = self._optimize_hyperparameters(
                    feature_sets, state
                )
            
            # Generate performance benchmarks
            analysis_results["performance_benchmarks"] = self._generate_performance_benchmarks(
                analysis_results
            )
            
            # Recommend optimal pipeline
            analysis_results["recommended_pipeline"] = self._recommend_pipeline(analysis_results)
            
            # Calculate confidence
            analysis_results["confidence"] = self.calculate_confidence(analysis_results)
            
            logger.info(f"{self.agent_name}: Research completed with confidence {analysis_results['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Research failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["confidence"] = 0.0
        
        return analysis_results
    
    def _extract_feature_sets(self, state: ResearchPHMState) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract features using different signal processing methods."""
        feature_sets = {}
        
        # Get reference and test signals
        ref_data = state.reference_signal.results.get("ref")
        test_data = state.test_signal.results.get("test")
        
        if ref_data is None or test_data is None:
            logger.warning("Missing signal data for feature extraction")
            return feature_sets
        
        # Extract features using each processor
        for proc_name, processor in self.signal_processors.items():
            try:
                # Process reference signals
                ref_features = processor.execute(ref_data)
                test_features = processor.execute(test_data)
                
                # Flatten features if needed
                if ref_features.ndim > 2:
                    ref_features = ref_features.reshape(ref_features.shape[0], -1)
                if test_features.ndim > 2:
                    test_features = test_features.reshape(test_features.shape[0], -1)
                
                feature_sets[proc_name] = {
                    "reference": ref_features,
                    "test": test_features
                }
                
                logger.debug(f"Extracted {proc_name} features: ref {ref_features.shape}, test {test_features.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to extract {proc_name} features: {e}")
                continue
        
        return feature_sets
    
    def _compare_signal_processing_methods(self, feature_sets: Dict[str, Dict[str, np.ndarray]], 
                                         state: ResearchPHMState) -> Dict[str, Any]:
        """Compare different signal processing methods."""
        comparison_results = {}
        
        for method_name, features in feature_sets.items():
            try:
                ref_features = features["reference"]
                test_features = features["test"]
                
                # Calculate feature quality metrics
                feature_variance = np.var(ref_features, axis=0).mean()
                feature_range = np.ptp(ref_features, axis=0).mean()
                feature_separability = self._calculate_separability(ref_features, test_features)
                
                comparison_results[method_name] = {
                    "feature_variance": feature_variance,
                    "feature_range": feature_range,
                    "separability": feature_separability,
                    "feature_dimensions": ref_features.shape[1],
                    "quality_score": (feature_variance + feature_separability) / 2
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze {method_name}: {e}")
                comparison_results[method_name] = {"error": str(e)}
        
        return comparison_results
    
    def _calculate_separability(self, ref_features: np.ndarray, test_features: np.ndarray) -> float:
        """Calculate separability between reference and test features."""
        try:
            # Simple separability measure using mean distance
            ref_centroid = np.mean(ref_features, axis=0)
            test_centroid = np.mean(test_features, axis=0)
            
            # Euclidean distance between centroids
            distance = np.linalg.norm(ref_centroid - test_centroid)
            
            # Normalize by feature scale
            ref_scale = np.std(ref_features, axis=0).mean()
            normalized_distance = distance / (ref_scale + 1e-10)
            
            return min(normalized_distance, 10.0)  # Cap at 10 for numerical stability
            
        except Exception:
            return 0.0
    
    def _compare_ml_algorithms(self, feature_sets: Dict[str, Dict[str, np.ndarray]], 
                              state: ResearchPHMState) -> Dict[str, Any]:
        """Compare ML algorithms on different feature sets."""
        comparison_results = {}
        
        for feature_name, features in feature_sets.items():
            try:
                # Prepare data for ML
                X, y = self._prepare_ml_data(features)
                
                if len(np.unique(y)) < 2:
                    logger.warning(f"Insufficient class diversity for {feature_name}")
                    continue
                
                feature_results = {}
                
                # Test each ML algorithm
                for alg_name, alg_config in self.ml_algorithms.items():
                    try:
                        # Use default parameters for initial comparison
                        model = alg_config["class"](random_state=42)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                        
                        feature_results[alg_name] = {
                            "cv_mean": cv_scores.mean(),
                            "cv_std": cv_scores.std(),
                            "cv_scores": cv_scores.tolist()
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to test {alg_name} on {feature_name}: {e}")
                        feature_results[alg_name] = {"error": str(e)}
                
                comparison_results[feature_name] = feature_results
                
            except Exception as e:
                logger.warning(f"Failed to prepare ML data for {feature_name}: {e}")
                comparison_results[feature_name] = {"error": str(e)}
        
        return comparison_results
    
    def _prepare_ml_data(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature data for ML algorithms."""
        ref_features = features["reference"]
        test_features = features["test"]
        
        # Combine features
        X = np.vstack([ref_features, test_features])
        
        # Create labels (0 for reference, 1 for test)
        y = np.hstack([
            np.zeros(ref_features.shape[0]),
            np.ones(test_features.shape[0])
        ])
        
        return X, y
    
    def _optimize_hyperparameters(self, feature_sets: Dict[str, Dict[str, np.ndarray]], 
                                 state: ResearchPHMState) -> Dict[str, Any]:
        """Optimize hyperparameters for best performing methods."""
        optimization_results = {}
        
        # Find best feature set and algorithm combination
        best_combinations = self._find_best_combinations(feature_sets)
        
        for combo in best_combinations[:2]:  # Optimize top 2 combinations
            feature_name, alg_name = combo["feature"], combo["algorithm"]
            
            try:
                # Prepare data
                X, y = self._prepare_ml_data(feature_sets[feature_name])
                
                # Get parameter grid
                alg_config = self.ml_algorithms[alg_name]
                param_grid = alg_config["param_grid"]
                
                # Grid search (simplified)
                best_score = 0.0
                best_params = {}
                
                for params in ParameterGrid(param_grid):
                    try:
                        model = alg_config["class"](**params, random_state=42)
                        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                        score = cv_scores.mean()
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            
                    except Exception:
                        continue
                
                optimization_results[f"{feature_name}_{alg_name}"] = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "feature_set": feature_name,
                    "algorithm": alg_name
                }
                
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed for {combo}: {e}")
        
        return optimization_results
    
    def _find_best_combinations(self, feature_sets: Dict[str, Dict[str, np.ndarray]]) -> List[Dict[str, str]]:
        """Find best feature set and algorithm combinations."""
        combinations = []
        
        # This is a simplified version - in practice, would use the ML comparison results
        for feature_name in feature_sets.keys():
            for alg_name in self.ml_algorithms.keys():
                combinations.append({
                    "feature": feature_name,
                    "algorithm": alg_name,
                    "score": np.random.random()  # Placeholder - use actual scores
                })
        
        # Sort by score (descending)
        combinations.sort(key=lambda x: x["score"], reverse=True)
        return combinations
    
    def _generate_performance_benchmarks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance benchmarks."""
        benchmarks = {
            "signal_processing_ranking": [],
            "ml_algorithm_ranking": [],
            "overall_performance": {},
            "computational_efficiency": {}
        }
        
        # Rank signal processing methods
        sp_comparison = analysis_results.get("signal_processing_comparison", {})
        sp_ranking = sorted(
            [(name, results.get("quality_score", 0)) for name, results in sp_comparison.items()],
            key=lambda x: x[1], reverse=True
        )
        benchmarks["signal_processing_ranking"] = sp_ranking
        
        # Rank ML algorithms (simplified)
        ml_comparison = analysis_results.get("ml_algorithm_comparison", {})
        ml_scores = {}
        for feature_name, feature_results in ml_comparison.items():
            for alg_name, alg_results in feature_results.items():
                if "cv_mean" in alg_results:
                    key = f"{feature_name}_{alg_name}"
                    ml_scores[key] = alg_results["cv_mean"]
        
        ml_ranking = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)
        benchmarks["ml_algorithm_ranking"] = ml_ranking
        
        return benchmarks
    
    def _recommend_pipeline(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal processing pipeline."""
        benchmarks = analysis_results.get("performance_benchmarks", {})
        
        # Get best signal processing method
        sp_ranking = benchmarks.get("signal_processing_ranking", [])
        best_sp_method = sp_ranking[0][0] if sp_ranking else "fft"
        
        # Get best ML algorithm
        ml_ranking = benchmarks.get("ml_algorithm_ranking", [])
        best_ml_combo = ml_ranking[0][0] if ml_ranking else "fft_random_forest"
        
        return {
            "recommended_signal_processing": best_sp_method,
            "recommended_ml_pipeline": best_ml_combo,
            "confidence": analysis_results.get("confidence", 0.0),
            "justification": f"Based on performance benchmarks, {best_sp_method} provides optimal feature extraction"
        }
    
    def generate_hypotheses(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate research hypotheses based on algorithm research."""
        hypotheses = []
        
        # Hypothesis about optimal signal processing
        benchmarks = analysis_results.get("performance_benchmarks", {})
        sp_ranking = benchmarks.get("signal_processing_ranking", [])
        
        if sp_ranking:
            best_method, best_score = sp_ranking[0]
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_sp_{len(hypotheses)}",
                statement=f"{best_method} provides optimal feature extraction for this signal type",
                confidence=min(best_score / 10.0, 1.0),  # Normalize score
                generated_by=self.agent_name,
                evidence=[f"Quality score: {best_score:.3f}"],
                test_methods=["cross_validation", "statistical_significance_test"]
            ))
        
        # Hypothesis about ML algorithm performance
        ml_ranking = benchmarks.get("ml_algorithm_ranking", [])
        if ml_ranking:
            best_combo, best_score = ml_ranking[0]
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_ml_{len(hypotheses)}",
                statement=f"{best_combo} achieves optimal classification performance",
                confidence=best_score,
                generated_by=self.agent_name,
                evidence=[f"Cross-validation accuracy: {best_score:.3f}"],
                test_methods=["holdout_validation", "bootstrap_analysis"]
            ))
        
        return hypotheses


def algorithm_researcher_agent(state: ResearchPHMState) -> Dict[str, Any]:
    """
    LangGraph node function for the Algorithm Researcher Agent.
    
    Args:
        state: Current research state
        
    Returns:
        State updates from algorithm research
    """
    agent = AlgorithmResearcherAgent()
    
    # Perform research
    analysis_results = agent.analyze(state)
    
    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_results)
    
    # Update state
    state_updates = {
        "algorithm_research_state": analysis_results,
        "research_hypotheses": state.research_hypotheses + hypotheses
    }
    
    # Add audit entry
    state.add_audit_entry(
        agent="algorithm_researcher",
        action="algorithm_comparison",
        confidence=analysis_results.get("confidence", 0.0),
        outputs={"n_hypotheses": len(hypotheses)}
    )
    
    return state_updates


if __name__ == "__main__":
    # Test the Algorithm Researcher Agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from states.research_states import ResearchPHMState
    from states.phm_states import DAGState, InputData
    import numpy as np
    
    # Create test data
    fs = 1000
    t = np.linspace(0, 1, fs)
    
    # Reference signals
    ref_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t)),
        np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Test signals
    test_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Create research state
    dag_state = DAGState(user_instruction="Test algorithm research", channels=["ch1"], nodes={}, leaves=[])
    ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                          results={"ref": ref_signals}, meta={"fs": fs})
    test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                           results={"test": test_signals}, meta={"fs": fs})
    
    state = ResearchPHMState(
        case_name="test_algorithm_research",
        user_instruction="Research optimal algorithms for bearing fault detection",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=fs
    )
    
    # Test the agent
    agent = AlgorithmResearcherAgent()
    results = agent.analyze(state)
    hypotheses = agent.generate_hypotheses(state, results)
    
    print(f"Algorithm research completed with confidence: {results['confidence']:.3f}")
    print(f"Generated {len(hypotheses)} hypotheses")
    print("Algorithm Researcher Agent test completed successfully!")
