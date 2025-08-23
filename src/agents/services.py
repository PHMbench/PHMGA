"""
Service layer for PHM Research Agents.

This module provides decoupled services for signal processing, statistical analysis,
machine learning, and other common operations used by research agents.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple, Protocol
import numpy as np
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import hashlib
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# Service interfaces
class SignalProcessorService(Protocol):
    """Protocol for signal processing services."""
    
    def process_signals(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        """Process signals and return results."""
        ...


class FeatureExtractorService(Protocol):
    """Protocol for feature extraction services."""
    
    def extract_features(self, signals: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Extract specified features from signals."""
        ...


class MLModelService(Protocol):
    """Protocol for machine learning services."""
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str, **kwargs) -> Any:
        """Train a machine learning model."""
        ...
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model."""
        ...


# Caching service
class CacheService:
    """Service for caching computation results."""
    
    def __init__(self, cache_dir: str = "/tmp/phm_cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit."""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > self.max_size_mb:
                # Remove oldest files
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.rglob('*') if f.is_file()]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                removed_size = 0
                for file_path, _ in files:
                    if removed_size >= total_size_mb - self.max_size_mb:
                        break
                    removed_size += file_path.stat().st_size / (1024 * 1024)
                    file_path.unlink()
                    
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_cache_key(self, data: Any, operation: str) -> str:
        """Generate cache key for data and operation."""
        if isinstance(data, np.ndarray):
            data_hash = hashlib.md5(data.tobytes()).hexdigest()
        else:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()
        
        return f"{operation}_{data_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                cache_file.unlink()  # Remove corrupted cache
        return None
    
    def set(self, key: str, value: Any):
        """Set cached result."""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")


# Parallel processing service
class ParallelProcessingService:
    """Service for parallel processing operations."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.use_processes = use_processes
    
    def process_in_parallel(self, func: callable, data_list: List[Any], **kwargs) -> List[Any]:
        """Process data in parallel using threads or processes."""
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, data, **kwargs) for data in data_list]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Parallel processing failed: {e}")
                    results.append(None)
        
        return results
    
    def extract_features_parallel(self, signals: np.ndarray, extractors: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features in parallel."""
        def extract_single_feature(extractor_item):
            name, extractor = extractor_item
            try:
                return name, extractor.execute(signals)
            except Exception as e:
                logger.warning(f"Feature extraction failed for {name}: {e}")
                return name, None
        
        results = self.process_in_parallel(extract_single_feature, list(extractors.items()))
        return {name: result for name, result in results if result is not None}


# Enhanced feature extraction service
class EnhancedFeatureExtractionService:
    """Enhanced service for feature extraction with caching and parallel processing."""
    
    def __init__(self, cache_service: Optional[CacheService] = None, 
                 parallel_service: Optional[ParallelProcessingService] = None):
        self.cache_service = cache_service or CacheService()
        self.parallel_service = parallel_service or ParallelProcessingService()
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize feature extractors."""
        try:
            from ..tools.aggregate_schemas import (
                MeanOp, StdOp, RMSOp, KurtosisOp, SkewnessOp, 
                EntropyOp, CrestFactorOp, PeakToPeakOp
            )
            from ..tools.transform_schemas import FFTOp, NormalizeOp
            from ..tools.expand_schemas import STFTOp, PatchOp
            
            self.extractors = {
                "mean": MeanOp(axis=-2),
                "std": StdOp(axis=-2),
                "rms": RMSOp(axis=-2),
                "kurtosis": KurtosisOp(axis=-2),
                "skewness": SkewnessOp(axis=-2),
                "entropy": EntropyOp(axis=-2),
                "crest_factor": CrestFactorOp(axis=-2),
                "peak_to_peak": PeakToPeakOp(axis=-2),
                "fft": FFTOp(),
                "normalize": NormalizeOp(),
                "stft": STFTOp(fs=1000, nperseg=256, noverlap=128),
                "patch": PatchOp(patch_size=128, stride=64)
            }
        except ImportError as e:
            logger.warning(f"Could not import feature extractors: {e}")
            self.extractors = {}
    
    def extract_features(self, signals: np.ndarray, feature_names: Optional[List[str]] = None,
                        use_cache: bool = True, use_parallel: bool = True) -> Dict[str, np.ndarray]:
        """Extract features with caching and parallel processing."""
        if feature_names is None:
            feature_names = list(self.extractors.keys())
        
        # Filter available extractors
        available_extractors = {name: self.extractors[name] for name in feature_names 
                              if name in self.extractors}
        
        if not available_extractors:
            logger.warning("No valid feature extractors found")
            return {}
        
        # Check cache first
        if use_cache:
            cache_key = self.cache_service.get_cache_key(signals, f"features_{'_'.join(sorted(feature_names))}")
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached features for {len(feature_names)} extractors")
                return cached_result
        
        # Extract features
        if use_parallel and len(available_extractors) > 1:
            results = self.parallel_service.extract_features_parallel(signals, available_extractors)
        else:
            results = {}
            for name, extractor in available_extractors.items():
                try:
                    results[name] = extractor.execute(signals)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {name}: {e}")
        
        # Cache results
        if use_cache and results:
            self.cache_service.set(cache_key, results)
        
        return results
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature extractors."""
        return list(self.extractors.keys())


# Statistical analysis service
class StatisticalAnalysisService:
    """Service for statistical analysis operations."""
    
    def __init__(self):
        self.analysis_methods = {
            "basic_stats": self._calculate_basic_stats,
            "distribution_analysis": self._analyze_distribution,
            "correlation_analysis": self._analyze_correlations,
            "outlier_detection": self._detect_outliers
        }
    
    def analyze(self, data: np.ndarray, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform statistical analysis on data."""
        if methods is None:
            methods = list(self.analysis_methods.keys())
        
        results = {}
        for method in methods:
            if method in self.analysis_methods:
                try:
                    results[method] = self.analysis_methods[method](data)
                except Exception as e:
                    logger.warning(f"Statistical analysis failed for {method}: {e}")
                    results[method] = {"error": str(e)}
        
        return results
    
    def _calculate_basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        if data.size == 0:
            return {"error": "Empty data"}
        
        # Flatten data for analysis
        flat_data = data.reshape(-1)
        valid_data = flat_data[np.isfinite(flat_data)]
        
        if len(valid_data) == 0:
            return {"error": "No valid data points"}
        
        return {
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "median": float(np.median(valid_data)),
            "q25": float(np.percentile(valid_data, 25)),
            "q75": float(np.percentile(valid_data, 75)),
            "skewness": float(self._calculate_skewness(valid_data)),
            "kurtosis": float(self._calculate_kurtosis(valid_data))
        }
    
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
    
    def _analyze_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data distribution."""
        flat_data = data.reshape(-1)
        valid_data = flat_data[np.isfinite(flat_data)]
        
        if len(valid_data) == 0:
            return {"error": "No valid data"}
        
        # Histogram analysis
        hist, bin_edges = np.histogram(valid_data, bins=50, density=True)
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "entropy": float(-np.sum(hist * np.log2(hist + 1e-10))),
            "range": float(np.ptp(valid_data))
        }
    
    def _analyze_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations in multi-dimensional data."""
        if data.ndim < 2:
            return {"error": "Need at least 2D data for correlation analysis"}
        
        # Reshape to (samples, features)
        if data.ndim > 2:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data
        
        try:
            correlation_matrix = np.corrcoef(data_2d.T)
            return {
                "correlation_matrix": correlation_matrix.tolist(),
                "max_correlation": float(np.max(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0])))),
                "mean_correlation": float(np.mean(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0]))))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        flat_data = data.reshape(-1)
        valid_data = flat_data[np.isfinite(flat_data)]
        
        if len(valid_data) == 0:
            return {"error": "No valid data"}
        
        # Z-score method
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        z_scores = np.abs((valid_data - mean) / (std + 1e-10))
        z_outliers = np.sum(z_scores > 3)
        
        # IQR method
        q25, q75 = np.percentile(valid_data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        iqr_outliers = np.sum((valid_data < lower_bound) | (valid_data > upper_bound))
        
        return {
            "z_score_outliers": int(z_outliers),
            "iqr_outliers": int(iqr_outliers),
            "outlier_percentage_z": float(z_outliers / len(valid_data) * 100),
            "outlier_percentage_iqr": float(iqr_outliers / len(valid_data) * 100)
        }


# Machine learning service
class MLModelService:
    """Service for machine learning operations."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.models = {
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
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str, **kwargs) -> Any:
        """Train a machine learning model."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self.models[model_type]
        model_class = model_config["class"]
        
        # Set random state if provided
        if self.random_state is not None:
            kwargs["random_state"] = self.random_state
        
        model = model_class(**kwargs)
        model.fit(X, y)
        return model
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """Evaluate a trained model."""
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            
            return {
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "cv_scores": cv_scores.tolist()
            }
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a model."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self.models[model_type]
        param_grid = model_config["param_grid"]
        
        best_score = 0.0
        best_params = {}
        
        for params in ParameterGrid(param_grid):
            try:
                model = self.train_model(X, y, model_type, **params)
                evaluation = self.evaluate_model(model, X, y)
                score = evaluation.get("cv_mean", 0.0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed for {params}: {e}")
                continue
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "model_type": model_type
        }


# Service registry
class ServiceRegistry:
    """Registry for managing services."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self._register_default_services()
    
    def _register_default_services(self):
        """Register default services."""
        self.services["cache"] = CacheService()
        self.services["parallel_processing"] = ParallelProcessingService()
        self.services["feature_extraction"] = EnhancedFeatureExtractionService(
            self.services["cache"], self.services["parallel_processing"]
        )
        self.services["statistical_analysis"] = StatisticalAnalysisService()
        self.services["ml_model"] = MLModelService()
    
    def register(self, name: str, service: Any):
        """Register a service."""
        self.services[name] = service
    
    def get(self, name: str) -> Any:
        """Get a service."""
        if name not in self.services:
            raise ValueError(f"Service not found: {name}")
        return self.services[name]
    
    def list_services(self) -> List[str]:
        """List available services."""
        return list(self.services.keys())


# Global service registry instance
service_registry = ServiceRegistry()


if __name__ == "__main__":
    # Example usage
    registry = ServiceRegistry()
    
    # Test feature extraction
    feature_service = registry.get("feature_extraction")
    test_signals = np.random.randn(2, 1000, 1)
    features = feature_service.extract_features(test_signals, ["mean", "std", "rms"])
    print(f"Extracted features: {list(features.keys())}")
    
    # Test statistical analysis
    stats_service = registry.get("statistical_analysis")
    stats = stats_service.analyze(test_signals)
    print(f"Statistical analysis: {list(stats.keys())}")
    
    print("Service layer test completed successfully!")
