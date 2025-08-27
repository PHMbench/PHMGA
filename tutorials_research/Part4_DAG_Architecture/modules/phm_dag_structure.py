"""
PHMGA DAG Structure Implementation

Simplified demonstration of how DAGs can be applied to 
Prognostics and Health Management (PHM) signal processing workflows.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import random
import math

# Smart import handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from dag_fundamentals import ResearchDAG, DAGNode, NodeType


# ====================================
# NUMPY-COMPATIBLE UTILITY FUNCTIONS  
# ====================================

def random_seed(seed):
    """Set random seed for reproducibility"""
    if NUMPY_AVAILABLE:
        np.random.seed(seed)
    else:
        random.seed(seed)


def random_randn(*shape):
    """Generate random normal distribution numbers"""
    if NUMPY_AVAILABLE:
        if len(shape) == 1:
            return np.random.randn(shape[0])
        return np.random.randn(*shape)
    else:
        # Simple normal distribution approximation using Box-Muller transform
        if len(shape) == 1:
            return [random.gauss(0, 1) for _ in range(shape[0])]
        # For simplicity, just return list for multi-dim
        total = 1
        for s in shape:
            total *= s
        return [random.gauss(0, 1) for _ in range(total)]


def linspace(start, stop, num):
    """Generate linearly spaced numbers"""
    if NUMPY_AVAILABLE:
        return np.linspace(start, stop, num)
    else:
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]


def sin(x):
    """Sine function compatible with arrays"""
    if NUMPY_AVAILABLE and hasattr(x, '__iter__'):
        return np.sin(x)
    elif hasattr(x, '__iter__'):
        return [math.sin(val) for val in x]
    else:
        return math.sin(x)


def array(data):
    """Create array from data"""
    if NUMPY_AVAILABLE:
        return np.array(data)
    else:
        return list(data)


def mean(data, **kwargs):
    """Calculate mean"""
    if NUMPY_AVAILABLE and hasattr(data, 'mean'):
        return data.mean(**kwargs)
    else:
        if hasattr(data, '__iter__'):
            return sum(data) / len(data) if len(data) > 0 else 0
        return data


def std(data, **kwargs):
    """Calculate standard deviation"""
    if NUMPY_AVAILABLE and hasattr(data, 'std'):
        return data.std(**kwargs)
    else:
        if hasattr(data, '__iter__'):
            if len(data) <= 1:
                return 0
            mean_val = mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return math.sqrt(variance)
        return 0


def abs_func(data):
    """Absolute value function"""
    if NUMPY_AVAILABLE and hasattr(data, '__iter__'):
        return np.abs(data)
    elif hasattr(data, '__iter__'):
        return [abs(x) for x in data]
    else:
        return abs(data)


def max_func(data):
    """Maximum value function"""
    if NUMPY_AVAILABLE and hasattr(data, 'max'):
        return data.max()
    elif hasattr(data, '__iter__'):
        return max(data) if len(data) > 0 else 0
    else:
        return data


def sqrt(data):
    """Square root function"""
    if NUMPY_AVAILABLE and hasattr(data, '__iter__'):
        return np.sqrt(data)
    elif hasattr(data, '__iter__'):
        return [math.sqrt(x) for x in data if x >= 0]
    else:
        return math.sqrt(data) if data >= 0 else 0


class PHMProcessType(Enum):
    """Simple PHM processing types"""
    SIGNAL_INPUT = "signal_input"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction" 
    CLASSIFICATION = "classification"
    DIAGNOSIS = "diagnosis"


@dataclass
class PHMConfig:
    """Simple PHM analysis configuration"""
    analysis_type: str = "bearing_fault"
    sampling_rate: float = 10000.0
    signal_length: int = 1000
    fault_types: List[str] = None
    
    def __post_init__(self):
        if self.fault_types is None:
            self.fault_types = ["normal", "inner_fault", "outer_fault", "ball_fault"]

class SimplePHMDAG(ResearchDAG):
    """
    Simplified DAG for PHM (Prognostics and Health Management) workflows.
    
    Demonstrates core DAG concepts applied to bearing fault diagnosis
    with a clean, educational focus.
    """
    
    def __init__(self, config: PHMConfig):
        super().__init__(f"phm_{config.analysis_type}", f"PHM Analysis: {config.analysis_type}")
        self.config = config
        self._build_simple_pipeline()
    
    def _build_simple_pipeline(self):
        """Build a simple 5-stage PHM pipeline"""
        
        def signal_input_operation(inputs):
            """Generate synthetic vibration signal"""
            random_seed(42)  # Reproducible results
            
            fs = self.config.sampling_rate
            n_samples = self.config.signal_length
            t = linspace(0, n_samples/fs, n_samples)
            
            # Simulate bearing vibration with fault frequencies
            signal_base = sin([2 * math.pi * 60 * val for val in t])  # Shaft frequency
            signal_fault = [0.3 * math.sin(2 * math.pi * 157 * val) for val in t]  # Bearing fault frequency
            noise = [0.1 * val for val in random_randn(len(t))]  # Noise
            
            # Combine signals
            signal = [base + fault + n for base, fault, n in zip(signal_base, signal_fault, noise)]
            
            return {
                "raw_signal": signal,
                "sampling_rate": fs,
                "signal_length": len(signal),
                "signal_type": self.config.analysis_type
            }
        
        def preprocessing_operation(inputs):
            """Basic signal preprocessing"""
            time.sleep(0.1)  # Simulate processing time
            signal = inputs.get("signal_input", {}).get("raw_signal", [])
            
            # Simple preprocessing: normalize and filter
            if len(signal) > 0:
                signal_mean = mean(signal)
                signal_std = std(signal)
                normalized = [(x - signal_mean) / (signal_std + 1e-8) for x in signal]
                
                # Simple moving average filter  
                window_size = 5
                filtered = []
                for i in range(len(normalized)):
                    start = max(0, i - window_size // 2)
                    end = min(len(normalized), i + window_size // 2 + 1)
                    window_avg = sum(normalized[start:end]) / (end - start)
                    filtered.append(window_avg)
            else:
                filtered = signal
            
            return {
                "processed_signal": filtered,
                "preprocessing_applied": ["normalization", "moving_average_filter"]
            }
        
        def feature_extraction_operation(inputs):
            """Extract key features from signal"""
            time.sleep(0.15)
            signal = inputs.get("preprocessing", {}).get("processed_signal", [])
            
            if len(signal) == 0:
                return {"features": random_randn(5)}
            
            # Extract simple statistical features
            signal_squared = [x**2 for x in signal]
            signal_abs = abs_func(signal)
            signal_mean = mean(signal)
            signal_std = std(signal)
            
            features = {
                "rms": sqrt(mean(signal_squared)),
                "peak": max_func(signal_abs), 
                "kurtosis": mean([((x - signal_mean)/signal_std)**4 for x in signal]) if signal_std > 0 else 0,
                "crest_factor": max_func(signal_abs) / sqrt(mean(signal_squared)) if mean(signal_squared) > 0 else 0,
                "energy": sum(signal_squared)
            }
            
            return {
                "extracted_features": features,
                "feature_vector": list(features.values()),
                "feature_names": list(features.keys())
            }
        
        def classification_operation(inputs):
            """Classify fault type based on features"""
            time.sleep(0.1)
            features = inputs.get("feature_extraction", {}).get("feature_vector", [])
            
            if len(features) == 0:
                features = random_randn(5)
            
            # Simulate classification with simple thresholds
            # Simple alternative to numpy.random.dirichlet
            abs_features = [abs(f) + 0.1 for f in features[:4]]  # Take first 4 for 4 fault classes
            total = sum(abs_features)
            fault_probs = [f / total for f in abs_features]
            predicted_class = fault_probs.index(max(fault_probs))
            
            return {
                "fault_probabilities": {
                    fault: prob for fault, prob in zip(self.config.fault_types, fault_probs)
                },
                "predicted_fault": self.config.fault_types[predicted_class],
                "confidence": fault_probs[predicted_class]
            }
        
        def diagnosis_operation(inputs):
            """Generate final diagnosis report"""
            classification = inputs.get("classification", {})
            predicted_fault = classification.get("predicted_fault", "unknown")
            confidence = classification.get("confidence", 0.5)
            
            # Generate diagnosis recommendations
            if confidence > 0.8:
                severity = "High confidence"
                recommendation = "Monitor closely, schedule maintenance"
            elif confidence > 0.6:
                severity = "Medium confidence" 
                recommendation = "Continue monitoring, verify with additional data"
            else:
                severity = "Low confidence"
                recommendation = "Collect more data for reliable diagnosis"
            
            return {
                "final_diagnosis": predicted_fault,
                "confidence_level": confidence,
                "severity_assessment": severity,
                "maintenance_recommendation": recommendation,
                "diagnosis_timestamp": time.time()
            }
        
        # Create DAG nodes
        nodes = [
            DAGNode("signal_input", "Signal Acquisition", NodeType.INPUT, signal_input_operation),
            DAGNode("preprocessing", "Signal Preprocessing", NodeType.PROCESSING, preprocessing_operation),  
            DAGNode("feature_extraction", "Feature Extraction", NodeType.PROCESSING, feature_extraction_operation),
            DAGNode("classification", "Fault Classification", NodeType.DECISION, classification_operation),
            DAGNode("diagnosis", "Diagnosis Report", NodeType.OUTPUT, diagnosis_operation)
        ]
        
        # Add nodes to DAG
        for node in nodes:
            self.add_node(node)
        
        # Create pipeline connections
        edges = [
            ("signal_input", "preprocessing"),
            ("preprocessing", "feature_extraction"), 
            ("feature_extraction", "classification"),
            ("classification", "diagnosis")
        ]
        
        for from_node, to_node in edges:
            self.add_edge(from_node, to_node)


class ParallelPHMDAG(ResearchDAG):
    """
    PHM DAG with parallel feature extraction branches.
    
    Demonstrates fan-out/fan-in pattern in signal processing.
    """
    
    def __init__(self, config: PHMConfig):
        super().__init__(f"parallel_phm_{config.analysis_type}", 
                        f"Parallel PHM Analysis: {config.analysis_type}")
        self.config = config
        self._build_parallel_pipeline()
    
    def _build_parallel_pipeline(self):
        """Build parallel feature extraction pipeline"""
        
        def signal_input_operation(inputs):
            """Generate synthetic signal data"""
            random_seed(42)
            t_vals = linspace(0, 1, 1000)
            signal = [math.sin(2 * math.pi * 60 * t) for t in t_vals]
            noise = random_randn(1000)
            signal = [s + 0.1 * n for s, n in zip(signal, noise)]
            return {"signal": signal}
        
        def time_domain_features(inputs):
            """Extract time domain features"""
            time.sleep(0.1)
            signal = inputs.get("signal_input", {}).get("signal", [])
            return {
                "time_features": {
                    "rms": sqrt(mean([x**2 for x in signal])) if len(signal) > 0 else 0,
                    "peak": max_func(abs_func(signal)) if len(signal) > 0 else 0
                }
            }
        
        def frequency_domain_features(inputs):
            """Extract frequency domain features"""
            time.sleep(0.15)
            signal = inputs.get("signal_input", {}).get("signal", [])
            if len(signal) > 0:
                # Simplified spectral energy calculation (without FFT)
                signal_energy = sum([x**2 for x in signal])
                return {"freq_features": {"spectral_energy": signal_energy}}
            return {"freq_features": {"spectral_energy": 0}}
        
        def statistical_features(inputs):
            """Extract statistical features"""
            time.sleep(0.08)
            signal = inputs.get("signal_input", {}).get("signal", [])
            if len(signal) > 0:
                signal_mean = mean(signal)
                signal_std = std(signal)
                return {
                    "stat_features": {
                        "mean": signal_mean,
                        "std": signal_std,
                        "kurtosis": mean([((x - signal_mean)/signal_std)**4 for x in signal]) if signal_std > 0 else 0
                    }
                }
            return {"stat_features": {"mean": 0, "std": 0, "kurtosis": 0}}
        
        def feature_fusion(inputs):
            """Combine all extracted features"""
            time_feats = inputs.get("time_features", {}).get("time_features", {})
            freq_feats = inputs.get("freq_features", {}).get("freq_features", {})
            stat_feats = inputs.get("stat_features", {}).get("stat_features", {})
            
            all_features = {**time_feats, **freq_feats, **stat_feats}
            return {"fused_features": all_features}
        
        def final_diagnosis(inputs):
            """Make diagnosis from fused features"""
            features = inputs.get("feature_fusion", {}).get("fused_features", {})
            
            # Simple diagnosis logic
            if len(features) > 0:
                feature_values = list(features.values())
                avg_magnitude = mean(abs_func(feature_values))
                
                if avg_magnitude > 0.5:
                    diagnosis = "Potential fault detected"
                else:
                    diagnosis = "Normal operation"
            else:
                diagnosis = "Insufficient data"
            
            return {"diagnosis": diagnosis, "feature_summary": features}
        
        # Create nodes
        nodes = [
            DAGNode("signal_input", "Signal Input", NodeType.INPUT, signal_input_operation),
            DAGNode("time_features", "Time Domain Features", NodeType.PROCESSING, time_domain_features),
            DAGNode("freq_features", "Frequency Features", NodeType.PROCESSING, frequency_domain_features),
            DAGNode("stat_features", "Statistical Features", NodeType.PROCESSING, statistical_features),
            DAGNode("feature_fusion", "Feature Fusion", NodeType.AGGREGATION, feature_fusion),
            DAGNode("final_diagnosis", "Final Diagnosis", NodeType.OUTPUT, final_diagnosis)
        ]
        
        for node in nodes:
            self.add_node(node)
        
        # Fan-out to parallel feature extraction
        for feature_node in ["time_features", "freq_features", "stat_features"]:
            self.add_edge("signal_input", feature_node)
        
        # Fan-in to fusion
        for feature_node in ["time_features", "freq_features", "stat_features"]:
            self.add_edge(feature_node, "feature_fusion")
        
        # Final diagnosis
        self.add_edge("feature_fusion", "final_diagnosis")


# Convenience functions

def create_simple_phm_workflow(analysis_type: str = "bearing_fault") -> SimplePHMDAG:
    """Create a simple 5-stage PHM analysis workflow"""
    config = PHMConfig(
        analysis_type=analysis_type,
        sampling_rate=10000.0,
        signal_length=1000,
        fault_types=["normal", "inner_fault", "outer_fault", "ball_fault"]
    )
    return SimplePHMDAG(config)


def create_parallel_phm_workflow(analysis_type: str = "vibration_analysis") -> ParallelPHMDAG:
    """Create a parallel feature extraction PHM workflow"""
    config = PHMConfig(analysis_type=analysis_type)
    return ParallelPHMDAG(config)


def demonstrate_phm_dag_patterns():
    """Demonstrate PHM DAG patterns for educational purposes"""
    
    print("🔧 PHM DAG PATTERNS DEMONSTRATION")
    print("=" * 50)
    
    # Pattern 1: Simple Linear PHM Pipeline
    print("\n🔄 Pattern 1: Simple Linear PHM Pipeline")
    simple_dag = create_simple_phm_workflow("bearing_fault_diagnosis")
    print(f"   • Nodes: {len(simple_dag.nodes)} (linear sequence)")
    print(f"   • Workflow: Signal → Preprocessing → Features → Classification → Diagnosis")
    
    results = simple_dag.execute()
    stats = simple_dag.get_statistics()
    print(f"   • Execution: {stats['success_rate']:.1%} success rate")
    
    if 'diagnosis' in results:
        diagnosis = results['diagnosis']
        print(f"   • Result: {diagnosis.get('final_diagnosis', 'Unknown')}")
        print(f"   • Confidence: {diagnosis.get('confidence_level', 0):.2f}")
    
    # Pattern 2: Parallel Feature Extraction
    print("\n⚡ Pattern 2: Parallel Feature Extraction")
    parallel_dag = create_parallel_phm_workflow("multi_domain_analysis")
    print(f"   • Nodes: {len(parallel_dag.nodes)} (parallel branches)")
    print(f"   • Structure: Input → [Time|Frequency|Statistical] → Fusion → Diagnosis")
    
    parallel_results = parallel_dag.execute()
    parallel_stats = parallel_dag.get_statistics()
    print(f"   • Execution: {parallel_stats['success_rate']:.1%} success rate")
    
    if 'final_diagnosis' in parallel_results:
        parallel_diagnosis = parallel_results['final_diagnosis']
        print(f"   • Result: {parallel_diagnosis.get('diagnosis', 'Unknown')}")
    
    # Show timing comparison
    simple_time = sum(node.execution_time for node in simple_dag.nodes.values())
    parallel_time = max([node.execution_time for node in parallel_dag.nodes.values() 
                        if node.node_id != "signal_input"])
    
    print(f"\n⚡ Parallelization Benefit:")
    print(f"   • Linear pipeline: {simple_time:.2f}s total")
    print(f"   • Parallel pipeline: {parallel_time:.2f}s (max branch)")
    if simple_time > 0 and parallel_time > 0:
        speedup = simple_time / parallel_time
        print(f"   • Theoretical speedup: {speedup:.1f}x")
    
    print(f"\n📊 Both PHM patterns ready for visualization!")
    print("Use DAGVisualizer to plot these workflows and see the differences.")
    print("Example: simple_dag.plot_dag() or quick_plot(parallel_dag)")


if __name__ == "__main__":
    demonstrate_phm_dag_patterns()
