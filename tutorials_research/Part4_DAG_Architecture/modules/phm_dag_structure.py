"""
PHMGA DAG Structure Implementation

Demonstrates how the PHMGA system uses Directed Acyclic Graphs
for signal processing pipelines and fault diagnosis workflows.
"""

import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
from abc import ABC, abstractmethod

from dag_fundamentals import ResearchDAG, DAGNode, NodeType


class SignalProcessingNodeType(Enum):
    """Specialized node types for signal processing"""
    SIGNAL_INPUT = "signal_input"
    PREPROCESSING = "preprocessing" 
    FEATURE_EXTRACTION = "feature_extraction"
    TRANSFORMATION = "transformation"
    CLASSIFICATION = "classification"
    DIAGNOSIS_OUTPUT = "diagnosis_output"
    VALIDATION = "validation"


class OperatorCategory(Enum):
    """Categories of signal processing operators in PHMGA"""
    TIME_DOMAIN = "time_domain"
    FREQUENCY_DOMAIN = "frequency_domain" 
    TIME_FREQUENCY = "time_frequency"
    STATISTICAL = "statistical"
    FILTERING = "filtering"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class SignalMetadata:
    """Metadata for signal data in PHMGA workflows"""
    sampling_rate: float
    duration: float
    channels: int
    signal_type: str = "vibration"
    fault_labels: Optional[List[str]] = None
    quality_score: float = 1.0
    
    def __post_init__(self):
        if self.fault_labels is None:
            self.fault_labels = ["normal", "inner_race", "outer_race", "ball"]


@dataclass 
class OperatorSpec:
    """Specification for signal processing operators"""
    operator_name: str
    category: OperatorCategory
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)
    computational_cost: float = 1.0  # Relative cost metric
    memory_requirement: float = 1.0  # Relative memory metric


class SignalProcessingNode(DAGNode):
    """
    Specialized DAG node for signal processing operations.
    
    Extends the base DAGNode with signal processing specific
    functionality and metadata.
    """
    
    def __init__(self, node_id: str, operator_spec: OperatorSpec, 
                 operation: callable = None):
        # Convert to standard node type
        if operator_spec.category in [OperatorCategory.TIME_DOMAIN, 
                                    OperatorCategory.FREQUENCY_DOMAIN,
                                    OperatorCategory.TIME_FREQUENCY]:
            node_type = NodeType.PROCESSING
        elif operator_spec.category == OperatorCategory.MACHINE_LEARNING:
            node_type = NodeType.DECISION
        else:
            node_type = NodeType.PROCESSING
            
        super().__init__(node_id, operator_spec.operator_name, node_type, operation)
        
        self.operator_spec = operator_spec
        self.signal_metadata: Optional[SignalMetadata] = None
        self.performance_metrics = {
            "execution_time": 0.0,
            "memory_used": 0.0,
            "accuracy": 0.0,
            "throughput": 0.0
        }
    
    def validate_input_compatibility(self, input_shape: Tuple[int, ...]) -> bool:
        """Validate that input shape is compatible with operator"""
        expected_shape = self.operator_spec.input_shape
        
        # Allow flexible batch dimension (first dimension)
        if len(input_shape) != len(expected_shape):
            return False
            
        for i, (actual, expected) in enumerate(zip(input_shape, expected_shape)):
            if i == 0:  # Skip batch dimension check
                continue
            if expected != -1 and actual != expected:  # -1 means any size
                return False
                
        return True
    
    def estimate_computational_cost(self, input_size: int) -> float:
        """Estimate computational cost for given input size"""
        base_cost = self.operator_spec.computational_cost
        size_factor = input_size / 1000  # Normalize to 1k samples
        return base_cost * size_factor


class PHMSignalProcessingDAG(ResearchDAG):
    """
    Specialized DAG for PHM (Prognostics and Health Management) workflows.
    
    Implements signal processing pipelines for bearing fault diagnosis
    and other PHM applications using the PHMGA architecture.
    """
    
    def __init__(self, analysis_type: str = "bearing_fault_diagnosis"):
        super().__init__(f"phm_{analysis_type}", f"PHM Analysis: {analysis_type}")
        self.analysis_type = analysis_type
        self.signal_metadata: Optional[SignalMetadata] = None
        self.operator_registry: Dict[str, OperatorSpec] = {}
        self.execution_plan: List[Dict[str, Any]] = []
        
        # Register standard PHM operators
        self._register_standard_operators()
        
        # Build analysis pipeline
        self._build_analysis_pipeline()
    
    def _register_standard_operators(self):
        """Register standard PHMGA signal processing operators"""
        
        # Time domain operators
        self.operator_registry["mean"] = OperatorSpec(
            "Mean", OperatorCategory.TIME_DOMAIN,
            input_shape=(-1, -1), output_shape=(-1, 1),
            parameters={"axis": 1}
        )
        
        self.operator_registry["rms"] = OperatorSpec(
            "RMS", OperatorCategory.TIME_DOMAIN,
            input_shape=(-1, -1), output_shape=(-1, 1),
            computational_cost=1.2
        )
        
        self.operator_registry["kurtosis"] = OperatorSpec(
            "Kurtosis", OperatorCategory.STATISTICAL,
            input_shape=(-1, -1), output_shape=(-1, 1),
            computational_cost=2.0
        )
        
        # Frequency domain operators  
        self.operator_registry["fft"] = OperatorSpec(
            "FFT", OperatorCategory.FREQUENCY_DOMAIN,
            input_shape=(-1, -1), output_shape=(-1, -1),
            computational_cost=3.0, memory_requirement=2.0
        )
        
        self.operator_registry["power_spectrum"] = OperatorSpec(
            "Power Spectrum", OperatorCategory.FREQUENCY_DOMAIN,
            input_shape=(-1, -1), output_shape=(-1, -1),
            computational_cost=3.5
        )
        
        # Time-frequency operators
        self.operator_registry["stft"] = OperatorSpec(
            "STFT", OperatorCategory.TIME_FREQUENCY,
            input_shape=(-1, -1), output_shape=(-1, -1, -1),
            parameters={"window": "hann", "nperseg": 256, "noverlap": 128},
            computational_cost=5.0, memory_requirement=4.0
        )
        
        self.operator_registry["wavelet"] = OperatorSpec(
            "Wavelet Transform", OperatorCategory.TIME_FREQUENCY,
            input_shape=(-1, -1), output_shape=(-1, -1, -1),
            parameters={"wavelet": "db4", "levels": 6},
            computational_cost=4.0, memory_requirement=3.0
        )
        
        # Machine learning operators
        self.operator_registry["svm_classifier"] = OperatorSpec(
            "SVM Classifier", OperatorCategory.MACHINE_LEARNING,
            input_shape=(-1, -1), output_shape=(-1, 4),  # 4 fault classes
            parameters={"C": 1.0, "kernel": "rbf"},
            computational_cost=2.5
        )
        
        self.operator_registry["feature_selector"] = OperatorSpec(
            "Feature Selection", OperatorCategory.MACHINE_LEARNING,
            input_shape=(-1, -1), output_shape=(-1, -1),
            parameters={"k_best": 20, "score_func": "f_classif"},
            computational_cost=1.5
        )
    
    def _build_analysis_pipeline(self):
        """Build the complete PHM analysis pipeline"""
        
        if self.analysis_type == "bearing_fault_diagnosis":
            self._build_bearing_fault_pipeline()
        elif self.analysis_type == "vibration_analysis":
            self._build_vibration_analysis_pipeline()
        else:
            self._build_generic_phm_pipeline()
    
    def _build_bearing_fault_pipeline(self):
        """Build bearing fault diagnosis pipeline"""
        
        # Create operations for each processing step
        operations = self._create_processing_operations()
        
        # Stage 1: Signal Input
        input_node = SignalProcessingNode(
            "signal_input",
            OperatorSpec("Signal Input", OperatorCategory.TIME_DOMAIN, 
                        input_shape=(None,), output_shape=(-1, -1)),
            operations["signal_input"]
        )
        self.add_node(input_node)
        
        # Stage 2: Preprocessing (parallel)
        preprocessing_nodes = []
        for preprocess_op in ["normalization", "detrending", "filtering"]:
            node = SignalProcessingNode(
                preprocess_op,
                OperatorSpec(preprocess_op.title(), OperatorCategory.FILTERING,
                           input_shape=(-1, -1), output_shape=(-1, -1)),
                operations[preprocess_op]
            )
            preprocessing_nodes.append(node.node_id)
            self.add_node(node)
            self.add_edge("signal_input", node.node_id)
        
        # Stage 3: Feature Extraction (parallel streams)
        feature_nodes = []
        
        # Time domain features
        time_features = ["mean", "rms", "kurtosis", "skewness", "crest_factor"]
        for feature in time_features:
            node = SignalProcessingNode(
                f"time_{feature}",
                self.operator_registry.get(feature, 
                    OperatorSpec(feature.title(), OperatorCategory.TIME_DOMAIN,
                               input_shape=(-1, -1), output_shape=(-1, 1))),
                operations.get(feature, operations["generic_feature"])
            )
            feature_nodes.append(node.node_id)
            self.add_node(node)
            # Connect to all preprocessing nodes
            for prep_node in preprocessing_nodes:
                self.add_edge(prep_node, node.node_id)
        
        # Frequency domain features
        freq_features = ["fft", "power_spectrum", "spectral_centroid"]
        for feature in freq_features:
            node = SignalProcessingNode(
                f"freq_{feature}",
                self.operator_registry.get(feature,
                    OperatorSpec(feature.title(), OperatorCategory.FREQUENCY_DOMAIN,
                               input_shape=(-1, -1), output_shape=(-1, -1))),
                operations.get(feature, operations["generic_feature"])
            )
            feature_nodes.append(node.node_id)
            self.add_node(node)
            for prep_node in preprocessing_nodes:
                self.add_edge(prep_node, node.node_id)
        
        # Time-frequency features
        tf_features = ["stft", "wavelet"]
        for feature in tf_features:
            node = SignalProcessingNode(
                f"tf_{feature}",
                self.operator_registry[feature],
                operations.get(feature, operations["generic_feature"])
            )
            feature_nodes.append(node.node_id)
            self.add_node(node)
            for prep_node in preprocessing_nodes:
                self.add_edge(prep_node, node.node_id)
        
        # Stage 4: Feature Aggregation
        aggregation_node = SignalProcessingNode(
            "feature_aggregation",
            OperatorSpec("Feature Aggregation", OperatorCategory.STATISTICAL,
                        input_shape=(-1, -1), output_shape=(-1, -1)),
            operations["feature_aggregation"]
        )
        self.add_node(aggregation_node)
        for feature_node in feature_nodes:
            self.add_edge(feature_node, "feature_aggregation")
        
        # Stage 5: Feature Selection
        selection_node = SignalProcessingNode(
            "feature_selection",
            self.operator_registry["feature_selector"],
            operations["feature_selection"]
        )
        self.add_node(selection_node)
        self.add_edge("feature_aggregation", "feature_selection")
        
        # Stage 6: Classification
        classifier_node = SignalProcessingNode(
            "classification",
            self.operator_registry["svm_classifier"],
            operations["classification"]
        )
        self.add_node(classifier_node)
        self.add_edge("feature_selection", "classification")
        
        # Stage 7: Diagnosis Output
        output_node = SignalProcessingNode(
            "diagnosis_output",
            OperatorSpec("Diagnosis Output", OperatorCategory.MACHINE_LEARNING,
                        input_shape=(-1, 4), output_shape=(-1, 1)),
            operations["diagnosis_output"]
        )
        self.add_node(output_node)
        self.add_edge("classification", "diagnosis_output")
        
        # Stage 8: Validation (parallel)
        validation_node = SignalProcessingNode(
            "result_validation",
            OperatorSpec("Result Validation", OperatorCategory.STATISTICAL,
                        input_shape=(-1, 1), output_shape=(-1, 1)),
            operations["validation"]
        )
        self.add_node(validation_node)
        self.add_edge("diagnosis_output", "result_validation")
    
    def _create_processing_operations(self) -> Dict[str, callable]:
        """Create actual processing operations for demonstration"""
        
        operations = {}
        
        def signal_input(inputs):
            """Generate demo vibration signal data"""
            np.random.seed(42)  # Reproducible results
            
            # Simulate bearing vibration signals
            fs = 10000  # 10 kHz sampling rate
            duration = 1.0  # 1 second
            t = np.linspace(0, duration, int(fs * duration))
            
            # Base signal with bearing fault frequencies
            signal = np.sin(2 * np.pi * 60 * t)  # Shaft frequency
            signal += 0.3 * np.sin(2 * np.pi * 157 * t)  # Inner race fault
            signal += 0.2 * np.sin(2 * np.pi * 236 * t)  # Outer race fault
            signal += 0.1 * np.random.randn(len(t))  # Noise
            
            # Create batch of signals
            batch_size = 10
            signals = np.array([signal + 0.1 * np.random.randn(len(t)) 
                              for _ in range(batch_size)])
            
            return {
                "signals": signals,
                "sampling_rate": fs,
                "duration": duration,
                "signal_shape": signals.shape,
                "metadata": SignalMetadata(fs, duration, 1)
            }
        
        def normalization(inputs):
            """Normalize signal data"""
            signals = inputs.get("signals", np.array([]))
            if signals.size == 0:
                return {"normalized_signals": np.array([])}
            
            # Z-score normalization
            normalized = (signals - np.mean(signals, axis=1, keepdims=True)) / (
                np.std(signals, axis=1, keepdims=True) + 1e-8)
            
            return {"normalized_signals": normalized}
        
        def detrending(inputs):
            """Remove trend from signals"""
            signals = inputs.get("signals", np.array([]))
            if signals.size == 0:
                return {"detrended_signals": np.array([])}
            
            # Simple detrending (remove mean)
            detrended = signals - np.mean(signals, axis=1, keepdims=True)
            
            return {"detrended_signals": detrended}
        
        def filtering(inputs):
            """Apply filtering to signals"""
            signals = inputs.get("signals", np.array([]))
            if signals.size == 0:
                return {"filtered_signals": np.array([])}
            
            # Simple low-pass filter simulation
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 0.3)  # 4th order Butterworth filter
            
            filtered = np.array([filtfilt(b, a, sig) for sig in signals])
            
            return {"filtered_signals": filtered}
        
        def generic_feature(inputs):
            """Generic feature extraction"""
            # Find the first signal-like array in inputs
            signal_data = None
            for key, value in inputs.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    signal_data = value
                    break
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, np.ndarray) and v.ndim >= 2:
                            signal_data = v
                            break
            
            if signal_data is None:
                return {"features": np.random.randn(10, 1)}  # Dummy features
            
            # Extract simple statistical features
            features = np.array([
                np.mean(signal_data, axis=1),
                np.std(signal_data, axis=1),
                np.max(signal_data, axis=1) - np.min(signal_data, axis=1)
            ]).T
            
            return {"features": features}
        
        def feature_aggregation(inputs):
            """Aggregate features from multiple sources"""
            all_features = []
            
            for key, value in inputs.items():
                if isinstance(value, dict) and "features" in value:
                    features = value["features"]
                    if isinstance(features, np.ndarray):
                        all_features.append(features)
            
            if not all_features:
                return {"aggregated_features": np.random.randn(10, 20)}
            
            # Concatenate all features
            aggregated = np.concatenate(all_features, axis=1)
            
            return {"aggregated_features": aggregated}
        
        def feature_selection(inputs):
            """Select best features for classification"""
            features = inputs.get("feature_aggregation", {}).get("aggregated_features", 
                                                               np.random.randn(10, 20))
            
            # Simple feature selection - select top half
            n_features = features.shape[1]
            selected_indices = np.argsort(np.var(features, axis=0))[-n_features//2:]
            selected_features = features[:, selected_indices]
            
            return {
                "selected_features": selected_features,
                "selected_indices": selected_indices,
                "n_features_selected": len(selected_indices)
            }
        
        def classification(inputs):
            """Classify fault types"""
            features = inputs.get("feature_selection", {}).get("selected_features",
                                                              np.random.randn(10, 10))
            
            # Simulate classification probabilities
            n_samples = features.shape[0]
            n_classes = 4  # normal, inner, outer, ball
            
            # Generate realistic-looking probabilities
            logits = np.random.randn(n_samples, n_classes)
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            
            return {
                "class_probabilities": probabilities,
                "predicted_classes": np.argmax(probabilities, axis=1),
                "confidence_scores": np.max(probabilities, axis=1)
            }
        
        def diagnosis_output(inputs):
            """Generate final diagnosis"""
            classification_result = inputs.get("classification", {})
            predicted_classes = classification_result.get("predicted_classes", np.array([0]))
            confidence_scores = classification_result.get("confidence_scores", np.array([0.5]))
            
            # Map class indices to fault names
            fault_names = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
            
            diagnoses = []
            for class_idx, confidence in zip(predicted_classes, confidence_scores):
                diagnosis = {
                    "fault_type": fault_names[class_idx],
                    "confidence": float(confidence),
                    "severity": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
                    "recommendation": "Monitor closely" if confidence > 0.7 else "Further analysis needed"
                }
                diagnoses.append(diagnosis)
            
            return {"diagnoses": diagnoses}
        
        def validation(inputs):
            """Validate diagnosis results"""
            diagnoses = inputs.get("diagnosis_output", {}).get("diagnoses", [])
            
            validation_results = {
                "total_samples": len(diagnoses),
                "high_confidence_count": sum(1 for d in diagnoses if d.get("confidence", 0) > 0.8),
                "average_confidence": np.mean([d.get("confidence", 0) for d in diagnoses]),
                "fault_distribution": {},
                "validation_passed": True
            }
            
            # Count fault types
            for diagnosis in diagnoses:
                fault_type = diagnosis.get("fault_type", "Unknown")
                validation_results["fault_distribution"][fault_type] = (
                    validation_results["fault_distribution"].get(fault_type, 0) + 1
                )
            
            return validation_results
        
        # Populate operations dictionary
        operations.update({
            "signal_input": signal_input,
            "normalization": normalization,
            "detrending": detrending,
            "filtering": filtering,
            "generic_feature": generic_feature,
            "feature_aggregation": feature_aggregation,
            "feature_selection": feature_selection,
            "classification": classification,
            "diagnosis_output": diagnosis_output,
            "validation": validation
        })
        
        # Add specific feature operations
        for feature_name in ["mean", "rms", "kurtosis", "fft", "power_spectrum", "stft", "wavelet"]:
            operations[feature_name] = generic_feature
        
        return operations
    
    def _build_vibration_analysis_pipeline(self):
        """Build generic vibration analysis pipeline"""
        # Simplified pipeline for demo
        pass
    
    def _build_generic_phm_pipeline(self):
        """Build generic PHM pipeline"""
        # Simplified pipeline for demo
        pass
    
    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """Generate execution plan with resource estimates"""
        if not self.execution_order:
            self.topological_sort()
        
        execution_plan = []
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            if isinstance(node, SignalProcessingNode):
                plan_item = {
                    "node_id": node_id,
                    "operator": node.operator_spec.operator_name,
                    "category": node.operator_spec.category.value,
                    "estimated_cost": node.operator_spec.computational_cost,
                    "memory_requirement": node.operator_spec.memory_requirement,
                    "dependencies": list(node.dependencies),
                    "parallel_group": self._get_parallel_group(node_id)
                }
                execution_plan.append(plan_item)
        
        return execution_plan
    
    def _get_parallel_group(self, node_id: str) -> int:
        """Determine which nodes can execute in parallel"""
        # Simple grouping based on dependencies
        if not self.nodes[node_id].dependencies:
            return 0  # Input nodes
        
        max_dep_group = 0
        for dep in self.nodes[node_id].dependencies:
            if dep in [n["node_id"] for n in self.execution_plan]:
                dep_group = next(n["parallel_group"] for n in self.execution_plan 
                               if n["node_id"] == dep)
                max_dep_group = max(max_dep_group, dep_group)
        
        return max_dep_group + 1
    
    def optimize_execution(self) -> Dict[str, Any]:
        """Optimize DAG execution based on resource constraints"""
        plan = self.get_execution_plan()
        
        optimization_results = {
            "original_sequential_cost": sum(item["estimated_cost"] for item in plan),
            "parallel_groups": {},
            "critical_path": [],
            "memory_peaks": [],
            "optimization_suggestions": []
        }
        
        # Group by parallel execution possibility
        parallel_groups = {}
        for item in plan:
            group = item["parallel_group"]
            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append(item)
        
        optimization_results["parallel_groups"] = parallel_groups
        
        # Calculate optimized execution time (parallel)
        group_costs = [max(item["estimated_cost"] for item in group) 
                      for group in parallel_groups.values()]
        optimization_results["optimized_parallel_cost"] = sum(group_costs)
        
        # Calculate speedup
        speedup = optimization_results["original_sequential_cost"] / optimization_results["optimized_parallel_cost"]
        optimization_results["speedup_factor"] = speedup
        
        # Memory analysis
        for group_id, group_items in parallel_groups.items():
            memory_usage = sum(item["memory_requirement"] for item in group_items)
            optimization_results["memory_peaks"].append({
                "group": group_id,
                "memory_usage": memory_usage
            })
        
        # Generate optimization suggestions
        max_memory_group = max(optimization_results["memory_peaks"], 
                             key=lambda x: x["memory_usage"])
        if max_memory_group["memory_usage"] > 8.0:  # Threshold
            optimization_results["optimization_suggestions"].append(
                "Consider memory optimization for high-memory operations"
            )
        
        if speedup < 2.0:
            optimization_results["optimization_suggestions"].append(
                "Limited parallelization benefit - consider algorithmic optimizations"
            )
        
        return optimization_results


def demonstrate_phm_dag():
    """Demonstrate PHM DAG structure and execution"""
    
    print("⚙️ PHMGA DAG STRUCTURE DEMONSTRATION")
    print("=" * 50)
    
    print("\\n🔧 Creating Bearing Fault Diagnosis DAG...")
    phm_dag = PHMSignalProcessingDAG("bearing_fault_diagnosis")
    
    print(f"✅ Created PHM DAG with {len(phm_dag.nodes)} nodes")
    print(f"📋 Operator registry: {len(phm_dag.operator_registry)} operators")
    
    # Show DAG structure
    print("\\n📊 DAG Structure:")
    print(phm_dag.visualize_structure())
    
    # Show execution plan
    print("\\n🗺️ Execution Plan:")
    execution_plan = phm_dag.get_execution_plan()
    for i, item in enumerate(execution_plan):
        print(f"   {i+1}. {item['operator']} ({item['category']}) - Group {item['parallel_group']}")
        print(f"      Cost: {item['estimated_cost']:.1f}, Memory: {item['memory_requirement']:.1f}")
    
    # Show optimization analysis
    print("\\n🚀 Execution Optimization Analysis:")
    optimization = phm_dag.optimize_execution()
    print(f"   • Sequential cost: {optimization['original_sequential_cost']:.1f}")
    print(f"   • Parallel cost: {optimization['optimized_parallel_cost']:.1f}")
    print(f"   • Speedup factor: {optimization['speedup_factor']:.1f}x")
    print(f"   • Parallel groups: {len(optimization['parallel_groups'])}")
    
    if optimization['optimization_suggestions']:
        print("\\n💡 Optimization Suggestions:")
        for suggestion in optimization['optimization_suggestions']:
            print(f"   • {suggestion}")
    
    # Execute the DAG
    print("\\n🚀 Executing PHM Analysis Pipeline...")
    try:
        start_time = time.time()
        results = phm_dag.execute()
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline completed in {execution_time:.2f} seconds")
        
        # Show diagnosis results
        if "diagnosis_output" in results:
            diagnoses = results["diagnosis_output"].get("diagnoses", [])
            print(f"\\n🔍 Diagnosis Results ({len(diagnoses)} samples):")
            for i, diagnosis in enumerate(diagnoses[:3]):  # Show first 3
                print(f"   Sample {i+1}: {diagnosis['fault_type']} "
                     f"(confidence: {diagnosis['confidence']:.2f}, "
                     f"severity: {diagnosis['severity']})")
        
        # Show validation results
        if "result_validation" in results:
            validation = results["result_validation"]
            print(f"\\n✅ Validation Results:")
            print(f"   • Average confidence: {validation.get('average_confidence', 0):.2f}")
            print(f"   • High confidence samples: {validation.get('high_confidence_count', 0)}")
            print(f"   • Fault distribution: {validation.get('fault_distribution', {})}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_phm_dag()