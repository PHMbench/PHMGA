"""
Demo Configurations Module

Provides pre-configured educational scenarios and demo setups for different learning objectives.
Makes it easy for tutorial users to get started with various types of PHM analyses.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))


class LearningLevel(Enum):
    """Educational learning levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AnalysisType(Enum):
    """Types of PHM analyses"""
    BEARING_FAULT_DETECTION = "bearing_fault_detection"
    VIBRATION_ANALYSIS = "vibration_analysis"
    CONDITION_MONITORING = "condition_monitoring"
    FAULT_CLASSIFICATION = "fault_classification"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"


@dataclass
class DemoConfiguration:
    """Configuration for educational demo scenarios"""
    
    # Basic identification
    name: str
    description: str
    learning_level: LearningLevel
    analysis_type: AnalysisType
    learning_objectives: List[str]
    
    # PHMGA system configuration
    llm_provider: str = "google"
    llm_model: str = "gemini-2.0-flash-exp" 
    min_depth: int = 2
    max_depth: int = 4
    
    # Data configuration
    metadata_path: str = ""
    h5_path: str = ""
    ref_ids: List[int] = field(default_factory=list)
    test_ids: List[int] = field(default_factory=list)
    
    # Educational settings
    verbose_output: bool = True
    show_timing: bool = True
    explain_concepts: bool = True
    show_plots: bool = True
    save_results: bool = True
    
    # Analysis parameters
    user_instruction: str = ""
    expected_faults: List[str] = field(default_factory=list)
    difficulty_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Tutorial guidance
    pre_analysis_notes: List[str] = field(default_factory=list)
    post_analysis_notes: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    troubleshooting_tips: List[str] = field(default_factory=list)


class DemoConfigurationManager:
    """Manages educational demo configurations"""
    
    def __init__(self):
        self.configurations = self._build_demo_configurations()
        self.synthetic_data_configs = self._build_synthetic_data_configs()
    
    def _build_demo_configurations(self) -> Dict[str, DemoConfiguration]:
        """Build pre-configured demo scenarios"""
        
        configs = {}
        
        # Beginner Level Configurations
        configs["beginner_bearing_intro"] = DemoConfiguration(
            name="Beginner: Bearing Fault Introduction",
            description="Introduction to bearing fault detection using simple vibration signals",
            learning_level=LearningLevel.BEGINNER,
            analysis_type=AnalysisType.BEARING_FAULT_DETECTION,
            learning_objectives=[
                "Understand basic bearing fault types",
                "Learn signal processing fundamentals", 
                "See DAG construction in action",
                "Interpret basic fault detection results"
            ],
            min_depth=2,
            max_depth=3,
            user_instruction="Detect bearing faults in vibration signals using basic signal processing",
            expected_faults=["normal", "inner_race"],
            pre_analysis_notes=[
                "This demo introduces bearing fault detection concepts",
                "We'll use simple signal processing to identify faults",
                "Focus on understanding the DAG construction process"
            ],
            post_analysis_notes=[
                "Notice how the system built a processing pipeline automatically",
                "The DAG shows the flow from raw signals to fault detection",
                "Results indicate whether bearings are healthy or faulty"
            ],
            key_concepts=[
                "Signal processing DAG", "Bearing fault signatures", "Automated analysis"
            ],
            difficulty_factors={
                "signal_complexity": "low",
                "fault_types": 2,
                "processing_depth": "shallow"
            }
        )
        
        configs["beginner_signal_basics"] = DemoConfiguration(
            name="Beginner: Signal Processing Basics",
            description="Learn fundamental signal processing concepts with vibration data",
            learning_level=LearningLevel.BEGINNER,
            analysis_type=AnalysisType.VIBRATION_ANALYSIS,
            learning_objectives=[
                "Understand time-domain vs frequency-domain analysis",
                "Learn about FFT and spectral analysis",
                "See how operators transform signals",
                "Understand feature extraction"
            ],
            min_depth=2,
            max_depth=3,
            user_instruction="Analyze vibration signals to understand basic signal processing concepts",
            pre_analysis_notes=[
                "This demo focuses on signal processing fundamentals",
                "You'll see how raw signals are transformed step by step",
                "Pay attention to the different processing operators used"
            ],
            key_concepts=[
                "Time-domain analysis", "Frequency-domain analysis", "Signal operators", "Feature extraction"
            ]
        )
        
        # Intermediate Level Configurations  
        configs["intermediate_multi_fault"] = DemoConfiguration(
            name="Intermediate: Multi-Fault Classification",
            description="Classify multiple bearing fault types using advanced signal processing",
            learning_level=LearningLevel.INTERMEDIATE,
            analysis_type=AnalysisType.FAULT_CLASSIFICATION,
            learning_objectives=[
                "Classify multiple fault types simultaneously",
                "Use advanced signal processing operators",
                "Understand machine learning integration",
                "Interpret confidence scores and uncertainty"
            ],
            min_depth=3,
            max_depth=5,
            user_instruction="Classify bearing faults into multiple categories: normal, inner race, outer race, and ball faults",
            expected_faults=["normal", "inner_race", "outer_race", "ball"],
            pre_analysis_notes=[
                "This demo handles multiple fault types simultaneously",
                "More complex signal processing will be used",
                "Machine learning models will help with classification"
            ],
            post_analysis_notes=[
                "Notice the deeper DAG structure with more processing stages",
                "Multiple fault types require more sophisticated analysis",
                "Confidence scores help assess classification reliability"
            ],
            key_concepts=[
                "Multi-class classification", "Feature engineering", "ML integration", "Uncertainty quantification"
            ],
            difficulty_factors={
                "signal_complexity": "medium",
                "fault_types": 4,
                "processing_depth": "medium"
            }
        )
        
        configs["intermediate_condition_monitoring"] = DemoConfiguration(
            name="Intermediate: Condition Monitoring System",
            description="Build a complete condition monitoring system for industrial equipment",
            learning_level=LearningLevel.INTERMEDIATE,
            analysis_type=AnalysisType.CONDITION_MONITORING,
            learning_objectives=[
                "Implement continuous monitoring workflows",
                "Use multi-agent coordination for analysis",
                "Generate automated maintenance recommendations",
                "Understand system health indicators"
            ],
            min_depth=3,
            max_depth=5,
            user_instruction="Implement condition monitoring to assess equipment health and generate maintenance recommendations",
            pre_analysis_notes=[
                "This demo simulates a real condition monitoring system",
                "Multiple agents will work together to assess equipment health",
                "The system will generate actionable maintenance recommendations"
            ],
            key_concepts=[
                "Condition monitoring", "Multi-agent systems", "Health indicators", "Maintenance planning"
            ]
        )
        
        # Advanced Level Configurations
        configs["advanced_predictive_maintenance"] = DemoConfiguration(
            name="Advanced: Predictive Maintenance System",
            description="Implement sophisticated predictive maintenance with trend analysis and RUL estimation",
            learning_level=LearningLevel.ADVANCED,
            analysis_type=AnalysisType.PREDICTIVE_MAINTENANCE,
            learning_objectives=[
                "Implement remaining useful life (RUL) estimation",
                "Use advanced feature engineering techniques",
                "Integrate research-based approaches",
                "Generate comprehensive maintenance strategies"
            ],
            min_depth=4,
            max_depth=7,
            user_instruction="Develop predictive maintenance system with RUL estimation and trend analysis for bearing health assessment",
            expected_faults=["normal", "inner_race", "outer_race", "ball", "cage"],
            pre_analysis_notes=[
                "This advanced demo implements predictive maintenance concepts",
                "Complex signal processing and ML techniques will be used",
                "Research integration will enhance analysis capabilities"
            ],
            post_analysis_notes=[
                "Notice the sophisticated DAG with multiple processing branches",
                "Advanced features enable more accurate predictions",
                "Research integration provides cutting-edge analysis methods"
            ],
            key_concepts=[
                "Predictive maintenance", "RUL estimation", "Trend analysis", "Advanced ML", "Research integration"
            ],
            difficulty_factors={
                "signal_complexity": "high",
                "fault_types": 5,
                "processing_depth": "deep",
                "research_integration": True
            }
        )
        
        configs["advanced_multi_sensor"] = DemoConfiguration(
            name="Advanced: Multi-Sensor Fusion Analysis",
            description="Combine multiple sensor types for comprehensive equipment analysis",
            learning_level=LearningLevel.ADVANCED, 
            analysis_type=AnalysisType.CONDITION_MONITORING,
            learning_objectives=[
                "Implement multi-sensor data fusion",
                "Handle heterogeneous sensor data",
                "Use advanced correlation analysis",
                "Generate holistic equipment health assessment"
            ],
            min_depth=4,
            max_depth=6,
            user_instruction="Fuse vibration, temperature, and acoustic data for comprehensive equipment health assessment",
            key_concepts=[
                "Multi-sensor fusion", "Heterogeneous data", "Correlation analysis", "Holistic assessment"
            ],
            difficulty_factors={
                "signal_complexity": "high",
                "sensor_types": 3,
                "fusion_complexity": "high"
            }
        )
        
        # Expert Level Configuration
        configs["expert_research_integration"] = DemoConfiguration(
            name="Expert: Research-Integrated Analysis",
            description="Cutting-edge analysis integrating latest research findings and novel techniques",
            learning_level=LearningLevel.EXPERT,
            analysis_type=AnalysisType.FAULT_CLASSIFICATION,
            learning_objectives=[
                "Apply latest research findings to analysis",
                "Implement novel signal processing techniques", 
                "Use advanced AI/ML approaches",
                "Contribute to research knowledge base"
            ],
            min_depth=5,
            max_depth=8,
            user_instruction="Implement research-grade bearing fault analysis using novel techniques and latest scientific findings",
            expected_faults=["normal", "inner_race", "outer_race", "ball", "cage", "combination"],
            pre_analysis_notes=[
                "This expert-level demo integrates cutting-edge research",
                "Novel techniques and advanced AI methods will be employed",
                "The analysis contributes to research knowledge advancement"
            ],
            key_concepts=[
                "Research integration", "Novel techniques", "Advanced AI", "Knowledge contribution"
            ],
            difficulty_factors={
                "signal_complexity": "very_high", 
                "fault_types": 6,
                "processing_depth": "very_deep",
                "research_integration": True,
                "novel_methods": True
            }
        )
        
        return configs
    
    def _build_synthetic_data_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build synthetic data generation configurations for demos without real data"""
        
        return {
            "simple_bearing": {
                "name": "Simple Bearing Signals",
                "description": "Basic synthetic bearing vibration signals",
                "signals": {
                    "normal": {
                        "sampling_rate": 10000,
                        "duration": 1.0,
                        "components": [
                            {"type": "sine", "frequency": 60, "amplitude": 1.0},
                            {"type": "noise", "amplitude": 0.1}
                        ]
                    },
                    "inner_race": {
                        "sampling_rate": 10000,
                        "duration": 1.0,
                        "components": [
                            {"type": "sine", "frequency": 60, "amplitude": 1.0},
                            {"type": "sine", "frequency": 157, "amplitude": 0.5},
                            {"type": "noise", "amplitude": 0.1}
                        ]
                    },
                    "outer_race": {
                        "sampling_rate": 10000,
                        "duration": 1.0,
                        "components": [
                            {"type": "sine", "frequency": 60, "amplitude": 1.0},
                            {"type": "sine", "frequency": 236, "amplitude": 0.4},
                            {"type": "noise", "amplitude": 0.1}
                        ]
                    }
                }
            },
            
            "complex_bearing": {
                "name": "Complex Bearing Signals",
                "description": "Advanced synthetic bearing signals with harmonics and modulation",
                "signals": {
                    "normal": {
                        "sampling_rate": 25600,
                        "duration": 2.0,
                        "components": [
                            {"type": "sine", "frequency": 60, "amplitude": 1.0},
                            {"type": "sine", "frequency": 120, "amplitude": 0.3},
                            {"type": "noise", "amplitude": 0.05}
                        ]
                    },
                    "inner_race": {
                        "sampling_rate": 25600,
                        "duration": 2.0, 
                        "components": [
                            {"type": "sine", "frequency": 60, "amplitude": 1.0},
                            {"type": "sine", "frequency": 157, "amplitude": 0.6},
                            {"type": "sine", "frequency": 314, "amplitude": 0.3},
                            {"type": "modulated", "carrier": 157, "modulation": 5, "amplitude": 0.2},
                            {"type": "noise", "amplitude": 0.08}
                        ]
                    }
                }
            }
        }
    
    def get_configuration(self, config_name: str) -> Optional[DemoConfiguration]:
        """Get a specific demo configuration"""
        return self.configurations.get(config_name)
    
    def list_configurations(self, level: Optional[LearningLevel] = None, 
                           analysis_type: Optional[AnalysisType] = None) -> List[str]:
        """List available configurations, optionally filtered by level or type"""
        
        configs = []
        for name, config in self.configurations.items():
            if level and config.learning_level != level:
                continue
            if analysis_type and config.analysis_type != analysis_type:
                continue
            configs.append(name)
        
        return sorted(configs)
    
    def get_learning_progression(self) -> List[str]:
        """Get recommended learning progression of configurations"""
        
        progression = []
        
        # Beginner level
        progression.extend([
            "beginner_bearing_intro",
            "beginner_signal_basics"
        ])
        
        # Intermediate level
        progression.extend([
            "intermediate_multi_fault", 
            "intermediate_condition_monitoring"
        ])
        
        # Advanced level
        progression.extend([
            "advanced_predictive_maintenance",
            "advanced_multi_sensor"
        ])
        
        # Expert level
        progression.extend([
            "expert_research_integration"
        ])
        
        return progression
    
    def generate_synthetic_data(self, config_name: str) -> Dict[str, np.ndarray]:
        """Generate synthetic data for demo configuration"""
        
        if config_name not in self.synthetic_data_configs:
            raise ValueError(f"No synthetic data config for: {config_name}")
        
        config = self.synthetic_data_configs[config_name]
        synthetic_signals = {}
        
        for signal_name, signal_config in config["signals"].items():
            fs = signal_config["sampling_rate"]
            duration = signal_config["duration"]
            t = np.linspace(0, duration, int(fs * duration))
            
            signal = np.zeros_like(t)
            
            for component in signal_config["components"]:
                if component["type"] == "sine":
                    signal += component["amplitude"] * np.sin(2 * np.pi * component["frequency"] * t)
                elif component["type"] == "noise":
                    signal += component["amplitude"] * np.random.randn(len(t))
                elif component["type"] == "modulated":
                    carrier = component["carrier"]
                    mod_freq = component["modulation"]
                    amplitude = component["amplitude"]
                    envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
                    signal += amplitude * envelope * np.sin(2 * np.pi * carrier * t)
            
            synthetic_signals[signal_name] = signal
        
        return synthetic_signals
    
    def print_configuration_guide(self):
        """Print comprehensive guide to available configurations"""
        
        print("🎓 PHMGA Tutorial Configuration Guide")
        print("=" * 50)
        
        for level in LearningLevel:
            level_configs = self.list_configurations(level=level)
            if not level_configs:
                continue
                
            print(f"\n📚 {level.value.upper()} LEVEL ({len(level_configs)} configurations):")
            
            for config_name in level_configs:
                config = self.configurations[config_name]
                print(f"\n   🔹 {config.name}")
                print(f"      {config.description}")
                print(f"      Analysis: {config.analysis_type.value}")
                print(f"      Objectives: {len(config.learning_objectives)} learning goals")
                if config.expected_faults:
                    print(f"      Fault types: {', '.join(config.expected_faults)}")
        
        print(f"\n🎯 Recommended Learning Path:")
        progression = self.get_learning_progression()
        for i, config_name in enumerate(progression, 1):
            config = self.configurations[config_name]
            print(f"   {i}. {config.name}")
    
    def create_tutorial_sequence(self, level: LearningLevel) -> List[DemoConfiguration]:
        """Create a sequence of configurations for a specific learning level"""
        
        level_configs = []
        for config_name in self.list_configurations(level=level):
            level_configs.append(self.configurations[config_name])
        
        # Sort by complexity (roughly by max_depth and expected_faults count)
        level_configs.sort(key=lambda c: (c.max_depth, len(c.expected_faults)))
        
        return level_configs


def create_demo_manager() -> DemoConfigurationManager:
    """Create and return a demo configuration manager"""
    return DemoConfigurationManager()


def demonstrate_configurations():
    """Demonstrate the configuration system"""
    
    print("🎛️ Demo Configuration System")
    print("=" * 35)
    
    manager = create_demo_manager()
    manager.print_configuration_guide()
    
    print(f"\n🧪 Synthetic Data Available:")
    for config_name, config in manager.synthetic_data_configs.items():
        print(f"   • {config['name']}: {config['description']}")
        print(f"     Signals: {', '.join(config['signals'].keys())}")
    
    return manager


if __name__ == "__main__":
    demonstrate_configurations()