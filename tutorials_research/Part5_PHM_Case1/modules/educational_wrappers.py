"""
Educational Wrappers Module

Provides tutorial-friendly wrappers around complex production PHMGA components.
Simplifies interfaces and adds educational context for learning purposes.
"""

import sys
import os
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
from contextlib import contextmanager

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Production imports
from states.phm_states import PHMState, DAGState, InputData, ProcessedData, SimilarityNode, DataSetNode, OutputNode
from tools.signal_processing_schemas import OP_REGISTRY, PHMOperator
from agents.reflect_agent import get_dag_depth
from utils import initialize_state, save_state, load_state


class EducationalLogger:
    """Educational logging with tutorial-friendly output"""
    
    def __init__(self, verbose: bool = True, show_timing: bool = True):
        self.verbose = verbose
        self.show_timing = show_timing
        self.step_count = 0
        self.start_times = {}
    
    @contextmanager
    def step(self, description: str, details: str = None):
        """Context manager for educational step tracking"""
        self.step_count += 1
        step_id = f"step_{self.step_count}"
        
        if self.verbose:
            print(f"\n📝 Step {self.step_count}: {description}")
            if details:
                print(f"   💡 {details}")
        
        if self.show_timing:
            self.start_times[step_id] = time.time()
        
        try:
            yield self.step_count
        finally:
            if self.show_timing and step_id in self.start_times:
                elapsed = time.time() - self.start_times[step_id]
                if self.verbose:
                    print(f"   ⏱️  Completed in {elapsed:.2f}s")
    
    def info(self, message: str, indent: int = 1):
        """Log informational message"""
        if self.verbose:
            prefix = "   " * indent
            print(f"{prefix}ℹ️  {message}")
    
    def success(self, message: str, indent: int = 1):
        """Log success message"""
        if self.verbose:
            prefix = "   " * indent
            print(f"{prefix}✅ {message}")
    
    def warning(self, message: str, indent: int = 1):
        """Log warning message"""
        if self.verbose:
            prefix = "   " * indent
            print(f"{prefix}⚠️  {message}")
    
    def error(self, message: str, indent: int = 1):
        """Log error message"""
        if self.verbose:
            prefix = "   " * indent
            print(f"{prefix}❌ {message}")


@dataclass
class EducationalConfig:
    """Configuration for educational components"""
    verbose: bool = True
    show_timing: bool = True
    show_intermediate_results: bool = True
    max_display_items: int = 5
    plot_results: bool = False
    save_plots: bool = False
    plot_directory: str = "plots"
    explain_concepts: bool = True


class StateExplainer:
    """Explains PHMGA state components in educational terms"""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = EducationalLogger(config.verbose, config.show_timing)
    
    def explain_state(self, state: PHMState) -> None:
        """Provide educational explanation of PHM state"""
        if not self.config.explain_concepts:
            return
            
        print(f"\n🧠 Understanding PHMState")
        print(f"=" * 40)
        
        self.logger.info(f"Case Name: '{state.case_name}'")
        self.logger.info(f"User Instruction: '{state.user_instruction}'")
        self.logger.info(f"LLM Configuration: {state.llm_provider} / {state.llm_model}")
        
        print(f"\n📊 DAG Structure:")
        self._explain_dag_state(state.dag_state)
        
        print(f"\n🔧 Processing Configuration:")
        self.logger.info(f"Depth Range: {state.min_depth} - {state.max_depth}")
        self.logger.info(f"Needs Revision: {state.needs_revision}")
        
        if state.insights:
            print(f"\n💡 Analysis Insights:")
            for i, insight in enumerate(state.insights[:self.config.max_display_items]):
                self.logger.info(f"Insight {i+1}: {insight}")
    
    def _explain_dag_state(self, dag_state: DAGState) -> None:
        """Explain DAG state structure"""
        self.logger.info(f"Signal Channels: {dag_state.channels}")
        self.logger.info(f"Total Nodes: {len(dag_state.nodes)}")
        self.logger.info(f"Current Leaves: {dag_state.leaves}")
        self.logger.info(f"DAG Depth: {get_dag_depth(dag_state)}")
        
        if self.config.show_intermediate_results:
            print(f"\n   🏗️ Node Breakdown:")
            node_types = {}
            for node in dag_state.nodes.values():
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            for node_type, count in node_types.items():
                self.logger.info(f"{node_type}: {count} nodes", indent=2)


class OperatorExplainer:
    """Explains signal processing operators in educational terms"""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = EducationalLogger(config.verbose, config.show_timing)
    
    def explain_available_operators(self) -> None:
        """Explain all available operators"""
        print(f"\n🔧 Signal Processing Operators")
        print(f"=" * 40)
        
        # Group operators by type
        operator_groups = {}
        for op_name, op_class in OP_REGISTRY.items():
            base_classes = [cls.__name__ for cls in op_class.__bases__]
            category = "Unknown"
            
            if "ExpandOp" in base_classes:
                category = "EXPAND - Signal Expansion"
            elif "TransformOp" in base_classes:
                category = "TRANSFORM - Signal Transformation"
            elif "AggregateOp" in base_classes:
                category = "AGGREGATE - Feature Aggregation"
            elif "DecisionOp" in base_classes:
                category = "DECISION - Processing Decisions"
            elif "MultiVariableOp" in base_classes:
                category = "MULTIVARIABLE - Multi-Signal Operations"
            
            if category not in operator_groups:
                operator_groups[category] = []
            operator_groups[category].append((op_name, op_class))
        
        for category, operators in operator_groups.items():
            print(f"\n📁 {category} ({len(operators)} operators):")
            for op_name, op_class in operators[:self.config.max_display_items]:
                description = getattr(op_class, 'description', 'Signal processing operator')
                self.logger.info(f"{op_name}: {description}", indent=1)
            
            if len(operators) > self.config.max_display_items:
                remaining = len(operators) - self.config.max_display_items
                self.logger.info(f"... and {remaining} more operators", indent=1)
    
    def explain_operator(self, operator_name: str) -> None:
        """Explain a specific operator in detail"""
        if operator_name not in OP_REGISTRY:
            self.logger.error(f"Operator '{operator_name}' not found in registry")
            return
        
        op_class = OP_REGISTRY[operator_name]
        
        print(f"\n🔍 Operator Deep Dive: {operator_name}")
        print(f"=" * 50)
        
        self.logger.info(f"Class: {op_class.__name__}")
        self.logger.info(f"Description: {getattr(op_class, 'description', 'No description available')}")
        
        # Show base classes to understand operator type
        base_classes = [cls.__name__ for cls in op_class.__bases__ if cls.__name__ != 'object']
        if base_classes:
            self.logger.info(f"Type: {', '.join(base_classes)}")
        
        # Show required parameters if available
        if hasattr(op_class, '__init__'):
            try:
                import inspect
                sig = inspect.signature(op_class.__init__)
                params = [p for p in sig.parameters.keys() if p != 'self']
                if params:
                    self.logger.info(f"Parameters: {', '.join(params)}")
            except:
                pass


class WorkflowExplainer:
    """Explains PHMGA workflows in educational terms"""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = EducationalLogger(config.verbose, config.show_timing)
    
    def explain_builder_workflow(self) -> None:
        """Explain the DAG builder workflow"""
        print(f"\n🏗️ DAG Builder Workflow")
        print(f"=" * 30)
        
        workflow_steps = [
            ("plan", "Plan Agent generates next processing steps based on current DAG state"),
            ("execute", "Execute Agent applies selected operators to extend the DAG"),
            ("reflect", "Reflect Agent evaluates DAG quality and decides whether to continue")
        ]
        
        for i, (step, description) in enumerate(workflow_steps, 1):
            print(f"\n{i}. {step.upper()} Stage:")
            self.logger.info(description, indent=1)
        
        print(f"\n🔄 Loop Logic:")
        self.logger.info("The workflow loops until DAG reaches desired depth or quality")
        self.logger.info("Reflect agent sets 'needs_revision' flag to control loop continuation")
    
    def explain_executor_workflow(self) -> None:
        """Explain the analysis executor workflow"""
        print(f"\n🔬 Analysis Executor Workflow")
        print(f"=" * 35)
        
        workflow_steps = [
            ("inquire", "Inquirer Agent computes similarities between signals using various metrics"),
            ("prepare", "Dataset Preparer creates ML-ready datasets from processed signals"),
            ("train", "ML Agent trains shallow learning models for fault classification"),
            ("report", "Report Agent generates comprehensive analysis report with insights")
        ]
        
        for i, (step, description) in enumerate(workflow_steps, 1):
            print(f"\n{i}. {step.upper()} Stage:")
            self.logger.info(description, indent=1)


class EducationalDAGVisualization:
    """Educational DAG visualization tools"""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = EducationalLogger(config.verbose, config.show_timing)
    
    def visualize_dag_evolution(self, dag_states: List[DAGState], title: str = "DAG Evolution") -> None:
        """Visualize how DAG evolves over iterations"""
        if not self.config.plot_results:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            iterations = list(range(len(dag_states)))
            node_counts = [len(dag.nodes) for dag in dag_states]
            depths = [get_dag_depth(dag) for dag in dag_states]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Node count evolution
            ax1.plot(iterations, node_counts, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Number of Nodes')
            ax1.set_title('DAG Size Evolution')
            ax1.grid(True, alpha=0.3)
            
            # Depth evolution  
            ax2.plot(iterations, depths, 'r-s', linewidth=2, markersize=6)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('DAG Depth')
            ax2.set_title('DAG Depth Evolution')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if self.config.save_plots:
                os.makedirs(self.config.plot_directory, exist_ok=True)
                plt.savefig(f"{self.config.plot_directory}/dag_evolution.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")
    
    def show_node_type_distribution(self, dag_state: DAGState, title: str = "Node Type Distribution") -> None:
        """Show distribution of node types in current DAG"""
        if not self.config.plot_results:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Count node types
            node_types = {}
            for node in dag_state.nodes.values():
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if not node_types:
                return
            
            # Create pie chart
            plt.figure(figsize=(8, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(node_types)))
            plt.pie(node_types.values(), labels=node_types.keys(), colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(title, fontsize=14, fontweight='bold')
            
            if self.config.save_plots:
                os.makedirs(self.config.plot_directory, exist_ok=True)
                plt.savefig(f"{self.config.plot_directory}/node_distribution.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")


class EducationalCaseRunner:
    """Educational wrapper for running PHMGA cases with learning support"""
    
    def __init__(self, config: EducationalConfig = None):
        self.config = config or EducationalConfig()
        self.logger = EducationalLogger(self.config.verbose, self.config.show_timing)
        self.state_explainer = StateExplainer(self.config)
        self.operator_explainer = OperatorExplainer(self.config)
        self.workflow_explainer = WorkflowExplainer(self.config)
        self.visualizer = EducationalDAGVisualization(self.config)
        
        # Educational tracking
        self.iteration_history = []
        self.performance_metrics = {}
        
    def initialize_case_with_explanation(self, user_instruction: str, metadata_path: str, 
                                       h5_path: str, ref_ids: List[int], test_ids: List[int],
                                       case_name: str = "educational_case") -> PHMState:
        """Initialize case with educational explanations"""
        
        with self.logger.step("Case Initialization", "Setting up PHM analysis with real signal data"):
            
            print(f"\n📋 Educational Case Setup")
            print(f"=" * 30)
            
            self.logger.info(f"Case: {case_name}")
            self.logger.info(f"Objective: {user_instruction}")
            self.logger.info(f"Reference signals (healthy): {ref_ids}")
            self.logger.info(f"Test signals (analyze): {test_ids}")
            
            # Initialize using production function
            try:
                initial_state = initialize_state(
                    user_instruction=user_instruction,
                    metadata_path=metadata_path,
                    h5_path=h5_path,
                    ref_ids=ref_ids,
                    test_ids=test_ids,
                    case_name=case_name
                )
                
                self.logger.success("Case initialized successfully")
                
                if self.config.explain_concepts:
                    self.state_explainer.explain_state(initial_state)
                
                return initial_state
                
            except Exception as e:
                self.logger.error(f"Case initialization failed: {e}")
                print(f"\n🎓 Educational Note:")
                print(f"   This error typically occurs when:")
                print(f"   • Signal data files are missing or corrupted")
                print(f"   • File paths are incorrect")  
                print(f"   • Signal IDs don't exist in the dataset")
                print(f"   • Required dependencies are not installed")
                raise
    
    def explain_system_architecture(self):
        """Provide comprehensive explanation of PHMGA system architecture"""
        
        print(f"\n🏛️ PHMGA System Architecture")
        print(f"=" * 40)
        
        print(f"\n🎓 Educational Overview:")
        print(f"   The PHMGA system uses a two-phase approach:")
        print(f"   1. BUILD Phase: Constructs signal processing DAG")
        print(f"   2. EXECUTE Phase: Runs analysis on completed DAG")
        
        self.workflow_explainer.explain_builder_workflow()
        self.workflow_explainer.explain_executor_workflow()
        
        print(f"\n🔧 Available Processing Capabilities:")
        self.operator_explainer.explain_available_operators()
    
    def run_educational_demo(self):
        """Run a complete educational demonstration"""
        
        print(f"\n🎓 PHMGA Educational System Demo")
        print(f"=" * 50)
        
        with self.logger.step("System Architecture Overview", "Understanding PHMGA components"):
            self.explain_system_architecture()
        
        with self.logger.step("Operator Deep Dive", "Exploring signal processing operators"):
            # Show a few example operators in detail
            example_operators = list(OP_REGISTRY.keys())[:3]
            for op_name in example_operators:
                self.operator_explainer.explain_operator(op_name)
        
        self.logger.success("Educational demo completed!")
        print(f"\n💡 Next Steps:")
        print(f"   • Use initialize_case_with_explanation() to start a real case")
        print(f"   • Run build_dag_with_explanation() to see DAG construction")
        print(f"   • Use execute_analysis_with_explanation() for full analysis")


def create_educational_system(verbose: bool = True, show_plots: bool = True) -> EducationalCaseRunner:
    """Create an educational PHMGA system with tutorial-friendly configuration"""
    
    config = EducationalConfig(
        verbose=verbose,
        show_timing=True,
        show_intermediate_results=True,
        plot_results=show_plots,
        save_plots=True,
        explain_concepts=True
    )
    
    return EducationalCaseRunner(config)


def demo_educational_wrappers():
    """Demonstrate the educational wrappers"""
    
    print("🎓 Educational Wrappers Demo")
    print("=" * 40)
    
    # Create educational system
    edu_system = create_educational_system(verbose=True, show_plots=False)
    
    # Run demo
    edu_system.run_educational_demo()
    
    return edu_system


if __name__ == "__main__":
    demo_educational_wrappers()