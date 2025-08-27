"""
Complete PHMGA System Integration

Educational wrapper for the production PHMGA system.
Integrates all tutorial concepts with the real production system from src/.
"""

import sys
import os
import time
import yaml
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Import production PHMGA components
from states.phm_states import PHMState, DAGState, InputData, ProcessedData
from phm_outer_graph import build_builder_graph, build_executor_graph
from utils import initialize_state, save_state, load_state, generate_final_report
from agents.reflect_agent import get_dag_depth
from tools.signal_processing_schemas import OP_REGISTRY, list_available_operators

# Import LangGraph for workflow management
from langgraph.graph import StateGraph


@dataclass
class PHMGAConfig:
    """Configuration for educational PHMGA system"""
    
    # Core PHMGA Configuration
    llm_provider: str = "google"  # or "openai" 
    llm_model: str = "gemini-2.0-flash-exp"
    min_depth: int = 2  # Tutorial-friendly depth
    max_depth: int = 4  # Tutorial-friendly depth
    
    # Tutorial-specific settings
    tutorial_mode: bool = True
    verbose_output: bool = True
    save_intermediate_results: bool = True
    
    # Signal processing settings
    sampling_rate: Optional[float] = None
    fault_classes: List[str] = field(default_factory=lambda: ["normal", "inner_race", "outer_race", "ball"])
    confidence_threshold: float = 0.7
    alert_threshold: float = 0.8
    
    @classmethod
    def for_tutorial(cls) -> "PHMGAConfig":
        """Create tutorial-friendly configuration"""
        return cls(
            tutorial_mode=True,
            verbose_output=True,
            min_depth=2,
            max_depth=4,
            save_intermediate_results=True
        )
    
    @classmethod
    def for_production(cls) -> "PHMGAConfig":
        """Create production-optimized configuration"""
        return cls(
            tutorial_mode=False,
            verbose_output=False,
            min_depth=4,
            max_depth=8,
            save_intermediate_results=False
        )


class PHMGASystem:
    """
    Educational wrapper for the production PHMGA system.
    
    This class provides a simplified interface to the sophisticated 
    production PHMGA system while maintaining educational clarity.
    """
    
    def __init__(self, config: PHMGAConfig):
        self.config = config
        self.session_id = f"tutorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize production system components
        self.builder_graph = None
        self.executor_graph = None
        self.current_state = None
        
        # Tutorial tracking
        self.processing_history = []
        self.operator_usage = {}
        self.performance_metrics = {}
        
        self._initialize_system()
        
        if config.verbose_output:
            print(f"🎓 PHMGA Tutorial System initialized")
            print(f"   Session ID: {self.session_id}")
            print(f"   Available operators: {len(OP_REGISTRY)}")
    
    def _initialize_system(self):
        """Initialize production PHMGA system components"""
        try:
            # Build production LangGraph workflows
            self.builder_graph = build_builder_graph()
            self.executor_graph = build_executor_graph()
            
            # Show available operators for educational purposes
            if self.config.tutorial_mode:
                self._display_available_operators()
                
        except Exception as e:
            print(f"⚠️ System initialization error: {e}")
    
    def _display_available_operators(self):
        """Display available signal processing operators for educational purposes"""
        if self.config.verbose_output:
            print("\n🔧 Available Signal Processing Operators:")
            operators_info = list_available_operators()
            for category, ops in operators_info.items():
                print(f"   {category.upper()}:")
                for op_name in ops:
                    op_class = OP_REGISTRY.get(op_name)
                    if op_class:
                        print(f"      • {op_name}: {getattr(op_class, 'description', 'Signal processing operator')}")
    
    def initialize_case(self, user_instruction: str, metadata_path: str, h5_path: str, 
                       ref_ids: List[int], test_ids: List[int], case_name: str = "tutorial_case") -> PHMState:
        """
        Initialize a new PHM analysis case using production system.
        
        Args:
            user_instruction: Description of the analysis objective
            metadata_path: Path to signal metadata Excel file
            h5_path: Path to HDF5 signal data file
            ref_ids: Reference signal IDs for healthy/normal condition
            test_ids: Test signal IDs for analysis
            case_name: Name for this case study
            
        Returns:
            Initialized PHMState object
        """
        
        if self.config.verbose_output:
            print(f"🏗️ Initializing PHM Case: {case_name}")
            print(f"   User instruction: {user_instruction}")
            print(f"   Reference IDs: {ref_ids}")
            print(f"   Test IDs: {test_ids}")
        
        try:
            # Use production initialization function
            initial_state = initialize_state(
                user_instruction=user_instruction,
                metadata_path=metadata_path,
                h5_path=h5_path,
                ref_ids=ref_ids,
                test_ids=test_ids,
                case_name=case_name
            )
            
            # Configure for tutorial settings
            initial_state.llm_provider = self.config.llm_provider
            initial_state.llm_model = self.config.llm_model
            initial_state.min_depth = self.config.min_depth
            initial_state.max_depth = self.config.max_depth
            
            self.current_state = initial_state
            
            if self.config.verbose_output:
                print(f"✅ Case initialized successfully")
                print(f"   Signal channels: {initial_state.dag_state.channels}")
                print(f"   Initial nodes: {len(initial_state.dag_state.nodes)}")
            
            return initial_state
            
        except Exception as e:
            print(f"❌ Case initialization failed: {e}")
            raise
    
    def build_processing_dag(self, initial_state: PHMState = None) -> PHMState:
        """
        Build signal processing DAG using production builder workflow.
        
        Args:
            initial_state: Initial PHM state (uses current_state if None)
            
        Returns:
            State with completed DAG
        """
        
        state = initial_state or self.current_state
        if not state:
            raise ValueError("No state available. Initialize a case first.")
        
        if self.config.verbose_output:
            print(f"\n🕸️ Building Signal Processing DAG")
            print(f"   Initial depth: {get_dag_depth(state.dag_state)}")
            print(f"   Target depth range: {self.config.min_depth}-{self.config.max_depth}")
        
        # Track DAG building process for educational purposes
        build_history = []
        
        try:
            built_state = state.model_copy(deep=True)
            iteration = 0
            
            # Use production builder graph
            while True:
                iteration += 1
                if self.config.verbose_output:
                    print(f"\n   Building Iteration {iteration}")
                
                # Execute one iteration of the builder graph
                thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                
                # Track state before iteration
                pre_iteration_nodes = len(built_state.dag_state.nodes)
                pre_iteration_leaves = built_state.dag_state.leaves.copy()
                
                # Execute builder iteration
                for event in self.builder_graph.stream(built_state, config=thread_config):
                    for node_name, state_update in event.items():
                        if self.config.verbose_output:
                            print(f"      Executing: {node_name}")
                        
                        if state_update is not None:
                            for key, value in state_update.items():
                                setattr(built_state, key, value)
                
                # Track what happened in this iteration
                post_iteration_nodes = len(built_state.dag_state.nodes)
                post_iteration_leaves = built_state.dag_state.leaves.copy()
                
                iteration_info = {
                    "iteration": iteration,
                    "nodes_added": post_iteration_nodes - pre_iteration_nodes,
                    "nodes_before": pre_iteration_nodes,
                    "nodes_after": post_iteration_nodes,
                    "leaves_before": pre_iteration_leaves,
                    "leaves_after": post_iteration_leaves,
                    "depth": get_dag_depth(built_state.dag_state)
                }
                build_history.append(iteration_info)
                
                if self.config.verbose_output:
                    print(f"      Added {iteration_info['nodes_added']} nodes")
                    print(f"      Current depth: {iteration_info['depth']}")
                    print(f"      Current leaves: {post_iteration_leaves}")
                
                # Check stopping conditions
                current_depth = get_dag_depth(built_state.dag_state)
                
                if current_depth >= self.config.max_depth:
                    if self.config.verbose_output:
                        print(f"   ✅ Reached max depth {self.config.max_depth}")
                    break
                
                if current_depth < self.config.min_depth:
                    if self.config.verbose_output:
                        print(f"   📈 Below min depth {self.config.min_depth}, continuing...")
                    built_state.needs_revision = True
                
                if not built_state.needs_revision:
                    if self.config.verbose_output:
                        print(f"   ✅ Reflect agent approved DAG")
                    break
            
            # Store build history for educational review
            self.processing_history.append({
                "phase": "dag_building",
                "iterations": build_history,
                "final_depth": get_dag_depth(built_state.dag_state),
                "total_nodes": len(built_state.dag_state.nodes),
                "final_leaves": built_state.dag_state.leaves
            })
            
            self.current_state = built_state
            
            if self.config.verbose_output:
                print(f"\n✅ DAG Building Complete")
                print(f"   Total iterations: {iteration}")
                print(f"   Final depth: {get_dag_depth(built_state.dag_state)}")
                print(f"   Total nodes: {len(built_state.dag_state.nodes)}")
                print(f"   Final leaves: {built_state.dag_state.leaves}")
            
            return built_state
            
        except Exception as e:
            print(f"❌ DAG building failed: {e}")
            raise
    
    def execute_analysis(self, built_state: PHMState = None) -> PHMState:
        """
        Execute analysis using production executor workflow.
        
        Args:
            built_state: State with completed DAG (uses current_state if None)
            
        Returns:
            State with analysis results
        """
        
        state = built_state or self.current_state
        if not state:
            raise ValueError("No built state available. Build a DAG first.")
        
        if self.config.verbose_output:
            print(f"\n🔬 Executing Analysis")
            print(f"   DAG nodes: {len(state.dag_state.nodes)}")
            print(f"   Processing leaves: {state.dag_state.leaves}")
        
        try:
            # Execute production executor graph
            thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            final_state = state.model_copy(deep=True)
            execution_stages = []
            
            for event in self.executor_graph.stream(final_state, config=thread_config):
                for node_name, state_update in event.items():
                    stage_start = time.time()
                    
                    if self.config.verbose_output:
                        print(f"   Executing stage: {node_name}")
                    
                    if state_update is not None:
                        for key, value in state_update.items():
                            setattr(final_state, key, value)
                    
                    execution_stages.append({
                        "stage": node_name,
                        "execution_time": time.time() - stage_start,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Store execution history
            self.processing_history.append({
                "phase": "analysis_execution", 
                "stages": execution_stages,
                "ml_results_available": bool(final_state.ml_results),
                "insights_generated": len(final_state.insights),
                "final_report_available": bool(final_state.final_report)
            })
            
            self.current_state = final_state
            
            if self.config.verbose_output:
                print(f"\n✅ Analysis Execution Complete")
                print(f"   Stages executed: {len(execution_stages)}")
                print(f"   ML results: {'Available' if final_state.ml_results else 'None'}")
                print(f"   Insights: {len(final_state.insights)}")
                print(f"   Report: {'Generated' if final_state.final_report else 'None'}")
            
            return final_state
            
        except Exception as e:
            print(f"❌ Analysis execution failed: {e}")
            raise
    
    def run_complete_analysis(self, user_instruction: str, metadata_path: str, h5_path: str,
                             ref_ids: List[int], test_ids: List[int], 
                             case_name: str = "tutorial_analysis") -> PHMState:
        """
        Run complete end-to-end analysis using production PHMGA system.
        
        Args:
            user_instruction: Analysis objective description
            metadata_path: Path to signal metadata
            h5_path: Path to signal data
            ref_ids: Reference (healthy) signal IDs
            test_ids: Test signal IDs for analysis
            case_name: Name for the analysis case
            
        Returns:
            Complete analysis results
        """
        
        if self.config.verbose_output:
            print(f"🚀 Running Complete PHMGA Analysis")
            print(f"   Case: {case_name}")
            print("=" * 50)
        
        analysis_start = time.time()
        
        try:
            # Phase 1: Initialize case
            print(f"\n📋 Phase 1: Case Initialization")
            initial_state = self.initialize_case(
                user_instruction=user_instruction,
                metadata_path=metadata_path, 
                h5_path=h5_path,
                ref_ids=ref_ids,
                test_ids=test_ids,
                case_name=case_name
            )
            
            # Phase 2: Build processing DAG
            print(f"\n🕸️ Phase 2: DAG Construction") 
            built_state = self.build_processing_dag(initial_state)
            
            # Phase 3: Execute analysis
            print(f"\n🔬 Phase 3: Analysis Execution")
            final_state = self.execute_analysis(built_state)
            
            total_time = time.time() - analysis_start
            
            if self.config.verbose_output:
                print(f"\n🎯 Complete Analysis Finished")
                print(f"   Total time: {total_time:.2f} seconds")
                print(f"   Final DAG depth: {get_dag_depth(final_state.dag_state)}")
                print(f"   ML results: {'✅' if final_state.ml_results else '❌'}")
                print(f"   Final report: {'✅' if final_state.final_report else '❌'}")
            
            # Store overall performance metrics
            self.performance_metrics[case_name] = {
                "total_execution_time": total_time,
                "dag_depth": get_dag_depth(final_state.dag_state),
                "total_nodes": len(final_state.dag_state.nodes),
                "insights_generated": len(final_state.insights),
                "ml_results_available": bool(final_state.ml_results),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return final_state
            
        except Exception as e:
            print(f"❌ Complete analysis failed: {e}")
            raise
    
    def save_results(self, state: PHMState, save_path: str):
        """Save analysis results using production save function"""
        try:
            save_state(state, save_path)
            if self.config.verbose_output:
                print(f"💾 Results saved to: {save_path}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
            raise
    
    def load_results(self, save_path: str) -> PHMState:
        """Load analysis results using production load function"""
        try:
            loaded_state = load_state(save_path)
            if loaded_state:
                self.current_state = loaded_state
                if self.config.verbose_output:
                    print(f"📂 Results loaded from: {save_path}")
                return loaded_state
            else:
                raise ValueError("Failed to load state")
        except Exception as e:
            print(f"❌ Load failed: {e}")
            raise
    
    def generate_report(self, state: PHMState, report_path: str):
        """Generate final report using production function"""
        try:
            generate_final_report(state, report_path)
            if self.config.verbose_output:
                print(f"📄 Report generated: {report_path}")
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            raise
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing history for educational review"""
        
        return {
            "session_id": self.session_id,
            "config": self.config.__dict__,
            "processing_history": self.processing_history,
            "operator_usage": self.operator_usage,
            "performance_metrics": self.performance_metrics,
            "current_state_summary": {
                "available": self.current_state is not None,
                "case_name": self.current_state.case_name if self.current_state else None,
                "dag_nodes": len(self.current_state.dag_state.nodes) if self.current_state else 0,
                "dag_depth": get_dag_depth(self.current_state.dag_state) if self.current_state else 0
            } if self.current_state else {"available": False}
        }


def create_tutorial_system(config_type: str = "tutorial") -> PHMGASystem:
    """
    Create a PHMGA system configured for educational use.
    
    Args:
        config_type: "tutorial", "production", or custom PHMGAConfig
        
    Returns:
        Configured PHMGA system
    """
    
    if config_type == "tutorial":
        config = PHMGAConfig.for_tutorial()
    elif config_type == "production":
        config = PHMGAConfig.for_production()
    else:
        config = PHMGAConfig()
    
    return PHMGASystem(config)


def demo_real_case_execution():
    """
    Demonstrate real case execution using production system data.
    This mirrors src/cases/case1.py but with educational output.
    """
    
    print("🎓 PHMGA Tutorial System - Real Case Demonstration")
    print("=" * 60)
    
    # Create tutorial system
    phmga = create_tutorial_system("tutorial")
    
    # Show system capabilities
    summary = phmga.get_processing_summary()
    print(f"\n📊 System Summary:")
    print(f"   Session ID: {summary['session_id']}")
    print(f"   Tutorial mode: {summary['config']['tutorial_mode']}")
    print(f"   Available operators: {len(OP_REGISTRY)}")
    
    print(f"\n💡 This system integrates with the production PHMGA architecture:")
    print(f"   • Real LangGraph workflows from src/phm_outer_graph.py")
    print(f"   • Production agents from src/agents/")
    print(f"   • Real signal processing operators from src/tools/")
    print(f"   • Actual PHMState management from src/states/")
    
    return phmga


if __name__ == "__main__":
    demo_real_case_execution()