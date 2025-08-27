"""
Case 1 Tutorial: Educational Bearing Fault Diagnosis

This module provides an educational version of the real case1.py from src/cases/,
demonstrating the complete PHMGA workflow with tutorial-friendly explanations.
"""

import sys
import os
import time
import yaml
import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Import production PHMGA components
from states.phm_states import PHMState, DAGState
from phm_outer_graph import build_builder_graph, build_executor_graph
from utils import initialize_state, save_state, load_state, generate_final_report
from agents.reflect_agent import get_dag_depth

# Import tutorial components
from educational_wrappers import EducationalLogger, EducationalConfig
from demo_configurations import create_demo_manager


class Case1Tutorial:
    """
    Educational wrapper for the real Case 1 bearing fault diagnosis.
    
    This mirrors the functionality of src/cases/case1.py but with:
    - Educational explanations at each step
    - Progress visualization and monitoring
    - Beginner-friendly error messages
    - Step-by-step workflow breakdown
    """
    
    def __init__(self, config: EducationalConfig = None):
        self.config = config or EducationalConfig()
        self.logger = EducationalLogger(self.config.verbose, self.config.show_timing)
        
        # Initialize production components
        self.builder_graph = None
        self.executor_graph = None
        self.case_history = []
        
        self._initialize_graphs()
        
    def _initialize_graphs(self):
        """Initialize production LangGraph workflows"""
        try:
            self.builder_graph = build_builder_graph()
            self.executor_graph = build_executor_graph()
            
            if self.config.verbose:
                print("✅ Production LangGraph workflows initialized")
                print("   • Builder Graph: plan → execute → reflect")
                print("   • Executor Graph: inquire → prepare → train → report")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize graphs: {e}")
            print("\n🎓 Educational Note:")
            print("   This error typically occurs when:")
            print("   • Required dependencies are not installed")
            print("   • LLM API keys are not configured")
            print("   • Production components cannot be imported")
            raise
    
    def run_educational_case(self, config_path: str = None) -> PHMState:
        """
        Run the educational version of Case 1 bearing fault diagnosis.
        
        This follows the same workflow as src/cases/case1.py but with educational explanations.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            
        Returns:
            Final PHMState with complete analysis results
        """
        
        print("🎓 CASE 1 TUTORIAL: BEARING FAULT DIAGNOSIS")
        print("=" * 60)
        print("This educational case demonstrates the complete PHMGA workflow")
        print("following the same pattern as the real src/cases/case1.py")
        
        case_start_time = time.time()
        
        # Step 1: Load Configuration
        with self.logger.step("Configuration Loading", "Reading analysis parameters from config file"):
            
            if config_path and os.path.exists(config_path):
                # Load real configuration
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    self.logger.success(f"Loaded configuration from {config_path}")
                    
                    # Show key configuration parameters
                    if self.config.show_intermediate_results:
                        print("   📋 Configuration Parameters:")
                        print(f"      • Case Name: {config.get('name', 'N/A')}")
                        print(f"      • User Instruction: {config.get('user_instruction', 'N/A')}")
                        print(f"      • Reference IDs: {config.get('ref_ids', [])}")
                        print(f"      • Test IDs: {config.get('test_ids', [])}")
                        print(f"      • State Save Path: {config.get('state_save_path', 'N/A')}")
                        
                        builder_cfg = config.get('builder', {})
                        print(f"      • Min Depth: {builder_cfg.get('min_depth', 0)}")
                        print(f"      • Max Depth: {builder_cfg.get('max_depth', float('inf'))}")
                
                except Exception as e:
                    self.logger.error(f"Failed to load configuration: {e}")
                    config = self._get_demo_configuration()
            else:
                # Use demo configuration
                self.logger.info("No configuration file provided, using demo configuration")
                config = self._get_demo_configuration()
        
        # Extract configuration parameters
        state_save_path = config['state_save_path']
        builder_cfg = config.get('builder', {})
        min_depth = builder_cfg.get('min_depth', 0)
        max_depth = builder_cfg.get('max_depth', float('inf'))
        
        # Step 2: Check for Existing State
        with self.logger.step("State Checking", "Looking for existing analysis results"):
            
            if os.path.exists(state_save_path):
                self.logger.info(f"Found existing state file at {state_save_path}")
                self.logger.info("Skipping builder workflow and loading existing results")
                
                try:
                    built_state = load_state(state_save_path)
                    if built_state is None:
                        raise ValueError("Failed to load state from file")
                    
                    self.logger.success("Successfully loaded existing state")
                    
                    if self.config.show_intermediate_results:
                        print("   📊 Loaded State Summary:")
                        print(f"      • Case Name: {built_state.case_name}")
                        print(f"      • DAG Nodes: {len(built_state.dag_state.nodes)}")
                        print(f"      • DAG Depth: {get_dag_depth(built_state.dag_state)}")
                        print(f"      • Signal Channels: {built_state.dag_state.channels}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load existing state: {e}")
                    self.logger.info("Will proceed with full workflow")
                    built_state = None
            else:
                self.logger.info("No existing state file found")
                self.logger.info("Will run complete workflow from initialization")
                built_state = None
        
        # Step 3: Initialize State (if needed)
        if built_state is None:
            with self.logger.step("State Initialization", "Setting up initial PHM analysis state"):
                
                print("   🏗️ Creating initial PHMState with signal data...")
                
                try:
                    initial_phm_state = initialize_state(
                        user_instruction=config['user_instruction'],
                        metadata_path=config['metadata_path'],
                        h5_path=config['h5_path'],
                        ref_ids=config['ref_ids'],
                        test_ids=config['test_ids'],
                        case_name=config['name']
                    )
                    
                    self.logger.success("Initial state created successfully")
                    
                    if self.config.show_intermediate_results:
                        print("   📊 Initial State Summary:")
                        print(f"      • Case: {initial_phm_state.case_name}")
                        print(f"      • Channels: {len(initial_phm_state.dag_state.channels)}")
                        print(f"      • Initial Nodes: {len(initial_phm_state.dag_state.nodes)}")
                        print(f"      • User Instruction: {initial_phm_state.user_instruction[:60]}...")
                
                except Exception as e:
                    self.logger.error(f"State initialization failed: {e}")
                    print("\n🎓 Educational Note:")
                    print("   In a real deployment, you would:")
                    print("   • Verify signal data files exist and are accessible")
                    print("   • Check metadata format and signal ID validity")
                    print("   • Ensure proper file permissions and paths")
                    raise
            
            # Step 4: Run DAG Builder Workflow
            with self.logger.step("DAG Builder Workflow", "Constructing signal processing pipeline"):
                
                print("   🕸️ Starting iterative DAG construction...")
                print("   This uses the real LangGraph builder workflow:")
                print("      • Plan Agent: Generates processing steps")
                print("      • Execute Agent: Applies signal processing operators")  
                print("      • Reflect Agent: Quality assessment and continuation")
                
                built_state = initial_phm_state.model_copy(deep=True)
                iteration = 0
                
                # Track DAG evolution for educational purposes
                dag_evolution = []
                
                while True:
                    iteration += 1
                    
                    with self.logger.step(f"Builder Iteration {iteration}", f"Plan-Execute-Reflect cycle"):
                        
                        # Record pre-iteration state
                        pre_nodes = len(built_state.dag_state.nodes)
                        pre_depth = get_dag_depth(built_state.dag_state)
                        pre_leaves = built_state.dag_state.leaves.copy()
                        
                        # Execute one builder iteration
                        thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                        
                        for event in self.builder_graph.stream(built_state, config=thread_config):
                            for node_name, state_update in event.items():
                                if self.config.verbose:
                                    self.logger.info(f"Executing: {node_name}")
                                
                                if state_update is not None:
                                    for key, value in state_update.items():
                                        setattr(built_state, key, value)
                        
                        # Record post-iteration state
                        post_nodes = len(built_state.dag_state.nodes)
                        post_depth = get_dag_depth(built_state.dag_state)
                        post_leaves = built_state.dag_state.leaves.copy()
                        
                        # Track evolution
                        iteration_info = {
                            "iteration": iteration,
                            "nodes_added": post_nodes - pre_nodes,
                            "depth": post_depth,
                            "leaves": post_leaves,
                            "timestamp": datetime.now().isoformat()
                        }
                        dag_evolution.append(iteration_info)
                        
                        if self.config.show_intermediate_results:
                            print(f"      📈 Progress: {pre_nodes} → {post_nodes} nodes, depth: {post_depth}")
                            print(f"      🍃 Current leaves: {post_leaves}")
                    
                    current_depth = get_dag_depth(built_state.dag_state)
                    
                    # Check stopping conditions (same logic as real case1.py)
                    if current_depth >= max_depth:
                        self.logger.success(f"Reached max depth {max_depth}")
                        break
                    
                    if current_depth < min_depth:
                        if self.config.verbose:
                            self.logger.info(f"Depth {current_depth} below min_depth {min_depth}, continuing...")
                        built_state.needs_revision = True
                    
                    if not built_state.needs_revision:
                        self.logger.success("Reflect agent approved DAG quality")
                        break
                
                self.logger.success(f"DAG construction completed in {iteration} iterations")
                
                if self.config.show_intermediate_results:
                    print("   📊 Final DAG Summary:")
                    print(f"      • Total Iterations: {iteration}")
                    print(f"      • Final Depth: {get_dag_depth(built_state.dag_state)}")
                    print(f"      • Total Nodes: {len(built_state.dag_state.nodes)}")
                    print(f"      • Final Leaves: {built_state.dag_state.leaves}")
                    
                    if built_state.dag_state.error_log:
                        print(f"      • Errors During Build: {len(built_state.dag_state.error_log)}")
                
                # Store evolution data
                self.case_history.append({
                    "phase": "dag_building",
                    "iterations": dag_evolution,
                    "final_state": {
                        "depth": get_dag_depth(built_state.dag_state),
                        "nodes": len(built_state.dag_state.nodes),
                        "leaves": built_state.dag_state.leaves
                    }
                })
                
                # Save the built state (same as real case1.py)
                try:
                    save_state(built_state, state_save_path)
                    self.logger.success(f"Built state saved to {state_save_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save state: {e}")
        
        # Step 5: Explain the Built DAG
        with self.logger.step("DAG Analysis", "Understanding the constructed processing pipeline"):
            
            if self.config.explain_concepts:
                print("   🧠 Understanding Your DAG:")
                print("   The system has automatically constructed a signal processing pipeline")
                print("   tailored to your bearing fault diagnosis requirements.")
                
                # Analyze node types
                node_types = {}
                for node in built_state.dag_state.nodes.values():
                    node_type = type(node).__name__
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print(f"\n   📊 Pipeline Composition:")
                for node_type, count in node_types.items():
                    print(f"      • {node_type}: {count} nodes")
                
                print(f"\n   🔄 Processing Flow:")
                print(f"      • Input signals are processed through {len(built_state.dag_state.nodes)} operations")
                print(f"      • Pipeline depth: {get_dag_depth(built_state.dag_state)} levels")
                print(f"      • Current outputs: {built_state.dag_state.leaves}")
        
        # NOTE: The real case1.py has the executor workflow commented out
        # We'll demonstrate this conceptually while noting it's not executed
        print("\n   💡 Educational Note: Executor Workflow")
        print("   In the real src/cases/case1.py, the executor workflow is commented out.")
        print("   Here's what it would do if enabled:")
        
        with self.logger.step("Executor Workflow (Conceptual)", "Analysis execution would proceed as follows"):
            
            print("   🔬 Phase 2: Analysis Execution (Conceptual)")
            print("      1. Inquire Agent: Compute signal similarities")
            print("      2. Dataset Preparer: Create ML-ready datasets")  
            print("      3. ML Agent: Train fault classification models")
            print("      4. Report Agent: Generate comprehensive reports")
            
            # Show what the workflow would produce
            print("\n   📊 Expected Results:")
            print("      • Signal similarity matrices")
            print("      • ML datasets with features and labels")
            print("      • Trained classification models")
            print("      • Fault diagnosis reports")
            print("      • Maintenance recommendations")
        
        # Calculate total time
        total_time = time.time() - case_start_time
        
        # Step 6: Final Summary
        with self.logger.step("Case Completion", "Summarizing analysis results"):
            
            print("   🎉 Case 1 Tutorial Successfully Completed!")
            print(f"      • Total execution time: {total_time:.2f} seconds")
            print(f"      • DAG construction: {'✅ Complete' if built_state else '❌ Failed'}")
            print(f"      • State persistence: {'✅ Saved' if os.path.exists(state_save_path) else '❌ Failed'}")
            
            if built_state:
                print(f"      • Signal processing pipeline: {len(built_state.dag_state.nodes)} operations")
                print(f"      • Analysis depth: {get_dag_depth(built_state.dag_state)} levels")
                
                if self.config.save_results:
                    results_summary = {
                        "case_name": built_state.case_name,
                        "execution_time": total_time,
                        "dag_summary": {
                            "nodes": len(built_state.dag_state.nodes),
                            "depth": get_dag_depth(built_state.dag_state),
                            "channels": built_state.dag_state.channels,
                            "leaves": built_state.dag_state.leaves
                        },
                        "case_history": self.case_history,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save tutorial results
                    results_path = "case1_tutorial_results.yaml"
                    try:
                        with open(results_path, 'w') as f:
                            yaml.dump(results_summary, f, default_flow_style=False)
                        self.logger.success(f"Tutorial results saved to {results_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save tutorial results: {e}")
        
        return built_state
    
    def _get_demo_configuration(self) -> Dict[str, Any]:
        """Get demo configuration when no config file is provided"""
        
        return {
            "name": "tutorial_bearing_case",
            "user_instruction": "Educational bearing fault diagnosis using real PHMGA system",
            "metadata_path": "demo/metadata.xlsx",  # Would need real files
            "h5_path": "demo/signals.h5",           # Would need real files
            "ref_ids": [1, 2, 3],                   # Demo reference IDs
            "test_ids": [4, 5, 6],                  # Demo test IDs
            "state_save_path": "tutorial_state.pkl",
            "report_path": "tutorial_report.md",
            "builder": {
                "min_depth": 2,
                "max_depth": 4
            }
        }
    
    def explain_case_architecture(self):
        """Provide detailed explanation of the case architecture"""
        
        print("🏗️ CASE 1 ARCHITECTURE EXPLANATION")
        print("=" * 45)
        
        print("\n📋 Real Case 1 Structure (src/cases/case1.py):")
        case_structure = [
            "1. Configuration Loading - YAML parameters",
            "2. State Management - Load existing or create new",
            "3. State Initialization - Signal data and metadata",
            "4. DAG Builder Workflow - Iterative pipeline construction",
            "5. Quality Control - Min/max depth and reflection",
            "6. State Persistence - Save results for later use",
            "7. Optional Executor - Analysis execution (commented out)"
        ]
        
        for step in case_structure:
            print(f"   {step}")
        
        print("\n🎓 Tutorial Enhancements:")
        tutorial_enhancements = [
            "Educational logging with step-by-step explanations",
            "Progress visualization and monitoring", 
            "Beginner-friendly error messages and guidance",
            "DAG evolution tracking for learning purposes",
            "Conceptual demonstration of executor workflow",
            "Results summary and tutorial completion tracking"
        ]
        
        for enhancement in tutorial_enhancements:
            print(f"   • {enhancement}")
        
        print("\n🔧 Production Integration:")
        print("   • Uses real LangGraph workflows from src/phm_outer_graph.py")
        print("   • Imports actual production agents from src/agents/")
        print("   • Utilizes production state management from src/states/")
        print("   • Follows exact same logic flow as production case")
        
        print("\n💡 Learning Value:")
        print("   This tutorial case demonstrates how production PHMGA")
        print("   systems work in practice, with the same workflows,")
        print("   agents, and state management used in real deployments.")


def run_case1_tutorial(config_path: str = None, verbose: bool = True) -> PHMState:
    """
    Convenience function to run Case 1 tutorial with default settings.
    
    Args:
        config_path: Optional path to YAML configuration
        verbose: Enable verbose educational output
        
    Returns:
        Final PHMState with analysis results
    """
    
    config = EducationalConfig(
        verbose=verbose,
        show_timing=True,
        show_intermediate_results=True,
        explain_concepts=True,
        save_results=True
    )
    
    tutorial = Case1Tutorial(config)
    
    # Show architecture explanation first
    if verbose:
        tutorial.explain_case_architecture()
        print("\n" + "=" * 60)
    
    # Run the educational case
    return tutorial.run_educational_case(config_path)


def demonstrate_case1_concepts():
    """Demonstrate the educational concepts in Case 1"""
    
    print("🎓 CASE 1 EDUCATIONAL DEMONSTRATION")
    print("=" * 50)
    
    print("\n📚 What You'll Learn:")
    learning_objectives = [
        "Real PHMGA system workflow execution",
        "LangGraph builder workflow (plan-execute-reflect)",
        "Dynamic DAG construction with intelligent agents",
        "Production state management and persistence", 
        "Quality control and stopping criteria",
        "Industrial bearing fault diagnosis pipeline"
    ]
    
    for i, objective in enumerate(learning_objectives, 1):
        print(f"   {i}. {objective}")
    
    print("\n🏭 Production System Integration:")
    print("   This tutorial uses the ACTUAL production components:")
    print("   • src/phm_outer_graph.py - Real LangGraph workflows")
    print("   • src/agents/ - Production agent implementations")
    print("   • src/states/ - Real PHMState management")
    print("   • src/utils/ - Production utility functions")
    
    print("\n🔬 Educational Features:")
    educational_features = [
        "Step-by-step workflow breakdown with timing",
        "Progress monitoring and DAG evolution tracking",
        "Intermediate result visualization and explanation",
        "Error handling with educational guidance",
        "Conceptual explanation of disabled components",
        "Results persistence and tutorial completion tracking"
    ]
    
    for feature in educational_features:
        print(f"   • {feature}")
    
    print("\n🚀 Ready to Run:")
    print("   Use run_case1_tutorial() to execute the educational case!")
    print("   This will demonstrate the complete PHMGA workflow")
    print("   using the same logic as the production system.")


if __name__ == "__main__":
    # Demonstrate the tutorial concepts
    demonstrate_case1_concepts()
    
    print("\n" + "=" * 60)
    print("To run the tutorial, execute:")
    print("   python case1_tutorial.py --run")
    print("\nOr in Python:")
    print("   from case1_tutorial import run_case1_tutorial")
    print("   result = run_case1_tutorial()")
    
    # Optionally run the tutorial if --run flag is provided
    import sys
    if "--run" in sys.argv:
        print("\n🏃 Running Case 1 Tutorial...")
        try:
            result = run_case1_tutorial()
            print(f"\n✅ Tutorial completed successfully!")
            print(f"   Final state available with {len(result.dag_state.nodes)} DAG nodes")
        except Exception as e:
            print(f"\n⚠️ Tutorial failed: {e}")
            print("   This is expected without real signal data files.")
            print("   The tutorial demonstrates the workflow structure.")