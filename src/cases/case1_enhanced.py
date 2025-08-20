"""
Enhanced Case 1: Research-Augmented Bearing Fault Diagnosis

This enhanced version of Case 1 integrates the research agent system with the traditional
bearing fault diagnosis workflow, providing research-driven insights and interpretability.
"""

from __future__ import annotations
import uuid
import yaml
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

# Load environment variables
load_dotenv()

# Disable LangSmith for cleaner logs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

from ..phm_outer_graph import build_builder_graph, build_executor_graph
from ..research_workflow import build_research_graph
from ..utils import initialize_state, save_state, load_state, generate_final_report
from ..states.research_states import ResearchPHMState
from ..agents.reflect_agent import get_dag_depth

logger = logging.getLogger(__name__)


def convert_to_research_state(phm_state) -> ResearchPHMState:
    """
    Convert traditional PHMState to ResearchPHMState for research workflow.
    
    Args:
        phm_state: Traditional PHMState object
        
    Returns:
        ResearchPHMState with research capabilities
    """
    # Extract core fields from traditional state
    research_state = ResearchPHMState(
        case_name=phm_state.case_name,
        user_instruction=phm_state.user_instruction,
        reference_signal=phm_state.reference_signal,
        test_signal=phm_state.test_signal,
        dag_state=phm_state.dag_state,
        
        # Copy processing configuration
        min_depth=phm_state.min_depth,
        max_depth=phm_state.max_depth,
        min_width=phm_state.min_width,
        max_steps=phm_state.max_steps,
        fs=phm_state.fs,
        
        # Copy paths configuration
        data_dir=phm_state.data_dir,
        save_dir=phm_state.save_dir,
        cache_dir=phm_state.cache_dir,
        
        # Copy processing state
        detailed_plan=phm_state.detailed_plan,
        error_logs=phm_state.error_logs,
        needs_revision=phm_state.needs_revision,
        
        # Copy reflection and iteration
        reflection_history=phm_state.reflection_history,
        is_sufficient=phm_state.is_sufficient,
        iteration_count=phm_state.iteration_count,
        
        # Initialize research-specific fields
        research_phase="initialization",
        research_objectives=[],
        research_hypotheses=[],
        research_confidence=0.0,
        max_research_iterations=3,
        research_quality_threshold=0.8
    )
    
    return research_state


def run_enhanced_case(config_path: str, enable_research: bool = True, 
                     research_only: bool = False) -> Optional[ResearchPHMState]:
    """
    Run enhanced Case 1 with optional research agent integration.
    
    Args:
        config_path: Path to configuration file
        enable_research: Whether to run research agents
        research_only: Whether to skip traditional workflow and run research only
        
    Returns:
        Final research state or None if failed
    """
    logger.info(f"Starting enhanced Case 1 with config: {config_path}")
    logger.info(f"Research enabled: {enable_research}, Research only: {research_only}")
    
    # Load configuration
    print(f"--- Loading configuration from {config_path} ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    state_save_path = config['state_save_path']
    research_save_path = state_save_path.replace('.pkl', '_research.pkl')
    builder_cfg = config.get('builder', {})
    min_depth = builder_cfg.get('min_depth', 0)
    max_depth = builder_cfg.get('max_depth', float('inf'))
    
    built_state = None
    
    if not research_only:
        # --- Traditional Workflow (Plan-Execute-Reflect) ---
        if os.path.exists(state_save_path):
            print(f"\n--- Found existing state file at {state_save_path}. Loading. ---")
            built_state = load_state(state_save_path)
            if built_state is None:
                print("Failed to load state. Exiting.")
                return None
        else:
            print(f"\n--- No existing state file found. Starting builder workflow. ---")
            
            # Initialize state
            print("\n--- [Part 0] Initializing State ---")
            initial_phm_state = initialize_state(
                user_instruction=config['user_instruction'],
                metadata_path=config['metadata_path'],
                h5_path=config['h5_path'],
                ref_ids=config['ref_ids'],
                test_ids=config['test_ids'],
                case_name=config['name']
            )
            
            # Build DAG using traditional workflow
            print("\n--- [Part 1] Starting DAG Builder Workflow ---")
            builder_app = build_builder_graph()
            thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            built_state = initial_phm_state.model_copy(deep=True)
            
            for event in builder_app.stream(initial_phm_state, config=thread_config):
                for node_name, state_update in event.items():
                    print(f"--- Builder Node Executed: {node_name} ---")
                    if state_update is not None:
                        for key, value in state_update.items():
                            setattr(built_state, key, value)
                    
                    # Check depth constraint
                    current_depth = get_dag_depth(built_state.dag_state)
                    if current_depth >= max_depth:
                        print(f"--- Maximum depth {max_depth} reached. Stopping builder. ---")
                        break
                
                # Break outer loop if max depth reached
                current_depth = get_dag_depth(built_state.dag_state)
                if current_depth >= max_depth:
                    break
            
            # Save traditional state
            save_state(built_state, state_save_path)
            print(f"--- Traditional workflow state saved to {state_save_path} ---")
    
    if enable_research:
        # --- Research Workflow ---
        print("\n--- [Part 2] Starting Research Agent Workflow ---")
        
        # Check for existing research state
        if os.path.exists(research_save_path):
            print(f"--- Found existing research state at {research_save_path}. Loading. ---")
            research_state = load_state(research_save_path)
            if research_state is None:
                print("Failed to load research state. Creating new one.")
                if built_state:
                    research_state = convert_to_research_state(built_state)
                else:
                    # Initialize minimal state for research-only mode
                    initial_phm_state = initialize_state(
                        user_instruction=config['user_instruction'],
                        metadata_path=config['metadata_path'],
                        h5_path=config['h5_path'],
                        ref_ids=config['ref_ids'],
                        test_ids=config['test_ids'],
                        case_name=config['name']
                    )
                    research_state = convert_to_research_state(initial_phm_state)
        else:
            print("--- No existing research state found. Creating new research state. ---")
            if built_state:
                research_state = convert_to_research_state(built_state)
            else:
                # Initialize minimal state for research-only mode
                initial_phm_state = initialize_state(
                    user_instruction=config['user_instruction'],
                    metadata_path=config['metadata_path'],
                    h5_path=config['h5_path'],
                    ref_ids=config['ref_ids'],
                    test_ids=config['test_ids'],
                    case_name=config['name']
                )
                research_state = convert_to_research_state(initial_phm_state)
        
        # Run research workflow
        research_app = build_research_graph()
        research_thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        final_research_state = research_state.model_copy(deep=True)
        
        try:
            for event in research_app.stream(research_state, config=research_thread_config):
                for node_name, state_update in event.items():
                    print(f"--- Research Node Executed: {node_name} ---")
                    if state_update is not None:
                        for key, value in state_update.items():
                            if hasattr(final_research_state, key):
                                setattr(final_research_state, key, value)
                            else:
                                logger.warning(f"Unknown state key: {key}")
                    
                    # Log research progress
                    if hasattr(final_research_state, 'research_confidence'):
                        print(f"    Research confidence: {final_research_state.research_confidence:.3f}")
                    if hasattr(final_research_state, 'research_phase'):
                        print(f"    Research phase: {final_research_state.research_phase}")
        
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            print(f"--- Research workflow error: {e} ---")
            return final_research_state
        
        # Save research state
        save_state(final_research_state, research_save_path)
        print(f"--- Research state saved to {research_save_path} ---")
        
        # Generate enhanced report
        if hasattr(final_research_state, 'research_report'):
            research_report_path = config.get('report_path', '').replace('.md', '_research.md')
            if research_report_path:
                try:
                    with open(research_report_path, 'w') as f:
                        f.write(final_research_state.research_report)
                    print(f"--- Research report saved to {research_report_path} ---")
                except Exception as e:
                    logger.error(f"Failed to save research report: {e}")
        
        return final_research_state
    
    else:
        # Return traditional state converted to research state for consistency
        if built_state:
            return convert_to_research_state(built_state)
        else:
            return None


def run_traditional_case(config_path: str):
    """
    Run traditional Case 1 workflow without research agents.
    
    Args:
        config_path: Path to configuration file
    """
    return run_enhanced_case(config_path, enable_research=False, research_only=False)


def run_research_only_case(config_path: str):
    """
    Run research agents only without traditional DAG building.
    
    Args:
        config_path: Path to configuration file
    """
    return run_enhanced_case(config_path, enable_research=True, research_only=True)


def compare_traditional_vs_research(config_path: str) -> Dict[str, Any]:
    """
    Compare traditional and research-enhanced approaches.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Comparison results
    """
    print("\n=== COMPARISON: Traditional vs Research-Enhanced ===")
    
    # Run traditional approach
    print("\n--- Running Traditional Approach ---")
    traditional_result = run_traditional_case(config_path)
    
    # Run research-enhanced approach
    print("\n--- Running Research-Enhanced Approach ---")
    research_result = run_enhanced_case(config_path, enable_research=True, research_only=False)
    
    # Compare results
    comparison = {
        "traditional": {
            "completed": traditional_result is not None,
            "confidence": getattr(traditional_result, 'research_confidence', 0.0),
            "iterations": getattr(traditional_result, 'iteration_count', 0),
            "dag_depth": get_dag_depth(traditional_result.dag_state) if traditional_result else 0
        },
        "research_enhanced": {
            "completed": research_result is not None,
            "confidence": getattr(research_result, 'research_confidence', 0.0),
            "iterations": getattr(research_result, 'iteration_count', 0),
            "hypotheses": len(getattr(research_result, 'research_hypotheses', [])),
            "research_quality": getattr(research_result, 'integration_state', {}).get('research_quality_score', 0.0)
        }
    }
    
    # Print comparison
    print("\n=== COMPARISON RESULTS ===")
    print(f"Traditional - Completed: {comparison['traditional']['completed']}, "
          f"Confidence: {comparison['traditional']['confidence']:.3f}")
    print(f"Research Enhanced - Completed: {comparison['research_enhanced']['completed']}, "
          f"Confidence: {comparison['research_enhanced']['confidence']:.3f}, "
          f"Hypotheses: {comparison['research_enhanced']['hypotheses']}")
    
    return comparison


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced Case 1 with research agents")
    parser.add_argument("--config", type=str, default="config/case1.yaml", 
                       help="Configuration file path")
    parser.add_argument("--mode", type=str, choices=["traditional", "research", "enhanced", "compare"],
                       default="enhanced", help="Execution mode")
    
    args = parser.parse_args()
    
    if args.mode == "traditional":
        result = run_traditional_case(args.config)
        print(f"Traditional case completed: {result is not None}")
    
    elif args.mode == "research":
        result = run_research_only_case(args.config)
        print(f"Research-only case completed: {result is not None}")
        if result:
            print(f"Research confidence: {result.research_confidence:.3f}")
            print(f"Generated hypotheses: {len(result.research_hypotheses)}")
    
    elif args.mode == "enhanced":
        result = run_enhanced_case(args.config)
        print(f"Enhanced case completed: {result is not None}")
        if result:
            print(f"Research confidence: {result.research_confidence:.3f}")
            print(f"Generated hypotheses: {len(result.research_hypotheses)}")
            print(f"Research phase: {result.research_phase}")
    
    elif args.mode == "compare":
        comparison = compare_traditional_vs_research(args.config)
        print("Comparison completed successfully!")
    
    else:
        print(f"Unknown mode: {args.mode}")
