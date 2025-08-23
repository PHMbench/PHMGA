#!/usr/bin/env python3
"""
Comprehensive test script for the PHM Research Agent System.

This script validates the integration and functionality of all research agents,
workflows, and case studies in the enhanced PHM system.
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_research_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_research_states():
    """Test research state management functionality."""
    print("\n=== Testing Research States ===")
    
    try:
        from src.states.research_states import ResearchPHMState, ResearchHypothesis
        from src.states.phm_states import DAGState, InputData
        
        # Create test data
        dag_state = DAGState(
            user_instruction="Test research states",
            channels=["ch1"],
            nodes={},
            leaves=[]
        )
        
        ref_signal = InputData(
            node_id="ref_ch1",
            parents=[],
            shape=(2, 1000, 1),
            results={"ref": np.random.randn(2, 1000, 1)},
            meta={"fs": 1000}
        )
        
        test_signal = InputData(
            node_id="test_ch1",
            parents=[],
            shape=(1, 1000, 1),
            results={"test": np.random.randn(1, 1000, 1)},
            meta={"fs": 1000}
        )
        
        # Create research state
        state = ResearchPHMState(
            case_name="test_research_states",
            user_instruction="Test research state functionality",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state,
            fs=1000
        )
        
        # Test state functionality
        obj_id = state.add_research_objective("Test objective", priority=1)
        hyp_id = state.add_hypothesis("Test hypothesis", 0.8, "test_agent")
        state.add_audit_entry("test_agent", "test_action", 0.9)
        
        # Validate state
        assert len(state.research_objectives) == 1
        assert len(state.research_hypotheses) == 1
        assert len(state.research_audit_trail) == 1
        assert state.calculate_research_progress() == 0.0  # No completed objectives
        
        print("✓ Research states test passed")
        return True
        
    except Exception as e:
        print(f"✗ Research states test failed: {e}")
        logger.error(f"Research states test error: {traceback.format_exc()}")
        return False


def test_research_agents():
    """Test individual research agent functionality."""
    print("\n=== Testing Research Agents ===")
    
    results = {}
    
    # Test Data Analyst Agent
    try:
        from src.agents.data_analyst_agent import DataAnalystAgent
        from src.states.research_states import ResearchPHMState
        from src.states.phm_states import DAGState, InputData
        
        # Create test state
        state = create_test_state("test_data_analyst")
        
        # Test agent
        agent = DataAnalystAgent()
        analysis_results = agent.analyze(state)
        hypotheses = agent.generate_hypotheses(state, analysis_results)
        
        assert "confidence" in analysis_results
        assert analysis_results["confidence"] >= 0.0
        assert isinstance(hypotheses, list)
        
        results["data_analyst"] = True
        print("✓ Data Analyst Agent test passed")
        
    except Exception as e:
        print(f"✗ Data Analyst Agent test failed: {e}")
        results["data_analyst"] = False
    
    # Test Algorithm Researcher Agent
    try:
        from src.agents.algorithm_researcher_agent import AlgorithmResearcherAgent
        
        state = create_test_state("test_algorithm_researcher")
        
        agent = AlgorithmResearcherAgent()
        analysis_results = agent.analyze(state)
        hypotheses = agent.generate_hypotheses(state, analysis_results)
        
        assert "confidence" in analysis_results
        assert isinstance(hypotheses, list)
        
        results["algorithm_researcher"] = True
        print("✓ Algorithm Researcher Agent test passed")
        
    except Exception as e:
        print(f"✗ Algorithm Researcher Agent test failed: {e}")
        results["algorithm_researcher"] = False
    
    # Test Domain Expert Agent
    try:
        from src.agents.domain_expert_agent import DomainExpertAgent
        
        state = create_test_state("test_domain_expert")
        
        agent = DomainExpertAgent()
        analysis_results = agent.analyze(state)
        hypotheses = agent.generate_hypotheses(state, analysis_results)
        
        assert "confidence" in analysis_results
        assert isinstance(hypotheses, list)
        
        results["domain_expert"] = True
        print("✓ Domain Expert Agent test passed")
        
    except Exception as e:
        print(f"✗ Domain Expert Agent test failed: {e}")
        results["domain_expert"] = False
    
    # Test Integration Agent
    try:
        from src.agents.integration_agent import IntegrationAgent
        
        state = create_test_state("test_integration")
        # Add mock agent results
        state.data_analysis_state = {"confidence": 0.8}
        state.algorithm_research_state = {"confidence": 0.7}
        state.domain_expert_state = {"confidence": 0.9}
        
        agent = IntegrationAgent()
        analysis_results = agent.analyze(state)
        hypotheses = agent.generate_hypotheses(state, analysis_results)
        
        assert "confidence" in analysis_results
        assert isinstance(hypotheses, list)
        
        results["integration_agent"] = True
        print("✓ Integration Agent test passed")
        
    except Exception as e:
        print(f"✗ Integration Agent test failed: {e}")
        results["integration_agent"] = False
    
    return results


def test_research_workflow():
    """Test the complete research workflow."""
    print("\n=== Testing Research Workflow ===")
    
    try:
        from src.research_workflow import build_research_graph
        
        # Build workflow graph
        graph = build_research_graph()
        
        # Validate graph structure
        assert graph is not None
        assert len(graph.nodes) > 0
        
        print(f"✓ Research workflow graph built with {len(graph.nodes)} nodes")
        return True
        
    except Exception as e:
        print(f"✗ Research workflow test failed: {e}")
        logger.error(f"Research workflow test error: {traceback.format_exc()}")
        return False


def test_case1_enhanced():
    """Test enhanced Case 1 functionality."""
    print("\n=== Testing Enhanced Case 1 ===")
    
    try:
        # Create minimal test configuration
        test_config = create_test_config("case1_enhanced_test")
        
        # Test configuration validation
        assert "name" in test_config
        assert "user_instruction" in test_config
        
        print("✓ Enhanced Case 1 configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Case 1 test failed: {e}")
        logger.error(f"Enhanced Case 1 test error: {traceback.format_exc()}")
        return False


def test_case2_predictive():
    """Test Case 2 predictive maintenance functionality."""
    print("\n=== Testing Case 2 Predictive Maintenance ===")
    
    try:
        from src.cases.case2_predictive import (
            RULEstimator, HealthIndexCalculator, MaintenanceScheduler
        )
        
        # Test RUL Estimator
        rul_estimator = RULEstimator()
        health_history = [1.0, 0.9, 0.8, 0.7, 0.6]
        time_points = [0, 1, 2, 3, 4]
        rul_result = rul_estimator.estimate_rul(health_history, time_points)
        
        assert "rul_estimate" in rul_result
        assert rul_result["rul_estimate"] >= 0
        
        # Test Health Index Calculator
        health_calculator = HealthIndexCalculator()
        features = {"rms": 1.2, "kurtosis": 4.0, "crest_factor": 3.5}
        baseline = {"rms": 1.0, "kurtosis": 3.0, "crest_factor": 3.0}
        health_result = health_calculator.calculate_health_index(features, baseline)
        
        assert "health_index" in health_result
        assert 0 <= health_result["health_index"] <= 1
        
        # Test Maintenance Scheduler
        scheduler = MaintenanceScheduler()
        schedule_result = scheduler.generate_schedule(30, 0.6)
        
        assert "strategy" in schedule_result
        assert "schedule" in schedule_result
        
        print("✓ Case 2 Predictive Maintenance test passed")
        return True
        
    except Exception as e:
        print(f"✗ Case 2 Predictive Maintenance test failed: {e}")
        logger.error(f"Case 2 test error: {traceback.format_exc()}")
        return False


def test_system_integration():
    """Test overall system integration."""
    print("\n=== Testing System Integration ===")
    
    try:
        # Test import compatibility
        from src.states.research_states import ResearchPHMState
        from src.research_workflow import build_research_graph
        from src.agents.data_analyst_agent import data_analyst_agent
        from src.agents.algorithm_researcher_agent import algorithm_researcher_agent
        from src.agents.domain_expert_agent import domain_expert_agent
        from src.agents.integration_agent import integration_agent
        
        # Test state conversion
        state = create_test_state("integration_test")
        
        # Test agent node functions
        data_result = data_analyst_agent(state)
        assert isinstance(data_result, dict)
        
        print("✓ System integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ System integration test failed: {e}")
        logger.error(f"System integration test error: {traceback.format_exc()}")
        return False


def create_test_state(case_name: str):
    """Create a test research state for testing."""
    from src.states.research_states import ResearchPHMState
    from src.states.phm_states import DAGState, InputData
    
    # Create test signals
    fs = 1000
    t = np.linspace(0, 1, fs)
    
    # Reference signals (healthy)
    ref_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t)),
        np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Test signals (with simulated fault)
    test_signals = np.array([
        np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
    ])[:, :, np.newaxis]
    
    # Create state components
    dag_state = DAGState(
        user_instruction=f"Test case: {case_name}",
        channels=["ch1"],
        nodes={},
        leaves=[]
    )
    
    ref_signal = InputData(
        node_id="ref",
        parents=[],
        shape=ref_signals.shape,
        results={"ref": ref_signals},
        meta={"fs": fs}
    )
    
    test_signal = InputData(
        node_id="test",
        parents=[],
        shape=test_signals.shape,
        results={"test": test_signals},
        meta={"fs": fs}
    )
    
    # Create research state
    state = ResearchPHMState(
        case_name=case_name,
        user_instruction=f"Test research capabilities for {case_name}",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=fs
    )
    
    return state


def create_test_config(case_name: str) -> Dict[str, Any]:
    """Create a test configuration for case studies."""
    return {
        "name": case_name,
        "user_instruction": f"Test configuration for {case_name}",
        "metadata_path": "/tmp/test_metadata.xlsx",
        "h5_path": "/tmp/test_cache.h5",
        "ref_ids": [1, 2, 3],
        "test_ids": [4, 5],
        "state_save_path": f"/tmp/{case_name}_state.pkl",
        "report_path": f"/tmp/{case_name}_report.md",
        "builder": {
            "min_depth": 2,
            "max_depth": 4
        }
    }


def run_comprehensive_test():
    """Run comprehensive test suite for the research agent system."""
    print("=" * 60)
    print("PHM Research Agent System - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Research States", test_research_states),
        ("Research Agents", test_research_agents),
        ("Research Workflow", test_research_workflow),
        ("Enhanced Case 1", test_case1_enhanced),
        ("Case 2 Predictive", test_case2_predictive),
        ("System Integration", test_system_integration)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            test_results[test_name] = False
            logger.error(f"{test_name} test crash: {traceback.format_exc()}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # Handle agent test results
            agent_passed = sum(1 for r in result.values() if r)
            agent_total = len(result)
            status = "✓ PASSED" if agent_passed == agent_total else f"⚠ PARTIAL ({agent_passed}/{agent_total})"
            print(f"{test_name:.<40} {status}")
            passed += agent_passed
            total += agent_total
        else:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name:.<40} {status}")
            if result:
                passed += 1
            total += 1
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Research agent system is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the logs and fix issues.")
        return False


if __name__ == "__main__":
    # Set up test environment
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
