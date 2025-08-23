#!/usr/bin/env python3
"""
Test suite for the enhanced PHM Research Agent System.

This script validates the refactored agent architecture, performance improvements,
and new features including service layer, agent factory, and monitoring capabilities.
"""

import os
import sys
import numpy as np
import logging
import time
from typing import Dict, Any, List
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up the environment to avoid import issues
os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_enhanced_agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_agent_factory():
    """Test the agent factory and configuration system."""
    print("\n=== Testing Agent Factory ===")
    
    try:
        from src.agents.agent_factory import AgentFactory, AgentType, ConfigurationManager
        
        # Test configuration manager
        config_manager = ConfigurationManager()
        presets = config_manager.list_presets()
        print(f"Available presets: {presets}")
        assert len(presets) >= 3, "Should have at least 3 default presets"
        
        # Test agent factory
        factory = AgentFactory()
        
        # Test builder pattern
        agent = (factory.create_agent_builder(AgentType.DATA_ANALYST)
                .with_name("test_data_analyst")
                .with_config(quick_mode=True, enable_advanced_features=False)
                .with_validator("state", "state")
                .build())
        
        assert agent.agent_name == "test_data_analyst"
        assert agent.quick_mode == True
        print("✓ Agent factory and builder pattern working")
        
        # Test preset creation
        preset_agents = factory.create_agent_from_preset("QuickDiagnosis")
        assert len(preset_agents) >= 1, "QuickDiagnosis preset should create at least 1 agent"
        print(f"✓ Created {len(preset_agents)} agents from QuickDiagnosis preset")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent factory test failed: {e}")
        traceback.print_exc()
        return False


def test_service_layer():
    """Test the service layer functionality."""
    print("\n=== Testing Service Layer ===")
    
    try:
        from src.agents.services import service_registry
        
        # Test service registry
        services = service_registry.list_services()
        print(f"Available services: {services}")
        assert "feature_extraction" in services
        assert "statistical_analysis" in services
        assert "ml_model" in services
        
        # Test feature extraction service
        feature_service = service_registry.get("feature_extraction")
        test_signals = np.random.randn(2, 1000, 1)
        
        features = feature_service.extract_features(
            test_signals, 
            feature_names=["mean", "std", "rms"],
            use_cache=True,
            use_parallel=True
        )
        
        assert len(features) >= 2, "Should extract at least 2 features"
        print(f"✓ Extracted features: {list(features.keys())}")
        
        # Test statistical analysis service
        stats_service = service_registry.get("statistical_analysis")
        stats = stats_service.analyze(test_signals, methods=["basic_stats", "outlier_detection"])
        
        assert "basic_stats" in stats
        print("✓ Statistical analysis service working")
        
        # Test caching
        start_time = time.time()
        features_cached = feature_service.extract_features(
            test_signals, 
            feature_names=["mean", "std", "rms"],
            use_cache=True
        )
        cached_time = time.time() - start_time
        
        print(f"✓ Cached extraction time: {cached_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Service layer test failed: {e}")
        traceback.print_exc()
        return False


def test_enhanced_data_analyst():
    """Test the enhanced Data Analyst Agent."""
    print("\n=== Testing Enhanced Data Analyst Agent ===")
    
    try:
        from src.agents.data_analyst_agent import DataAnalystAgent
        from src.states.research_states import ResearchPHMState
        from src.states.phm_states import DAGState, InputData
        
        # Create test data
        fs = 1000
        t = np.linspace(0, 1, fs)
        
        # Reference signals (healthy)
        ref_signals = np.array([
            np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t)),
            np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))
        ])[:, :, np.newaxis]
        
        # Test signals (with fault)
        test_signals = np.array([
            np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
        ])[:, :, np.newaxis]
        
        # Create research state
        dag_state = DAGState(user_instruction="Test enhanced analysis", channels=["ch1"], nodes={}, leaves=[])
        ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                              results={"ref": ref_signals}, meta={"fs": fs})
        test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                               results={"test": test_signals}, meta={"fs": fs})
        
        state = ResearchPHMState(
            case_name="test_enhanced_analysis",
            user_instruction="Test enhanced data analyst agent",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state,
            fs=fs
        )
        
        # Test different configurations
        configs = [
            {"quick_mode": True, "enable_advanced_features": False},
            {"quick_mode": False, "enable_advanced_features": True, "enable_pca_analysis": True}
        ]
        
        for i, config in enumerate(configs):
            print(f"\nTesting configuration {i+1}: {config}")
            
            agent = DataAnalystAgent(config=config)
            
            # Perform analysis
            analysis_result = agent.analyze(state)
            
            assert analysis_result.is_successful(), "Analysis should be successful"
            assert analysis_result.confidence > 0, "Should have positive confidence"
            
            print(f"  ✓ Analysis confidence: {analysis_result.confidence:.3f}")
            print(f"  ✓ Execution time: {analysis_result.execution_time:.2f}s")
            print(f"  ✓ Memory usage: {analysis_result.memory_usage:.1f}MB")
            
            # Generate hypotheses
            hypotheses = agent.generate_hypotheses(state, analysis_result)
            assert len(hypotheses) > 0, "Should generate at least one hypothesis"
            print(f"  ✓ Generated {len(hypotheses)} hypotheses")
            
            # Check performance metrics
            perf_metrics = agent.get_performance_metrics()
            if not perf_metrics.get("no_executions"):
                print(f"  ✓ Success rate: {perf_metrics['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Data Analyst test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring and circuit breaker functionality."""
    print("\n=== Testing Performance Monitoring ===")
    
    try:
        from src.agents.research_base import monitor_performance, circuit_breaker
        
        # Test performance monitoring decorator
        @monitor_performance
        def test_function(duration=0.1):
            time.sleep(duration)
            return "success"
        
        result = test_function(0.05)
        assert result == "success"
        print("✓ Performance monitoring decorator working")
        
        # Test circuit breaker
        @circuit_breaker(max_failures=2)
        def failing_function(should_fail=True):
            if should_fail:
                raise ValueError("Test failure")
            return "success"
        
        # Test failures
        failure_count = 0
        for i in range(3):
            try:
                failing_function(True)
            except ValueError:
                failure_count += 1
            except Exception as e:
                if "Circuit breaker open" in str(e):
                    print("✓ Circuit breaker activated after failures")
                    break
        
        assert failure_count >= 2, "Should have at least 2 failures before circuit breaker"
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False


def test_input_validation():
    """Test input validation functionality."""
    print("\n=== Testing Input Validation ===")
    
    try:
        from src.agents.research_base import SignalValidator, StateValidator
        from src.states.research_states import ResearchPHMState
        from src.states.phm_states import DAGState, InputData
        
        # Test signal validator
        signal_validator = SignalValidator()
        
        # Valid signal
        valid_signal = np.random.randn(2, 1000, 1)
        is_valid, errors = signal_validator.validate(valid_signal)
        assert is_valid, f"Valid signal should pass validation: {errors}"
        print("✓ Valid signal passed validation")
        
        # Invalid signals
        invalid_signals = [
            None,  # None signal
            np.array([]),  # Empty signal
            np.random.randn(5),  # 1D signal
            np.full((2, 1000, 1), np.nan),  # NaN signal
        ]
        
        for i, invalid_signal in enumerate(invalid_signals):
            is_valid, errors = signal_validator.validate(invalid_signal)
            assert not is_valid, f"Invalid signal {i} should fail validation"
            print(f"✓ Invalid signal {i} correctly rejected: {errors[0] if errors else 'No error message'}")
        
        # Test state validator
        state_validator = StateValidator()
        
        # Create valid state
        dag_state = DAGState(user_instruction="Test validation", channels=["ch1"], nodes={}, leaves=[])
        ref_signal = InputData(node_id="ref", parents=[], shape=(2, 1000, 1), 
                              results={"ref": np.random.randn(2, 1000, 1)}, meta={"fs": 1000})
        test_signal = InputData(node_id="test", parents=[], shape=(1, 1000, 1),
                               results={"test": np.random.randn(1, 1000, 1)}, meta={"fs": 1000})
        
        valid_state = ResearchPHMState(
            case_name="test_validation",
            user_instruction="Test state validation",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state,
            fs=1000
        )
        
        is_valid, errors = state_validator.validate(valid_state)
        assert is_valid, f"Valid state should pass validation: {errors}"
        print("✓ Valid state passed validation")
        
        return True
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_and_performance():
    """Test memory usage and performance improvements."""
    print("\n=== Testing Memory and Performance ===")
    
    try:
        from src.agents.data_analyst_agent import DataAnalystAgent
        from src.states.research_states import ResearchPHMState
        from src.states.phm_states import DAGState, InputData
        import psutil
        
        # Create larger test data to test memory handling
        fs = 1000
        signal_length = 10000  # Larger signal
        n_signals = 5
        
        ref_signals = np.random.randn(n_signals, signal_length, 1)
        test_signals = np.random.randn(n_signals//2, signal_length, 1)
        
        # Create research state
        dag_state = DAGState(user_instruction="Test performance", channels=["ch1"], nodes={}, leaves=[])
        ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                              results={"ref": ref_signals}, meta={"fs": fs})
        test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                               results={"test": test_signals}, meta={"fs": fs})
        
        state = ResearchPHMState(
            case_name="test_performance",
            user_instruction="Test memory and performance",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state,
            fs=fs
        )
        
        # Test with different configurations
        configs = [
            {"quick_mode": True, "use_parallel": False, "use_cache": False},
            {"quick_mode": False, "use_parallel": True, "use_cache": True}
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nTesting performance configuration {i+1}: {config}")
            
            # Monitor memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            agent = DataAnalystAgent(config=config)
            
            start_time = time.time()
            analysis_result = agent.analyze(state)
            end_time = time.time()
            
            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            execution_time = end_time - start_time
            
            results.append({
                "config": config,
                "execution_time": execution_time,
                "memory_used": memory_used,
                "confidence": analysis_result.confidence,
                "success": analysis_result.is_successful()
            })
            
            print(f"  ✓ Execution time: {execution_time:.2f}s")
            print(f"  ✓ Memory used: {memory_used:.1f}MB")
            print(f"  ✓ Confidence: {analysis_result.confidence:.3f}")
        
        # Compare performance
        if len(results) >= 2:
            quick_result = results[0]
            full_result = results[1]
            
            print(f"\nPerformance Comparison:")
            print(f"  Quick mode: {quick_result['execution_time']:.2f}s, {quick_result['memory_used']:.1f}MB")
            print(f"  Full mode: {full_result['execution_time']:.2f}s, {full_result['memory_used']:.1f}MB")
            
            # Quick mode should be faster (though not always due to overhead)
            if quick_result['execution_time'] < full_result['execution_time']:
                print("  ✓ Quick mode is faster as expected")
            else:
                print("  ⚠ Quick mode not faster (may be due to small dataset)")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory and performance test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all enhanced agent tests."""
    print("Starting Enhanced PHM Research Agent System Tests")
    print("=" * 60)
    
    tests = [
        ("Agent Factory", test_agent_factory),
        ("Service Layer", test_service_layer),
        ("Enhanced Data Analyst", test_enhanced_data_analyst),
        ("Performance Monitoring", test_performance_monitoring),
        ("Input Validation", test_input_validation),
        ("Memory and Performance", test_memory_and_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            
            results[test_name] = {
                "success": success,
                "duration": end_time - start_time
            }
            
            if success:
                print(f"✓ {test_name} completed successfully in {end_time - start_time:.2f}s")
            else:
                print(f"✗ {test_name} failed")
                
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = {"success": False, "duration": 0, "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(tests)
    passed_tests = sum(1 for r in results.values() if r["success"])
    total_time = sum(r["duration"] for r in results.values())
    
    for test_name, result in results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        duration = result["duration"]
        print(f"{test_name:<25} {status:<8} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed in {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Enhanced agent system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
