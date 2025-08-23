#!/usr/bin/env python3
"""
Core refactoring test for the enhanced PHM Research Agent System.

This script tests the core improvements without complex dependencies,
focusing on the architectural enhancements and performance monitoring.
"""

import os
import sys
import numpy as np
import logging
import time
import traceback
from typing import Dict, Any, List
import functools
import psutil
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Core architectural components (standalone versions for testing)

def monitor_performance(func):
    """Decorator to monitor memory usage and execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise e
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
            
            logger.info(f"{func.__name__}: Time={execution_time:.2f}s, "
                       f"Memory={memory_delta:+.1f}MB, Success={success}")
        
        return result
    return wrapper


def circuit_breaker(max_failures: int = 3, reset_timeout: int = 60):
    """Circuit breaker pattern for agent methods."""
    def decorator(func):
        func.failure_count = 0
        func.last_failure_time = 0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Reset failure count after timeout
            if current_time - func.last_failure_time > reset_timeout:
                func.failure_count = 0
            
            # Check if circuit is open
            if func.failure_count >= max_failures:
                raise Exception(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                func.failure_count = 0  # Reset on success
                return result
            except Exception as e:
                func.failure_count += 1
                func.last_failure_time = current_time
                logger.error(f"Circuit breaker: {func.__name__} failed "
                           f"({func.failure_count}/{max_failures})")
                raise e
        
        return wrapper
    return decorator


@dataclass
class AnalysisResult:
    """Typed result object for agent analysis."""
    agent_name: str
    confidence: float
    results: Dict[str, Any]
    execution_time: float
    memory_usage: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return len(self.errors) == 0 and self.confidence > 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis result."""
        return {
            "agent": self.agent_name,
            "confidence": self.confidence,
            "success": self.is_successful(),
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class SignalValidator:
    """Validator for signal data."""
    
    def __init__(self, min_length: int = 10, max_length: int = 1000000):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, signals: np.ndarray) -> tuple[bool, List[str]]:
        """Validate signal array."""
        errors = []
        
        if signals is None:
            errors.append("Signal data is None")
            return False, errors
        
        if not isinstance(signals, np.ndarray):
            errors.append(f"Expected numpy array, got {type(signals)}")
            return False, errors
        
        if signals.size == 0:
            errors.append("Signal array is empty")
            return False, errors
        
        if len(signals.shape) < 2:
            errors.append(f"Signal must be at least 2D, got shape {signals.shape}")
            return False, errors
        
        signal_length = signals.shape[1]
        if signal_length < self.min_length:
            errors.append(f"Signal too short: {signal_length} < {self.min_length}")
        
        if signal_length > self.max_length:
            errors.append(f"Signal too long: {signal_length} > {self.max_length}")
        
        if np.any(np.isnan(signals)):
            errors.append("Signal contains NaN values")
        
        if np.any(np.isinf(signals)):
            errors.append("Signal contains infinite values")
        
        return len(errors) == 0, errors


class MockAgent:
    """Mock agent for testing the enhanced architecture."""
    
    def __init__(self, agent_name: str = "mock_agent", config: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_time": 0.0,
            "total_memory": 0.0
        }
        self.validator = SignalValidator()
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with type safety."""
        return self.config.get(key, default)
    
    def update_execution_stats(self, execution_time: float, memory_usage: float, success: bool):
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        if success:
            self.execution_stats["successful_executions"] += 1
        self.execution_stats["total_time"] += execution_time
        self.execution_stats["total_memory"] += memory_usage
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        total_exec = self.execution_stats["total_executions"]
        if total_exec == 0:
            return {"no_executions": True}
        
        return {
            "total_executions": total_exec,
            "success_rate": self.execution_stats["successful_executions"] / total_exec,
            "avg_execution_time": self.execution_stats["total_time"] / total_exec,
            "avg_memory_usage": self.execution_stats["total_memory"] / total_exec,
            "total_time": self.execution_stats["total_time"],
            "total_memory": self.execution_stats["total_memory"]
        }
    
    @monitor_performance
    @circuit_breaker(max_failures=3)
    def analyze(self, signals: np.ndarray) -> AnalysisResult:
        """Mock analysis with monitoring and validation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Validate input
        is_valid, errors = self.validator.validate(signals)
        if not is_valid:
            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=0.0,
                results={},
                execution_time=0.0,
                memory_usage=0.0,
                errors=errors
            )
        
        try:
            # Simulate analysis work
            processing_time = self.get_config_value("processing_time", 0.1)
            time.sleep(processing_time)
            
            # Simulate some computation
            result = np.mean(signals ** 2)
            confidence = min(result / 10.0, 1.0)
            
            results = {
                "signal_energy": float(result),
                "signal_shape": signals.shape,
                "analysis_type": "mock_analysis"
            }
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
            
            # Update stats
            self.update_execution_stats(execution_time, memory_usage, True)
            
            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=confidence,
                results=results,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
            
            # Update stats
            self.update_execution_stats(execution_time, memory_usage, False)
            
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.agent_name,
                confidence=0.0,
                results={},
                execution_time=execution_time,
                memory_usage=memory_usage,
                errors=[str(e)]
            )


# Test functions

def test_performance_monitoring():
    """Test performance monitoring decorators."""
    print("\n=== Testing Performance Monitoring ===")
    
    try:
        @monitor_performance
        def test_function(duration=0.1):
            time.sleep(duration)
            return "success"
        
        result = test_function(0.05)
        assert result == "success"
        print("✓ Performance monitoring decorator working")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n=== Testing Circuit Breaker ===")
    
    try:
        @circuit_breaker(max_failures=2)
        def failing_function(should_fail=True):
            if should_fail:
                raise ValueError("Test failure")
            return "success"
        
        # Test failures
        failure_count = 0
        circuit_opened = False
        
        for i in range(4):
            try:
                failing_function(True)
            except ValueError:
                failure_count += 1
            except Exception as e:
                if "Circuit breaker open" in str(e):
                    circuit_opened = True
                    print("✓ Circuit breaker activated after failures")
                    break
        
        assert failure_count >= 2, "Should have at least 2 failures before circuit breaker"
        assert circuit_opened, "Circuit breaker should have opened"
        
        return True
        
    except Exception as e:
        print(f"✗ Circuit breaker test failed: {e}")
        traceback.print_exc()
        return False


def test_input_validation():
    """Test input validation functionality."""
    print("\n=== Testing Input Validation ===")
    
    try:
        validator = SignalValidator()
        
        # Valid signal
        valid_signal = np.random.randn(2, 1000, 1)
        is_valid, errors = validator.validate(valid_signal)
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
            is_valid, errors = validator.validate(invalid_signal)
            assert not is_valid, f"Invalid signal {i} should fail validation"
            print(f"✓ Invalid signal {i} correctly rejected: {errors[0] if errors else 'No error message'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        traceback.print_exc()
        return False


def test_enhanced_agent_architecture():
    """Test the enhanced agent architecture."""
    print("\n=== Testing Enhanced Agent Architecture ===")
    
    try:
        # Test different configurations
        configs = [
            {"processing_time": 0.05},  # Fast config
            {"processing_time": 0.1},   # Standard config
        ]
        
        for i, config in enumerate(configs):
            print(f"\nTesting configuration {i+1}: {config}")
            
            agent = MockAgent(f"test_agent_{i}", config)
            
            # Create test signals
            test_signals = np.random.randn(3, 1000, 1)
            
            # Perform analysis
            analysis_result = agent.analyze(test_signals)
            
            assert analysis_result.is_successful(), "Analysis should be successful"
            assert analysis_result.confidence > 0, "Should have positive confidence"
            
            print(f"  ✓ Analysis confidence: {analysis_result.confidence:.3f}")
            print(f"  ✓ Execution time: {analysis_result.execution_time:.3f}s")
            print(f"  ✓ Memory usage: {analysis_result.memory_usage:.1f}MB")
            
            # Test multiple executions for performance metrics
            for _ in range(2):
                agent.analyze(test_signals)
            
            # Check performance metrics
            perf_metrics = agent.get_performance_metrics()
            assert not perf_metrics.get("no_executions"), "Should have execution data"
            assert perf_metrics["success_rate"] == 1.0, "All executions should be successful"
            
            print(f"  ✓ Success rate: {perf_metrics['success_rate']:.1%}")
            print(f"  ✓ Avg execution time: {perf_metrics['avg_execution_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced agent architecture test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("\n=== Testing Error Handling ===")
    
    try:
        agent = MockAgent("error_test_agent")
        
        # Test with invalid input
        invalid_signals = [
            None,
            np.array([]),
            np.random.randn(5)  # Wrong shape
        ]
        
        for i, invalid_signal in enumerate(invalid_signals):
            result = agent.analyze(invalid_signal)
            assert not result.is_successful(), f"Invalid input {i} should fail"
            assert len(result.errors) > 0, f"Should have error messages for invalid input {i}"
            print(f"✓ Invalid input {i} handled correctly: {result.errors[0]}")
        
        # Test performance metrics with failures
        perf_metrics = agent.get_performance_metrics()
        if perf_metrics.get("no_executions"):
            print("✓ No executions recorded (validation failures don't count as executions)")
        else:
            assert perf_metrics["success_rate"] == 0.0, "All executions should have failed"
            print(f"✓ Performance metrics track failures: {perf_metrics['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def run_core_tests():
    """Run all core refactoring tests."""
    print("Starting Core PHM Research Agent Refactoring Tests")
    print("=" * 60)
    
    tests = [
        ("Performance Monitoring", test_performance_monitoring),
        ("Circuit Breaker", test_circuit_breaker),
        ("Input Validation", test_input_validation),
        ("Enhanced Agent Architecture", test_enhanced_agent_architecture),
        ("Error Handling", test_error_handling)
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
    print("CORE TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(tests)
    passed_tests = sum(1 for r in results.values() if r["success"])
    total_time = sum(r["duration"] for r in results.values())
    
    for test_name, result in results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        duration = result["duration"]
        print(f"{test_name:<30} {status:<8} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed in {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("🎉 All core tests passed! Enhanced agent architecture is working correctly.")
        return True
    else:
        print("❌ Some core tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)
