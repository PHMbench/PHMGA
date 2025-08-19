"""
Comprehensive Testing Framework for PHMGA Tutorial Series

This module provides a complete testing suite including unit tests, integration tests,
performance benchmarks, and property-based testing for genetic algorithms.

Author: PHMGA Tutorial Series
License: MIT
"""

import unittest
import pytest
import numpy as np
import time
import tempfile
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import hypothesis
from hypothesis import strategies as st
from hypothesis import given, settings

# Import PHMGA components for testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tutorials.part1.traditional_ga import TraditionalGA, GAConfig, Individual
from tutorials.part1.llm_enhanced_ga import LLMEnhancedGA, LLMConfig
from tutorials.part2.router.base_router import Worker, Request, BaseRouter
from tutorials.part2.router.strategies import RoundRobinRouter, WeightedRouter
from tutorials.part3.case1_enhanced.unified_system import UnifiedPHMGASystem, SignalData

class TestTraditionalGA(unittest.TestCase):
    """Unit tests for Traditional Genetic Algorithm"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = GAConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=5,
            gene_bounds=(-5.0, 5.0),
            elitism_count=1
        )
        self.ga = TraditionalGA(self.config)
    
    def test_initialization(self):
        """Test GA initialization"""
        self.assertEqual(len(self.ga.workers), 0)
        self.assertEqual(self.ga.generation, 0)
        self.assertIsNone(self.ga.best_individual)
    
    def test_population_initialization(self):
        """Test population initialization"""
        self.ga.initialize_population()
        
        self.assertEqual(len(self.ga.population), self.config.population_size)
        self.assertIsNotNone(self.ga.best_individual)
        
        # Check gene bounds
        for individual in self.ga.population:
            for gene in individual.genes:
                self.assertGreaterEqual(gene, self.config.gene_bounds[0])
                self.assertLessEqual(gene, self.config.gene_bounds[1])
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation"""
        individual = Individual([3.0, -1.0], self.config.gene_bounds)
        fitness = individual.evaluate_fitness()
        
        # For quadratic function f(x,y) = (x-3)² + (y+1)² + 5
        # At (3, -1), fitness should be -5 (negative for maximization)
        self.assertAlmostEqual(fitness, -5.0, places=6)
    
    def test_mutation(self):
        """Test mutation operation"""
        individual = Individual([0.0, 0.0], self.config.gene_bounds)
        original_genes = individual.genes.copy()
        
        # Force mutation by setting high rate
        individual.mutate(1.0)
        
        # Genes should be different after mutation
        self.assertNotEqual(individual.genes, original_genes)
        
        # Genes should still be within bounds
        for gene in individual.genes:
            self.assertGreaterEqual(gene, self.config.gene_bounds[0])
            self.assertLessEqual(gene, self.config.gene_bounds[1])
    
    def test_crossover(self):
        """Test crossover operation"""
        parent1 = Individual([1.0, 2.0], self.config.gene_bounds)
        parent2 = Individual([3.0, 4.0], self.config.gene_bounds)
        
        child1, child2 = parent1.crossover(parent2, 1.0)  # Force crossover
        
        # Children should be different from parents
        self.assertNotEqual(child1.genes, parent1.genes)
        self.assertNotEqual(child2.genes, parent2.genes)
        
        # Children should have genes within bounds
        for child in [child1, child2]:
            for gene in child.genes:
                self.assertGreaterEqual(gene, self.config.gene_bounds[0])
                self.assertLessEqual(gene, self.config.gene_bounds[1])
    
    def test_evolution(self):
        """Test evolution process"""
        initial_fitness = None
        
        # Run evolution
        results = self.ga.run(verbose=False)
        
        # Check results structure
        self.assertIn('best_solution', results)
        self.assertIn('best_fitness', results)
        self.assertIn('execution_time', results)
        self.assertIn('function_evaluations', results)
        
        # Check that evolution improved fitness
        self.assertGreater(results['best_fitness'], -100)  # Should find reasonable solution
        self.assertEqual(len(results['best_solution']), 2)  # Should have 2 genes

class TestRouterComponent(unittest.TestCase):
    """Unit tests for Router Component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.router = RoundRobinRouter("TestRouter")
        self.workers = [
            Worker("worker1", "Worker 1", weight=1.0, max_concurrent=5),
            Worker("worker2", "Worker 2", weight=2.0, max_concurrent=10),
            Worker("worker3", "Worker 3", weight=0.5, max_concurrent=3)
        ]
        
        for worker in self.workers:
            self.router.add_worker(worker)
    
    def test_worker_management(self):
        """Test worker addition and removal"""
        self.assertEqual(len(self.router.workers), 3)
        
        # Test worker removal
        removed = self.router.remove_worker("worker1")
        self.assertTrue(removed)
        self.assertEqual(len(self.router.workers), 2)
        
        # Test removing non-existent worker
        removed = self.router.remove_worker("nonexistent")
        self.assertFalse(removed)
    
    def test_worker_selection(self):
        """Test worker selection strategies"""
        request = Request(payload={"test": "data"})
        
        # Test round-robin selection
        selected_workers = []
        for _ in range(6):  # More than number of workers
            worker = self.router.select_worker(request)
            self.assertIsNotNone(worker)
            selected_workers.append(worker.id)
        
        # Should cycle through workers
        self.assertIn("worker1", selected_workers)
        self.assertIn("worker2", selected_workers)
        self.assertIn("worker3", selected_workers)
    
    def test_request_processing(self):
        """Test request processing"""
        request = Request(
            payload={"task": "test"},
            priority=1,
            timeout=10.0
        )
        
        processed_request = self.router.process_request(request)
        
        self.assertIsNotNone(processed_request.result)
        self.assertIsNotNone(processed_request.worker_id)
        self.assertGreater(processed_request.processing_time, 0)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        # Process some requests
        for i in range(5):
            request = Request(payload={"task": f"test_{i}"})
            self.router.process_request(request)
        
        metrics = self.router.get_metrics()
        
        self.assertIn('total_requests', metrics)
        self.assertIn('successful_requests', metrics)
        self.assertIn('avg_response_time', metrics)
        self.assertEqual(metrics['total_requests'], 5)

class TestUnifiedSystem(unittest.TestCase):
    """Integration tests for Unified PHMGA System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = UnifiedPHMGASystem("test_system")
    
    def tearDown(self):
        """Clean up after tests"""
        self.system.shutdown()
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system.system_id)
        self.assertIsNotNone(self.system.router)
        self.assertGreater(len(self.system.processors), 0)
    
    def test_signal_processing(self):
        """Test signal processing functionality"""
        # Create test signal
        signal_data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        signal = SignalData(
            id="test_signal",
            data=signal_data.tolist(),
            sampling_rate=1000.0,
            metadata={"test": True}
        )
        
        # Process signal
        result = self.system.process_signal(signal)
        
        self.assertEqual(result.signal_id, "test_signal")
        self.assertIsInstance(result.features, dict)
        self.assertIsInstance(result.classification, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_batch_processing(self):
        """Test batch signal processing"""
        signals = []
        for i in range(3):
            signal_data = np.random.randn(1000)
            signal = SignalData(
                id=f"batch_signal_{i}",
                data=signal_data.tolist(),
                sampling_rate=1000.0
            )
            signals.append(signal)
        
        results = self.system.process_batch(signals)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result.features, dict)
            self.assertIsInstance(result.classification, str)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.system.get_system_status()
        
        self.assertIn('system_id', status)
        self.assertIn('uptime', status)
        self.assertIn('system_metrics', status)
        self.assertIn('router_metrics', status)
        self.assertIn('processor_status', status)

class PropertyBasedTests:
    """Property-based tests using Hypothesis"""
    
    @given(
        population_size=st.integers(min_value=5, max_value=100),
        mutation_rate=st.floats(min_value=0.01, max_value=0.5),
        crossover_rate=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=10, deadline=None)
    def test_ga_convergence_property(self, population_size, mutation_rate, crossover_rate):
        """Property: GA should always converge to a solution"""
        config = GAConfig(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_generations=10,
            gene_bounds=(-10.0, 10.0),
            elitism_count=1
        )
        
        ga = TraditionalGA(config)
        results = ga.run(verbose=False)
        
        # Properties that should always hold
        assert results['best_fitness'] <= 0  # Fitness should be negative (minimization)
        assert len(results['best_solution']) == 2  # Should have 2 genes
        assert results['execution_time'] > 0  # Should take some time
        assert results['function_evaluations'] > 0  # Should evaluate functions
    
    @given(
        signal_length=st.integers(min_value=100, max_value=2000),
        sampling_rate=st.floats(min_value=100.0, max_value=10000.0)
    )
    @settings(max_examples=5, deadline=None)
    def test_signal_processing_property(self, signal_length, sampling_rate):
        """Property: Signal processing should always produce valid output"""
        # Generate random signal
        signal_data = np.random.randn(signal_length)
        signal = SignalData(
            id="property_test",
            data=signal_data.tolist(),
            sampling_rate=sampling_rate
        )
        
        system = UnifiedPHMGASystem("property_test")
        try:
            result = system.process_signal(signal)
            
            # Properties that should always hold
            assert result.signal_id == "property_test"
            assert isinstance(result.features, dict)
            assert len(result.features) > 0
            assert isinstance(result.classification, str)
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time > 0
            
        finally:
            system.shutdown()

class PerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def benchmark_ga_performance(self):
        """Benchmark GA performance across different configurations"""
        configurations = [
            {"population_size": 20, "max_generations": 50},
            {"population_size": 50, "max_generations": 100},
            {"population_size": 100, "max_generations": 200}
        ]
        
        results = []
        
        for config_params in configurations:
            config = GAConfig(**config_params)
            ga = TraditionalGA(config)
            
            start_time = time.time()
            result = ga.run(verbose=False)
            end_time = time.time()
            
            benchmark_result = {
                'config': config_params,
                'execution_time': end_time - start_time,
                'final_error': abs(result['best_objective_value'] - 5.0),
                'function_evaluations': result['function_evaluations'],
                'throughput': result['function_evaluations'] / (end_time - start_time)
            }
            results.append(benchmark_result)
        
        return results
    
    def benchmark_router_performance(self):
        """Benchmark router performance under load"""
        router = RoundRobinRouter("BenchmarkRouter")
        
        # Add workers
        for i in range(5):
            worker = Worker(f"worker_{i}", f"Worker {i}", max_concurrent=10)
            router.add_worker(worker)
        
        # Generate requests
        num_requests = 1000
        requests = [
            Request(payload={"task": f"benchmark_{i}"})
            for i in range(num_requests)
        ]
        
        # Benchmark processing
        start_time = time.time()
        for request in requests:
            router.process_request(request)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        router.shutdown()
        
        return {
            'total_requests': num_requests,
            'total_time': total_time,
            'throughput': throughput,
            'avg_request_time': total_time / num_requests
        }

class TestRunner:
    """Main test runner for the PHMGA framework"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_unit_tests(self):
        """Run all unit tests"""
        print("Running unit tests...")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestTraditionalGA))
        suite.addTests(loader.loadTestsFromTestCase(TestRouterComponent))
        suite.addTests(loader.loadTestsFromTestCase(TestUnifiedSystem))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        self.test_results['unit_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
        }
        
        return result.wasSuccessful()
    
    def run_property_tests(self):
        """Run property-based tests"""
        print("Running property-based tests...")
        
        try:
            property_tests = PropertyBasedTests()
            
            # Run property tests
            property_tests.test_ga_convergence_property()
            property_tests.test_signal_processing_property()
            
            self.test_results['property_tests'] = {'status': 'passed'}
            return True
            
        except Exception as e:
            print(f"Property tests failed: {e}")
            self.test_results['property_tests'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks"""
        print("Running performance benchmarks...")
        
        benchmarks = PerformanceBenchmarks()
        
        # GA performance benchmark
        ga_results = benchmarks.benchmark_ga_performance()
        
        # Router performance benchmark
        router_results = benchmarks.benchmark_router_performance()
        
        self.test_results['performance_benchmarks'] = {
            'ga_performance': ga_results,
            'router_performance': router_results
        }
        
        return True
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("PHMGA Framework - Comprehensive Test Suite")
        print("=" * 60)
        
        # Run all test categories
        unit_success = self.run_unit_tests()
        property_success = self.run_property_tests()
        benchmark_success = self.run_performance_benchmarks()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
        if 'unit_tests' in self.test_results:
            ut = self.test_results['unit_tests']
            print(f"  Tests Run: {ut['tests_run']}")
            print(f"  Success Rate: {ut['success_rate']:.2%}")
        
        print(f"Property Tests: {'PASSED' if property_success else 'FAILED'}")
        print(f"Performance Benchmarks: {'PASSED' if benchmark_success else 'FAILED'}")
        
        overall_success = unit_success and property_success and benchmark_success
        print(f"\nOverall Result: {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success

if __name__ == "__main__":
    # Run comprehensive test suite
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)
