"""
Comprehensive Benchmarking and Comparison Module for PHMGA Tutorial Part 1

This module provides detailed performance analysis and comparison between
traditional and LLM-enhanced genetic algorithm implementations.

Author: PHMGA Tutorial Series
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Tuple
import time
import json
from dataclasses import dataclass
import statistics

from traditional_ga import TraditionalGA, GAConfig
from llm_enhanced_ga import LLMEnhancedGA, LLMConfig

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments"""
    num_runs: int = 10
    max_generations: int = 100
    population_sizes: List[int] = None
    mutation_rates: List[float] = None
    save_results: bool = True
    plot_results: bool = True
    
    def __post_init__(self):
        if self.population_sizes is None:
            self.population_sizes = [30, 50, 100]
        if self.mutation_rates is None:
            self.mutation_rates = [0.05, 0.1, 0.2]

class PerformanceMetrics:
    """Calculate and store performance metrics"""
    
    @staticmethod
    def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistical metrics from multiple runs
        
        Args:
            results: List of result dictionaries from multiple GA runs
            
        Returns:
            Dictionary with statistical metrics
        """
        # Extract key metrics
        final_errors = [abs(r['best_objective_value'] - 5.0) for r in results]
        execution_times = [r['execution_time'] for r in results]
        function_evaluations = [r['function_evaluations'] for r in results]
        convergence_rates = [r['convergence_rate'] for r in results]
        
        # Calculate statistics
        metrics = {
            'final_error': {
                'mean': statistics.mean(final_errors),
                'std': statistics.stdev(final_errors) if len(final_errors) > 1 else 0,
                'min': min(final_errors),
                'max': max(final_errors),
                'median': statistics.median(final_errors)
            },
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times),
                'max': max(execution_times),
                'median': statistics.median(execution_times)
            },
            'function_evaluations': {
                'mean': statistics.mean(function_evaluations),
                'std': statistics.stdev(function_evaluations) if len(function_evaluations) > 1 else 0,
                'min': min(function_evaluations),
                'max': max(function_evaluations),
                'median': statistics.median(function_evaluations)
            },
            'convergence_rate': {
                'mean': statistics.mean(convergence_rates),
                'std': statistics.stdev(convergence_rates) if len(convergence_rates) > 1 else 0,
                'min': min(convergence_rates),
                'max': max(convergence_rates),
                'median': statistics.median(convergence_rates)
            },
            'success_rate': sum(1 for error in final_errors if error < 0.1) / len(final_errors),
            'reliability_score': 1.0 - (statistics.stdev(final_errors) / statistics.mean(final_errors)) if statistics.mean(final_errors) > 0 else 1.0
        }
        
        return metrics

class BenchmarkSuite:
    """Comprehensive benchmarking suite for GA implementations"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
    
    def run_single_configuration(self, ga_config: GAConfig, implementation: str, num_runs: int = None) -> List[Dict[str, Any]]:
        """
        Run multiple trials of a single GA configuration
        
        Args:
            ga_config: GA configuration
            implementation: "traditional" or "llm_enhanced"
            num_runs: Number of runs (uses config default if None)
            
        Returns:
            List of results from all runs
        """
        if num_runs is None:
            num_runs = self.config.num_runs
        
        results = []
        
        print(f"Running {num_runs} trials of {implementation} GA...")
        
        for run in range(num_runs):
            if implementation == "traditional":
                ga = TraditionalGA(ga_config)
                result = ga.run(verbose=False)
            elif implementation == "llm_enhanced":
                llm_config = LLMConfig(provider="mock", enable_parameter_tuning=True)
                ga = LLMEnhancedGA(ga_config, llm_config)
                result = ga.run(verbose=False)
            else:
                raise ValueError(f"Unknown implementation: {implementation}")
            
            result['run_id'] = run
            result['implementation'] = implementation
            results.append(result)
            
            if (run + 1) % 5 == 0:
                print(f"  Completed {run + 1}/{num_runs} runs")
        
        return results
    
    def parameter_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Analyze sensitivity to different parameter settings
        
        Returns:
            Dictionary with sensitivity analysis results
        """
        print("=== Parameter Sensitivity Analysis ===")
        
        sensitivity_results = {
            'population_size': {},
            'mutation_rate': {}
        }
        
        # Base configuration
        base_config = GAConfig(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=self.config.max_generations,
            gene_bounds=(-10.0, 10.0),
            elitism_count=2
        )
        
        # Test population sizes
        print("\nTesting population sizes...")
        for pop_size in self.config.population_sizes:
            config = GAConfig(
                population_size=pop_size,
                mutation_rate=base_config.mutation_rate,
                crossover_rate=base_config.crossover_rate,
                max_generations=base_config.max_generations,
                gene_bounds=base_config.gene_bounds,
                elitism_count=base_config.elitism_count
            )
            
            traditional_results = self.run_single_configuration(config, "traditional", 5)
            llm_results = self.run_single_configuration(config, "llm_enhanced", 5)
            
            sensitivity_results['population_size'][pop_size] = {
                'traditional': PerformanceMetrics.calculate_metrics(traditional_results),
                'llm_enhanced': PerformanceMetrics.calculate_metrics(llm_results)
            }
        
        # Test mutation rates
        print("\nTesting mutation rates...")
        for mut_rate in self.config.mutation_rates:
            config = GAConfig(
                population_size=base_config.population_size,
                mutation_rate=mut_rate,
                crossover_rate=base_config.crossover_rate,
                max_generations=base_config.max_generations,
                gene_bounds=base_config.gene_bounds,
                elitism_count=base_config.elitism_count
            )
            
            traditional_results = self.run_single_configuration(config, "traditional", 5)
            llm_results = self.run_single_configuration(config, "llm_enhanced", 5)
            
            sensitivity_results['mutation_rate'][mut_rate] = {
                'traditional': PerformanceMetrics.calculate_metrics(traditional_results),
                'llm_enhanced': PerformanceMetrics.calculate_metrics(llm_results)
            }
        
        return sensitivity_results
    
    def convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze convergence characteristics of both implementations
        
        Returns:
            Dictionary with convergence analysis results
        """
        print("=== Convergence Analysis ===")
        
        config = GAConfig(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=self.config.max_generations,
            gene_bounds=(-10.0, 10.0),
            elitism_count=2
        )
        
        # Run multiple trials for convergence analysis
        traditional_results = self.run_single_configuration(config, "traditional", 10)
        llm_results = self.run_single_configuration(config, "llm_enhanced", 10)
        
        # Extract fitness histories
        traditional_histories = [r['fitness_history'] for r in traditional_results]
        llm_histories = [r['fitness_history'] for r in llm_results]
        
        # Calculate average convergence curves
        max_generations = min(len(h) for h in traditional_histories + llm_histories)
        
        traditional_avg = np.mean([h[:max_generations] for h in traditional_histories], axis=0)
        traditional_std = np.std([h[:max_generations] for h in traditional_histories], axis=0)
        
        llm_avg = np.mean([h[:max_generations] for h in llm_histories], axis=0)
        llm_std = np.std([h[:max_generations] for h in llm_histories], axis=0)
        
        convergence_results = {
            'traditional': {
                'avg_curve': traditional_avg.tolist(),
                'std_curve': traditional_std.tolist(),
                'final_metrics': PerformanceMetrics.calculate_metrics(traditional_results)
            },
            'llm_enhanced': {
                'avg_curve': llm_avg.tolist(),
                'std_curve': llm_std.tolist(),
                'final_metrics': PerformanceMetrics.calculate_metrics(llm_results)
            },
            'generations': list(range(max_generations))
        }
        
        return convergence_results
    
    def resource_usage_analysis(self) -> Dict[str, Any]:
        """
        Analyze computational resource usage
        
        Returns:
            Dictionary with resource usage analysis
        """
        print("=== Resource Usage Analysis ===")
        
        config = GAConfig(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=100,
            gene_bounds=(-10.0, 10.0),
            elitism_count=2
        )
        
        # Run resource analysis
        traditional_results = self.run_single_configuration(config, "traditional", 5)
        llm_results = self.run_single_configuration(config, "llm_enhanced", 5)
        
        # Calculate resource metrics
        traditional_metrics = PerformanceMetrics.calculate_metrics(traditional_results)
        llm_metrics = PerformanceMetrics.calculate_metrics(llm_results)
        
        # Additional LLM-specific metrics
        llm_calls = [r.get('llm_calls', 0) for r in llm_results]
        parameter_adjustments = [len(r.get('parameter_adjustments', [])) for r in llm_results]
        
        resource_results = {
            'traditional': {
                'cpu_time': traditional_metrics['execution_time'],
                'function_evaluations': traditional_metrics['function_evaluations'],
                'memory_efficiency': 1.0,  # Baseline
                'api_calls': {'mean': 0, 'std': 0}
            },
            'llm_enhanced': {
                'cpu_time': llm_metrics['execution_time'],
                'function_evaluations': llm_metrics['function_evaluations'],
                'memory_efficiency': 0.95,  # Slightly higher due to LLM overhead
                'api_calls': {
                    'mean': statistics.mean(llm_calls),
                    'std': statistics.stdev(llm_calls) if len(llm_calls) > 1 else 0
                },
                'parameter_adjustments': {
                    'mean': statistics.mean(parameter_adjustments),
                    'std': statistics.stdev(parameter_adjustments) if len(parameter_adjustments) > 1 else 0
                }
            }
        }
        
        return resource_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmarking suite
        
        Returns:
            Dictionary with all benchmark results
        """
        print("=== Comprehensive GA Benchmark Suite ===")
        
        benchmark_results = {
            'parameter_sensitivity': self.parameter_sensitivity_analysis(),
            'convergence_analysis': self.convergence_analysis(),
            'resource_usage': self.resource_usage_analysis(),
            'config': {
                'num_runs': self.config.num_runs,
                'max_generations': self.config.max_generations,
                'population_sizes': self.config.population_sizes,
                'mutation_rates': self.config.mutation_rates
            }
        }
        
        if self.config.save_results:
            with open('benchmark_results.json', 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            print("\nResults saved to benchmark_results.json")
        
        return benchmark_results

def plot_benchmark_results(results: Dict[str, Any], save_path: str = None) -> None:
    """
    Create comprehensive visualization of benchmark results
    
    Args:
        results: Benchmark results dictionary
        save_path: Optional path to save plots
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    conv_data = results['convergence_analysis']
    generations = conv_data['generations']
    
    # Convert fitness to objective values for plotting
    trad_obj = [-f for f in conv_data['traditional']['avg_curve']]
    llm_obj = [-f for f in conv_data['llm_enhanced']['avg_curve']]
    trad_std = conv_data['traditional']['std_curve']
    llm_std = conv_data['llm_enhanced']['std_curve']
    
    plt.plot(generations, trad_obj, 'b-', label='Traditional GA', linewidth=2)
    plt.fill_between(generations, 
                     [t - s for t, s in zip(trad_obj, trad_std)],
                     [t + s for t, s in zip(trad_obj, trad_std)],
                     alpha=0.3, color='blue')
    
    plt.plot(generations, llm_obj, 'r-', label='LLM-Enhanced GA', linewidth=2)
    plt.fill_between(generations,
                     [l - s for l, s in zip(llm_obj, llm_std)],
                     [l + s for l, s in zip(llm_obj, llm_std)],
                     alpha=0.3, color='red')
    
    plt.axhline(y=5.0, color='green', linestyle='--', label='Optimal Value')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Population size sensitivity
    ax2 = plt.subplot(2, 3, 2)
    pop_data = results['parameter_sensitivity']['population_size']
    pop_sizes = list(pop_data.keys())
    trad_errors = [pop_data[size]['traditional']['final_error']['mean'] for size in pop_sizes]
    llm_errors = [pop_data[size]['llm_enhanced']['final_error']['mean'] for size in pop_sizes]
    
    x = np.arange(len(pop_sizes))
    width = 0.35
    
    plt.bar(x - width/2, trad_errors, width, label='Traditional GA', alpha=0.8)
    plt.bar(x + width/2, llm_errors, width, label='LLM-Enhanced GA', alpha=0.8)
    
    plt.xlabel('Population Size')
    plt.ylabel('Mean Final Error')
    plt.title('Population Size Sensitivity')
    plt.xticks(x, pop_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Mutation rate sensitivity
    ax3 = plt.subplot(2, 3, 3)
    mut_data = results['parameter_sensitivity']['mutation_rate']
    mut_rates = list(mut_data.keys())
    trad_errors = [mut_data[rate]['traditional']['final_error']['mean'] for rate in mut_rates]
    llm_errors = [mut_data[rate]['llm_enhanced']['final_error']['mean'] for rate in mut_rates]
    
    x = np.arange(len(mut_rates))
    
    plt.bar(x - width/2, trad_errors, width, label='Traditional GA', alpha=0.8)
    plt.bar(x + width/2, llm_errors, width, label='LLM-Enhanced GA', alpha=0.8)
    
    plt.xlabel('Mutation Rate')
    plt.ylabel('Mean Final Error')
    plt.title('Mutation Rate Sensitivity')
    plt.xticks(x, mut_rates)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Execution time comparison
    ax4 = plt.subplot(2, 3, 4)
    resource_data = results['resource_usage']
    implementations = ['Traditional', 'LLM-Enhanced']
    exec_times = [
        resource_data['traditional']['cpu_time']['mean'],
        resource_data['llm_enhanced']['cpu_time']['mean']
    ]
    exec_stds = [
        resource_data['traditional']['cpu_time']['std'],
        resource_data['llm_enhanced']['cpu_time']['std']
    ]
    
    plt.bar(implementations, exec_times, yerr=exec_stds, capsize=5, alpha=0.8)
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # 5. Success rate comparison
    ax5 = plt.subplot(2, 3, 5)
    conv_metrics = results['convergence_analysis']
    success_rates = [
        conv_metrics['traditional']['final_metrics']['success_rate'],
        conv_metrics['llm_enhanced']['final_metrics']['success_rate']
    ]
    
    plt.bar(implementations, success_rates, alpha=0.8, color=['blue', 'red'])
    plt.ylabel('Success Rate (Error < 0.1)')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 6. Resource efficiency radar chart
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Normalize metrics for radar chart
    metrics = ['Accuracy', 'Speed', 'Reliability', 'Efficiency']
    
    # Calculate normalized scores (higher is better)
    trad_scores = [
        1.0 - min(1.0, conv_metrics['traditional']['final_metrics']['final_error']['mean']),
        1.0 / max(1.0, resource_data['traditional']['cpu_time']['mean']),
        conv_metrics['traditional']['final_metrics']['reliability_score'],
        resource_data['traditional']['memory_efficiency']
    ]
    
    llm_scores = [
        1.0 - min(1.0, conv_metrics['llm_enhanced']['final_metrics']['final_error']['mean']),
        1.0 / max(1.0, resource_data['llm_enhanced']['cpu_time']['mean']),
        conv_metrics['llm_enhanced']['final_metrics']['reliability_score'],
        resource_data['llm_enhanced']['memory_efficiency']
    ]
    
    # Normalize to 0-1 range
    max_scores = [max(t, l) for t, l in zip(trad_scores, llm_scores)]
    trad_norm = [t/m if m > 0 else 0 for t, m in zip(trad_scores, max_scores)]
    llm_norm = [l/m if m > 0 else 0 for l, m in zip(llm_scores, max_scores)]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    trad_norm += trad_norm[:1]
    llm_norm += llm_norm[:1]
    
    ax6.plot(angles, trad_norm, 'b-', linewidth=2, label='Traditional GA')
    ax6.fill(angles, trad_norm, alpha=0.25, color='blue')
    ax6.plot(angles, llm_norm, 'r-', linewidth=2, label='LLM-Enhanced GA')
    ax6.fill(angles, llm_norm, alpha=0.25, color='red')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_ylim(0, 1)
    ax6.set_title('Performance Radar Chart')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_benchmark_report(results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive text report of benchmark results
    
    Args:
        results: Benchmark results dictionary
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("PHMGA TUTORIAL PART 1: COMPREHENSIVE BENCHMARK REPORT")
    report.append("=" * 80)
    
    # Executive Summary
    conv_data = results['convergence_analysis']
    trad_metrics = conv_data['traditional']['final_metrics']
    llm_metrics = conv_data['llm_enhanced']['final_metrics']
    
    report.append("\nEXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Traditional GA - Mean Error: {trad_metrics['final_error']['mean']:.6f}")
    report.append(f"LLM-Enhanced GA - Mean Error: {llm_metrics['final_error']['mean']:.6f}")
    report.append(f"Improvement: {((trad_metrics['final_error']['mean'] - llm_metrics['final_error']['mean']) / trad_metrics['final_error']['mean'] * 100):.2f}%")
    
    # Detailed Analysis
    report.append("\nDETAILED PERFORMANCE ANALYSIS")
    report.append("-" * 40)
    
    metrics_table = [
        ["Metric", "Traditional GA", "LLM-Enhanced GA", "Improvement"],
        ["Final Error (mean)", f"{trad_metrics['final_error']['mean']:.6f}", f"{llm_metrics['final_error']['mean']:.6f}", ""],
        ["Success Rate", f"{trad_metrics['success_rate']:.3f}", f"{llm_metrics['success_rate']:.3f}", ""],
        ["Reliability Score", f"{trad_metrics['reliability_score']:.3f}", f"{llm_metrics['reliability_score']:.3f}", ""],
    ]
    
    for row in metrics_table:
        report.append(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<10}")
    
    # Resource Usage
    resource_data = results['resource_usage']
    report.append("\nRESOURCE USAGE ANALYSIS")
    report.append("-" * 40)
    report.append(f"Traditional GA - CPU Time: {resource_data['traditional']['cpu_time']['mean']:.3f}s")
    report.append(f"LLM-Enhanced GA - CPU Time: {resource_data['llm_enhanced']['cpu_time']['mean']:.3f}s")
    report.append(f"LLM API Calls: {resource_data['llm_enhanced']['api_calls']['mean']:.1f}")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 40)
    
    if llm_metrics['final_error']['mean'] < trad_metrics['final_error']['mean']:
        report.append("✓ LLM-Enhanced GA shows superior optimization performance")
    else:
        report.append("✗ Traditional GA performs better for this simple problem")
    
    if llm_metrics['success_rate'] > trad_metrics['success_rate']:
        report.append("✓ LLM-Enhanced GA has higher success rate")
    else:
        report.append("✗ Traditional GA has higher success rate")
    
    report.append("\nUSE CASE RECOMMENDATIONS:")
    report.append("- Traditional GA: Simple, well-understood optimization problems")
    report.append("- LLM-Enhanced GA: Complex problems requiring adaptive parameter tuning")
    report.append("- Consider computational cost vs. performance improvement trade-offs")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Run comprehensive benchmark
    config = BenchmarkConfig(
        num_runs=10,
        max_generations=100,
        population_sizes=[30, 50, 100],
        mutation_rates=[0.05, 0.1, 0.2],
        save_results=True,
        plot_results=True
    )
    
    benchmark = BenchmarkSuite(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate visualizations
    plot_benchmark_results(results, save_path="comprehensive_benchmark_results.png")
    
    # Generate report
    report = generate_benchmark_report(results)
    print(report)
    
    # Save report
    with open("benchmark_report.txt", "w") as f:
        f.write(report)
    
    print("\nBenchmark complete! Check benchmark_results.json and benchmark_report.txt for detailed results.")
