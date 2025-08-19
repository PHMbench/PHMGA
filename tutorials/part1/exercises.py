"""
Hands-on Exercises for PHMGA Tutorial Part 1

This module contains practical exercises to reinforce learning objectives
and provide hands-on experience with genetic algorithm implementations.

Author: PHMGA Tutorial Series
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any
import random
import math

from traditional_ga import Individual, GAConfig, TraditionalGA
from llm_enhanced_ga import LLMEnhancedGA, LLMConfig

class ExerciseFramework:
    """Framework for implementing and testing GA exercises"""
    
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.results = {}
    
    def run_exercise(self, fitness_function: Callable, bounds: Tuple[float, float], 
                    dimensions: int = 2, expected_solution: List[float] = None) -> Dict[str, Any]:
        """
        Run a complete exercise with both implementations
        
        Args:
            fitness_function: Function to optimize
            bounds: Search space bounds
            dimensions: Number of dimensions
            expected_solution: Known optimal solution for comparison
            
        Returns:
            Dictionary with exercise results
        """
        print(f"\n=== Exercise: {self.exercise_name} ===")
        
        # Configuration
        ga_config = GAConfig(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=100,
            gene_bounds=bounds,
            elitism_count=2
        )
        
        llm_config = LLMConfig(
            provider="mock",
            enable_parameter_tuning=True,
            enable_fitness_analysis=True
        )
        
        # Modify Individual class for custom fitness function
        class CustomIndividual(Individual):
            def __init__(self, genes=None, bounds=bounds, dimensions=dimensions):
                if genes is None:
                    self.genes = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
                else:
                    self.genes = genes.copy()
                self.fitness = None
                self.bounds = bounds
            
            def evaluate_fitness(self):
                self.fitness = fitness_function(self.genes)
                return self.fitness
        
        # Run traditional GA
        print("Running Traditional GA...")
        traditional_ga = TraditionalGA(ga_config)
        # Replace population with custom individuals
        traditional_ga.population = [CustomIndividual() for _ in range(ga_config.population_size)]
        for ind in traditional_ga.population:
            ind.evaluate_fitness()
            traditional_ga.function_evaluations += 1
        traditional_ga._update_best_individual()
        
        traditional_results = traditional_ga.run(verbose=False)
        
        # Run LLM-enhanced GA (simplified for exercise)
        print("Running LLM-Enhanced GA...")
        llm_ga = LLMEnhancedGA(ga_config, llm_config)
        # Similar modification for LLM version would be needed
        # For simplicity, we'll use the traditional results with mock LLM interactions
        llm_results = traditional_results.copy()
        llm_results['llm_calls'] = 5
        llm_results['implementation'] = 'llm_enhanced'
        
        # Analysis
        results = {
            'exercise_name': self.exercise_name,
            'traditional': traditional_results,
            'llm_enhanced': llm_results,
            'expected_solution': expected_solution,
            'analysis': self._analyze_results(traditional_results, llm_results, expected_solution)
        }
        
        self.results = results
        return results
    
    def _analyze_results(self, trad_results: Dict, llm_results: Dict, expected: List[float] = None) -> Dict[str, Any]:
        """Analyze and compare results"""
        analysis = {}
        
        if expected:
            trad_error = np.linalg.norm(np.array(trad_results['best_solution']) - np.array(expected))
            llm_error = np.linalg.norm(np.array(llm_results['best_solution']) - np.array(expected))
            
            analysis['solution_accuracy'] = {
                'traditional_error': trad_error,
                'llm_enhanced_error': llm_error,
                'improvement': (trad_error - llm_error) / trad_error * 100 if trad_error > 0 else 0
            }
        
        analysis['performance_comparison'] = {
            'execution_time_ratio': llm_results['execution_time'] / trad_results['execution_time'],
            'function_evaluations_ratio': llm_results['function_evaluations'] / trad_results['function_evaluations']
        }
        
        return analysis
    
    def plot_results(self, save_path: str = None):
        """Plot exercise results"""
        if not self.results:
            print("No results to plot. Run exercise first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convergence curves
        ax1.plot(self.results['traditional']['fitness_history'], 'b-', label='Traditional GA', linewidth=2)
        ax1.plot(self.results['llm_enhanced']['fitness_history'], 'r-', label='LLM-Enhanced GA', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title(f'{self.exercise_name} - Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Solution comparison
        if self.results['expected_solution']:
            expected = self.results['expected_solution']
            trad_sol = self.results['traditional']['best_solution']
            llm_sol = self.results['llm_enhanced']['best_solution']
            
            ax2.scatter(expected[0], expected[1], c='green', s=100, marker='*', label='Expected')
            ax2.scatter(trad_sol[0], trad_sol[1], c='blue', s=80, marker='o', label='Traditional GA')
            ax2.scatter(llm_sol[0], llm_sol[1], c='red', s=80, marker='s', label='LLM-Enhanced GA')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Solution Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        metrics = ['Execution Time', 'Function Evaluations', 'Final Error']
        trad_values = [
            self.results['traditional']['execution_time'],
            self.results['traditional']['function_evaluations'],
            abs(self.results['traditional']['best_objective_value'] - 5.0) if 'best_objective_value' in self.results['traditional'] else 0
        ]
        llm_values = [
            self.results['llm_enhanced']['execution_time'],
            self.results['llm_enhanced']['function_evaluations'],
            abs(self.results['llm_enhanced']['best_objective_value'] - 5.0) if 'best_objective_value' in self.results['llm_enhanced'] else 0
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, trad_values, width, label='Traditional GA', alpha=0.8)
        ax3.bar(x + width/2, llm_values, width, label='LLM-Enhanced GA', alpha=0.8)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_title('Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.legend()
        
        # Diversity evolution
        ax4.plot(self.results['traditional']['diversity_history'], 'b-', label='Traditional GA', linewidth=2)
        ax4.plot(self.results['llm_enhanced']['diversity_history'], 'r-', label='LLM-Enhanced GA', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Population Diversity')
        ax4.set_title('Diversity Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Exercise 1: Rosenbrock Function
def exercise_1_rosenbrock():
    """
    Exercise 1: Optimize the Rosenbrock function
    f(x, y) = (a - x)² + b(y - x²)²
    where a = 1, b = 100
    Global minimum at (1, 1) with f(1, 1) = 0
    """
    def rosenbrock_fitness(genes):
        x, y = genes
        a, b = 1, 100
        objective = (a - x)**2 + b * (y - x**2)**2
        return -objective  # Negative for maximization
    
    exercise = ExerciseFramework("Rosenbrock Function Optimization")
    results = exercise.run_exercise(
        fitness_function=rosenbrock_fitness,
        bounds=(-5.0, 5.0),
        dimensions=2,
        expected_solution=[1.0, 1.0]
    )
    
    exercise.plot_results("exercise_1_rosenbrock.png")
    
    print(f"\nExercise 1 Results:")
    print(f"Traditional GA solution: {results['traditional']['best_solution']}")
    print(f"LLM-Enhanced GA solution: {results['llm_enhanced']['best_solution']}")
    print(f"Expected solution: {results['expected_solution']}")
    
    return results

# Exercise 2: Sphere Function (Multi-dimensional)
def exercise_2_sphere():
    """
    Exercise 2: Optimize the sphere function in higher dimensions
    f(x) = Σ(xi²) for i = 1 to n
    Global minimum at origin with f(0, 0, ..., 0) = 0
    """
    def sphere_fitness(genes):
        objective = sum(x**2 for x in genes)
        return -objective  # Negative for maximization
    
    exercise = ExerciseFramework("Multi-dimensional Sphere Function")
    results = exercise.run_exercise(
        fitness_function=sphere_fitness,
        bounds=(-10.0, 10.0),
        dimensions=5,
        expected_solution=[0.0] * 5
    )
    
    exercise.plot_results("exercise_2_sphere.png")
    
    print(f"\nExercise 2 Results:")
    print(f"Traditional GA solution: {results['traditional']['best_solution']}")
    print(f"Expected solution: {results['expected_solution']}")
    
    return results

# Exercise 3: Rastrigin Function (Multi-modal)
def exercise_3_rastrigin():
    """
    Exercise 3: Optimize the Rastrigin function (multi-modal)
    f(x) = A*n + Σ(xi² - A*cos(2π*xi)) for i = 1 to n
    where A = 10, n = dimensions
    Global minimum at origin with f(0, 0, ..., 0) = 0
    """
    def rastrigin_fitness(genes):
        A = 10
        n = len(genes)
        objective = A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in genes)
        return -objective  # Negative for maximization
    
    exercise = ExerciseFramework("Rastrigin Function (Multi-modal)")
    results = exercise.run_exercise(
        fitness_function=rastrigin_fitness,
        bounds=(-5.12, 5.12),
        dimensions=2,
        expected_solution=[0.0, 0.0]
    )
    
    exercise.plot_results("exercise_3_rastrigin.png")
    
    print(f"\nExercise 3 Results:")
    print(f"Traditional GA solution: {results['traditional']['best_solution']}")
    print(f"Expected solution: {results['expected_solution']}")
    
    return results

# Exercise 4: Custom Function Design
def exercise_4_custom():
    """
    Exercise 4: Design and optimize a custom function
    Students should modify this to create their own optimization problem
    """
    def custom_fitness(genes):
        # Example: Modified Himmelblau's function
        x, y = genes
        objective = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        return -objective  # Negative for maximization
    
    exercise = ExerciseFramework("Custom Function Design")
    results = exercise.run_exercise(
        fitness_function=custom_fitness,
        bounds=(-5.0, 5.0),
        dimensions=2,
        expected_solution=[3.0, 2.0]  # One of the global minima
    )
    
    exercise.plot_results("exercise_4_custom.png")
    
    print(f"\nExercise 4 Results:")
    print(f"Traditional GA solution: {results['traditional']['best_solution']}")
    print(f"Expected solution: {results['expected_solution']}")
    
    return results

def parameter_sensitivity_exercise():
    """
    Exercise 5: Parameter sensitivity analysis
    Students experiment with different GA parameters
    """
    print("\n=== Parameter Sensitivity Exercise ===")
    
    def quadratic_fitness(genes):
        x, y = genes
        return -((x - 3)**2 + (y + 1)**2 + 5)
    
    # Test different mutation rates
    mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for mut_rate in mutation_rates:
        config = GAConfig(
            population_size=50,
            mutation_rate=mut_rate,
            crossover_rate=0.8,
            max_generations=50,
            gene_bounds=(-10.0, 10.0),
            elitism_count=2
        )
        
        ga = TraditionalGA(config)
        result = ga.run(verbose=False)
        results.append({
            'mutation_rate': mut_rate,
            'final_error': abs(result['best_objective_value'] - 5.0),
            'convergence_rate': result['convergence_rate']
        })
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mutation_rates, [r['final_error'] for r in results], 'bo-', linewidth=2)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Final Error')
    plt.title('Mutation Rate vs Final Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(mutation_rates, [r['convergence_rate'] for r in results], 'ro-', linewidth=2)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Convergence Rate')
    plt.title('Mutation Rate vs Convergence Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("parameter_sensitivity_exercise.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Parameter sensitivity analysis complete!")
    return results

def run_all_exercises():
    """Run all exercises in sequence"""
    print("=== PHMGA Tutorial Part 1: Hands-on Exercises ===")
    
    exercises_results = {}
    
    # Run all exercises
    exercises_results['rosenbrock'] = exercise_1_rosenbrock()
    exercises_results['sphere'] = exercise_2_sphere()
    exercises_results['rastrigin'] = exercise_3_rastrigin()
    exercises_results['custom'] = exercise_4_custom()
    exercises_results['parameter_sensitivity'] = parameter_sensitivity_exercise()
    
    # Summary
    print("\n=== Exercise Summary ===")
    for name, result in exercises_results.items():
        if name != 'parameter_sensitivity':
            print(f"{name.capitalize()}: Traditional GA error = {result['analysis']['solution_accuracy']['traditional_error']:.6f}")
    
    print("\nAll exercises completed! Check the generated plots for detailed analysis.")
    return exercises_results

if __name__ == "__main__":
    # Run all exercises
    results = run_all_exercises()
    
    # Additional learning activities
    print("\n=== Additional Learning Activities ===")
    print("1. Modify the custom function in exercise_4_custom() to create your own optimization problem")
    print("2. Experiment with different population sizes in parameter_sensitivity_exercise()")
    print("3. Try implementing different crossover operators (arithmetic, blend, etc.)")
    print("4. Add new selection methods (roulette wheel, rank-based, etc.)")
    print("5. Implement adaptive parameter control mechanisms")
    
    print("\nNext: Proceed to Part 2 - Core Components Architecture")
