"""
Traditional Genetic Algorithm Implementation for PHMGA Tutorial Part 1

This module demonstrates a pure Python implementation of a genetic algorithm
for optimizing the quadratic function: f(x, y) = (x - 3)² + (y + 1)² + 5

Author: PHMGA Tutorial Series
License: MIT
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time
import json

@dataclass
class GAConfig:
    """Configuration for genetic algorithm parameters"""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 100
    gene_bounds: Tuple[float, float] = (-10.0, 10.0)
    elitism_count: int = 2
    tournament_size: int = 3

class Individual:
    """
    Represents a candidate solution with genes and fitness
    
    Attributes:
        genes (List[float]): The solution parameters [x, y]
        fitness (float): Fitness value (higher is better)
        bounds (Tuple[float, float]): Valid range for gene values
    """
    
    def __init__(self, genes: List[float] = None, bounds: Tuple[float, float] = (-10.0, 10.0)):
        """
        Initialize an individual
        
        Args:
            genes: Initial gene values, random if None
            bounds: Valid range for gene values
        """
        if genes is None:
            # Initialize with random genes within bounds
            self.genes = [random.uniform(bounds[0], bounds[1]) for _ in range(2)]
        else:
            self.genes = genes.copy()
        self.fitness = None
        self.bounds = bounds
    
    def evaluate_fitness(self) -> float:
        """
        Evaluate fitness for the quadratic function: f(x, y) = (x - 3)² + (y + 1)² + 5
        
        Returns:
            float: Fitness value (negative objective for maximization)
        """
        x, y = self.genes
        objective_value = (x - 3)**2 + (y + 1)**2 + 5
        self.fitness = -objective_value  # Negative for maximization
        return self.fitness
    
    def mutate(self, mutation_rate: float) -> None:
        """
        Apply Gaussian mutation to genes
        
        Args:
            mutation_rate: Probability of mutation per gene
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                # Gaussian mutation with standard deviation of 0.5
                mutation = random.gauss(0, 0.5)
                self.genes[i] += mutation
                # Ensure bounds are respected
                self.genes[i] = max(self.bounds[0], min(self.bounds[1], self.genes[i]))
    
    def crossover(self, other: 'Individual', crossover_rate: float) -> Tuple['Individual', 'Individual']:
        """
        Perform uniform crossover between two individuals
        
        Args:
            other: The other parent individual
            crossover_rate: Probability of crossover occurring
            
        Returns:
            Tuple of two offspring individuals
        """
        if random.random() > crossover_rate:
            # No crossover, return copies of parents
            return Individual(self.genes, self.bounds), Individual(other.genes, self.bounds)
        
        # Uniform crossover
        child1_genes = []
        child2_genes = []
        
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                child1_genes.append(self.genes[i])
                child2_genes.append(other.genes[i])
            else:
                child1_genes.append(other.genes[i])
                child2_genes.append(self.genes[i])
        
        return Individual(child1_genes, self.bounds), Individual(child2_genes, self.bounds)
    
    def __str__(self) -> str:
        return f"Individual(genes={[f'{g:.4f}' for g in self.genes]}, fitness={self.fitness:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()

class TraditionalGA:
    """
    Traditional Genetic Algorithm implementation for function optimization
    
    This implementation follows standard GA practices:
    - Tournament selection
    - Uniform crossover
    - Gaussian mutation
    - Elitism preservation
    """
    
    def __init__(self, config: GAConfig):
        """
        Initialize the genetic algorithm
        
        Args:
            config: Configuration object with GA parameters
        """
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Individual = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Performance tracking
        self.start_time = None
        self.execution_time = 0
        self.function_evaluations = 0
    
    def initialize_population(self) -> None:
        """Create initial random population and evaluate fitness"""
        self.population = [
            Individual(bounds=self.config.gene_bounds) 
            for _ in range(self.config.population_size)
        ]
        
        # Evaluate initial population
        for individual in self.population:
            individual.evaluate_fitness()
            self.function_evaluations += 1
        
        self._update_best_individual()
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """
        Select individual using tournament selection
        
        Args:
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        if tournament_size is None:
            tournament_size = self.config.tournament_size
            
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _update_best_individual(self) -> None:
        """Update the best individual found so far"""
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(current_best.genes, current_best.bounds)
            self.best_individual.fitness = current_best.fitness
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity as average pairwise distance
        
        Returns:
            Average Euclidean distance between individuals
        """
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = sum((a - b)**2 for a, b in zip(
                    self.population[i].genes, self.population[j].genes
                ))**0.5
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def evolve_generation(self) -> None:
        """Evolve one generation using selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: preserve best individuals
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        for i in range(self.config.elitism_count):
            elite = Individual(sorted_population[i].genes, sorted_population[i].bounds)
            elite.fitness = sorted_population[i].fitness
            new_population.append(elite)
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = parent1.crossover(parent2, self.config.crossover_rate)
            
            # Mutation
            child1.mutate(self.config.mutation_rate)
            child2.mutate(self.config.mutation_rate)
            
            # Evaluation
            child1.evaluate_fitness()
            child2.evaluate_fitness()
            self.function_evaluations += 2
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        
        # Update tracking
        self._update_best_individual()
        self.fitness_history.append(self.best_individual.fitness)
        self.diversity_history.append(self._calculate_diversity())
        self.generation += 1
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the genetic algorithm
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing results and performance metrics
        """
        self.start_time = time.time()
        
        # Initialize
        self.initialize_population()
        
        if verbose:
            print(f"Generation 0: Best fitness = {self.best_individual.fitness:.6f}")
            print(f"Best solution: x = {self.best_individual.genes[0]:.4f}, y = {self.best_individual.genes[1]:.4f}")
        
        # Evolution loop
        for gen in range(self.config.max_generations):
            self.evolve_generation()
            
            if verbose and (gen + 1) % 20 == 0:
                print(f"Generation {gen + 1}: Best fitness = {self.best_individual.fitness:.6f}")
                print(f"Best solution: x = {self.best_individual.genes[0]:.4f}, y = {self.best_individual.genes[1]:.4f}")
        
        self.execution_time = time.time() - self.start_time
        
        # Calculate final objective value (convert back from fitness)
        final_objective = -self.best_individual.fitness
        
        results = {
            'best_solution': self.best_individual.genes.copy(),
            'best_fitness': self.best_individual.fitness,
            'best_objective_value': final_objective,
            'generations': self.generation,
            'function_evaluations': self.function_evaluations,
            'execution_time': self.execution_time,
            'fitness_history': self.fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
            'convergence_rate': self._calculate_convergence_rate(),
            'solution_quality': self._assess_solution_quality()
        }
        
        if verbose:
            print(f"\n=== Final Results ===")
            print(f"Best solution: x = {results['best_solution'][0]:.6f}, y = {results['best_solution'][1]:.6f}")
            print(f"Objective value: {results['best_objective_value']:.6f}")
            print(f"Expected minimum: 5.0 (at x=3, y=-1)")
            print(f"Error: {abs(results['best_objective_value'] - 5.0):.6f}")
            print(f"Execution time: {results['execution_time']:.3f} seconds")
            print(f"Function evaluations: {results['function_evaluations']}")
        
        return results
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on fitness improvement"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        initial_fitness = self.fitness_history[0]
        final_fitness = self.fitness_history[-1]
        improvement = final_fitness - initial_fitness
        
        return improvement / len(self.fitness_history)
    
    def _assess_solution_quality(self) -> str:
        """Assess the quality of the final solution"""
        error = abs(-self.best_individual.fitness - 5.0)  # Distance from optimal value
        
        if error < 0.01:
            return "Excellent"
        elif error < 0.1:
            return "Good"
        elif error < 1.0:
            return "Fair"
        else:
            return "Poor"
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                json_results[key] = list(value)
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

def plot_results(results: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot convergence and diversity evolution
    
    Args:
        results: Results dictionary from GA run
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fitness convergence
    ax1.plot(results['fitness_history'], 'b-', linewidth=2, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Fitness Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Population diversity
    ax2.plot(results['diversity_history'], 'r-', linewidth=2, label='Population Diversity')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Diversity')
    ax2.set_title('Population Diversity Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def run_parameter_study() -> None:
    """Run a parameter sensitivity study"""
    print("=== Parameter Sensitivity Study ===")
    
    mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    results_summary = []
    
    for mut_rate in mutation_rates:
        print(f"\nTesting mutation rate: {mut_rate}")
        
        config = GAConfig(
            population_size=50,
            mutation_rate=mut_rate,
            crossover_rate=0.8,
            max_generations=100,
            gene_bounds=(-10.0, 10.0),
            elitism_count=2
        )
        
        ga = TraditionalGA(config)
        results = ga.run(verbose=False)
        
        results_summary.append({
            'mutation_rate': mut_rate,
            'final_error': abs(results['best_objective_value'] - 5.0),
            'execution_time': results['execution_time'],
            'function_evaluations': results['function_evaluations'],
            'solution_quality': results['solution_quality']
        })
        
        print(f"Final error: {results_summary[-1]['final_error']:.6f}")
        print(f"Quality: {results_summary[-1]['solution_quality']}")
    
    # Print summary
    print(f"\n=== Parameter Study Summary ===")
    print(f"{'Mutation Rate':<15} {'Final Error':<12} {'Quality':<10} {'Time (s)':<10}")
    print("-" * 50)
    for result in results_summary:
        print(f"{result['mutation_rate']:<15} {result['final_error']:<12.6f} {result['solution_quality']:<10} {result['execution_time']:<10.3f}")

if __name__ == "__main__":
    # Example usage and testing
    print("=== Traditional Genetic Algorithm Demo ===")
    
    # Configuration
    config = GAConfig(
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
        gene_bounds=(-10.0, 10.0),
        elitism_count=2
    )
    
    # Run algorithm
    ga = TraditionalGA(config)
    results = ga.run(verbose=True)
    
    # Visualize results
    plot_results(results, save_path="traditional_ga_results.png")
    
    # Save results
    ga.save_results(results, "traditional_ga_results.json")
    
    print(f"\nSolution Quality: {results['solution_quality']}")
    print(f"Convergence Rate: {results['convergence_rate']:.6f}")
    
    # Run parameter study
    run_parameter_study()
