# Part 1: Foundation - Basic Function Implementation

**Estimated Time**: 2-3 hours  
**Prerequisites**: Basic Python knowledge, understanding of optimization concepts  
**Difficulty**: Beginner

## 🎯 Learning Objectives

By the end of this tutorial, you will:

1. Understand the fundamental concepts of genetic algorithms
2. Implement a traditional genetic algorithm in pure Python
3. Create an LLM-enhanced version with natural language interfaces
4. Compare performance characteristics between approaches
5. Recognize when to use each implementation strategy

## 📋 Overview

This tutorial introduces PHMGA concepts through a simple mathematical optimization problem: finding the optimal parameters for a quadratic function. We'll implement two approaches:

1. **Traditional Implementation**: Pure Python genetic algorithm
2. **LLM-Enhanced Implementation**: Integration with language model capabilities

## 🔧 Setup

Ensure you have completed the [general setup instructions](../README.md#setup-instructions).

Additional requirements for this tutorial:
```bash
pip install matplotlib seaborn  # For visualization
pip install openai  # For LLM integration (optional)
```

## 📚 Section 1.1: Traditional Implementation

### Problem Definition

We'll optimize the quadratic function: `f(x, y) = (x - 3)² + (y + 1)² + 5`

**Objective**: Find values of `x` and `y` that minimize this function  
**Expected Solution**: `x = 3, y = -1` with minimum value `f(3, -1) = 5`

### Core Concepts

#### Genetic Algorithm Components

1. **Individual**: A candidate solution represented as `[x, y]`
2. **Population**: A collection of individuals
3. **Fitness Function**: Evaluates how good a solution is
4. **Selection**: Choose parents for reproduction
5. **Crossover**: Combine parent genes to create offspring
6. **Mutation**: Introduce random variations
7. **Evolution**: Iterative improvement over generations

#### Implementation Strategy

```python
class Individual:
    """Represents a candidate solution"""
    def __init__(self, genes: List[float]):
        self.genes = genes
        self.fitness = None
    
    def evaluate_fitness(self):
        """Calculate fitness for the quadratic function"""
        x, y = self.genes
        self.fitness = -((x - 3)**2 + (y + 1)**2 + 5)  # Negative for maximization
        return self.fitness

class GeneticAlgorithm:
    """Traditional genetic algorithm implementation"""
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
```

### Complete Implementation

Create `tutorials/part1/traditional_ga.py`:

```python
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
import time

@dataclass
class GAConfig:
    """Configuration for genetic algorithm parameters"""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 100
    gene_bounds: Tuple[float, float] = (-10.0, 10.0)
    elitism_count: int = 2

class Individual:
    """Represents a candidate solution with genes and fitness"""
    
    def __init__(self, genes: List[float] = None, bounds: Tuple[float, float] = (-10.0, 10.0)):
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
        We use negative value for maximization-based GA
        """
        x, y = self.genes
        objective_value = (x - 3)**2 + (y + 1)**2 + 5
        self.fitness = -objective_value  # Negative for maximization
        return self.fitness
    
    def mutate(self, mutation_rate: float) -> None:
        """Apply Gaussian mutation to genes"""
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
        Returns two offspring
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
        return f"Individual(genes={self.genes}, fitness={self.fitness})"

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
        """Create initial random population"""
        self.population = [
            Individual(bounds=self.config.gene_bounds) 
            for _ in range(self.config.population_size)
        ]
        
        # Evaluate initial population
        for individual in self.population:
            individual.evaluate_fitness()
            self.function_evaluations += 1
        
        self._update_best_individual()
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _update_best_individual(self) -> None:
        """Update the best individual found so far"""
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(current_best.genes, current_best.bounds)
            self.best_individual.fitness = current_best.fitness
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity as average pairwise distance"""
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
        """Evolve one generation"""
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
    
    def run(self, verbose: bool = True) -> dict:
        """
        Run the genetic algorithm
        
        Returns:
            dict: Results including best solution, convergence data, and performance metrics
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

def plot_results(results: dict, save_path: str = None) -> None:
    """Plot convergence and diversity evolution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fitness convergence
    ax1.plot(results['fitness_history'], 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Fitness Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Population diversity
    ax2.plot(results['diversity_history'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Diversity')
    ax2.set_title('Population Diversity Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

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
    plot_results(results)
    
    print(f"\nSolution Quality: {results['solution_quality']}")
    print(f"Convergence Rate: {results['convergence_rate']:.6f}")
```

This implementation provides a solid foundation for understanding genetic algorithms. In the next section, we'll enhance it with LLM capabilities.

## 🎯 Key Takeaways

1. **Genetic algorithms** are powerful optimization techniques inspired by natural evolution
2. **Core components** include population, selection, crossover, mutation, and fitness evaluation
3. **Parameter tuning** significantly affects performance (population size, mutation rate, etc.)
4. **Performance metrics** help evaluate algorithm effectiveness

## 📝 Exercises

1. **Parameter Sensitivity**: Experiment with different mutation rates (0.01, 0.1, 0.3) and observe the impact
2. **Function Modification**: Change the target function and verify the algorithm adapts
3. **Convergence Analysis**: Plot fitness vs. generation and identify convergence patterns

**Next**: [Section 1.2 - LLM-Enhanced Implementation](1.2-llm-enhanced.md)
