"""
LLM-Enhanced Genetic Algorithm Implementation for PHMGA Tutorial Part 1

This module demonstrates an LLM-enhanced genetic algorithm that uses natural language
interfaces for parameter tuning, fitness function descriptions, and automated code generation.

Author: PHMGA Tutorial Series
License: MIT
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import json
import os
from abc import ABC, abstractmethod

# Import the traditional GA as base
from traditional_ga import Individual, GAConfig, plot_results

# LLM Integration (optional - graceful fallback if not available)
try:
    import openai
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: OpenAI not available. LLM features will be disabled.")

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response from LLM"""
        pass

class OpenAIInterface(LLMInterface):
    """OpenAI GPT interface"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        if not LLM_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"

class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing when API is not available"""
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate mock responses based on prompt keywords"""
        prompt_lower = prompt.lower()
        
        if "mutation rate" in prompt_lower:
            if "increase" in prompt_lower or "higher" in prompt_lower:
                return "Increase mutation rate to 0.15 for better exploration"
            elif "decrease" in prompt_lower or "lower" in prompt_lower:
                return "Decrease mutation rate to 0.05 for better exploitation"
            else:
                return "Current mutation rate seems appropriate for this problem"
        
        elif "population size" in prompt_lower:
            return "Consider increasing population size to 75 for better diversity"
        
        elif "crossover" in prompt_lower:
            return "Uniform crossover with rate 0.8 is suitable for this continuous optimization"
        
        elif "fitness function" in prompt_lower:
            return "The quadratic fitness function has a single global minimum, making it ideal for testing convergence"
        
        elif "performance" in prompt_lower:
            return "Algorithm performance can be improved by balancing exploration and exploitation"
        
        else:
            return "The genetic algorithm parameters appear to be well-configured for this optimization problem"

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: str = "openai"  # "openai" or "mock"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    enable_parameter_tuning: bool = True
    enable_fitness_analysis: bool = True
    enable_code_generation: bool = False  # Advanced feature

class LLMEnhancedGA:
    """
    LLM-Enhanced Genetic Algorithm with natural language interfaces
    
    Features:
    - Natural language parameter tuning suggestions
    - Fitness function analysis and insights
    - Automated performance optimization recommendations
    - Code generation for custom operators (advanced)
    """
    
    def __init__(self, ga_config: GAConfig, llm_config: LLMConfig):
        """
        Initialize LLM-enhanced genetic algorithm
        
        Args:
            ga_config: Traditional GA configuration
            llm_config: LLM integration configuration
        """
        self.ga_config = ga_config
        self.llm_config = llm_config
        
        # Initialize LLM interface
        if llm_config.provider == "openai" and LLM_AVAILABLE:
            try:
                self.llm = OpenAIInterface(llm_config.api_key, llm_config.model)
            except Exception:
                print("Warning: OpenAI initialization failed. Using mock interface.")
                self.llm = MockLLMInterface()
        else:
            self.llm = MockLLMInterface()
        
        # GA components
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Individual = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # LLM interaction history
        self.llm_interactions: List[Dict[str, str]] = []
        self.parameter_adjustments: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = None
        self.execution_time = 0
        self.function_evaluations = 0
        self.llm_calls = 0
    
    def _query_llm(self, prompt: str, context: str = "") -> str:
        """
        Query LLM with context and track interactions
        
        Args:
            prompt: The question or request
            context: Additional context information
            
        Returns:
            LLM response
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = self.llm.generate_response(full_prompt)
        
        # Track interaction
        self.llm_interactions.append({
            'prompt': prompt,
            'context': context,
            'response': response,
            'generation': self.generation
        })
        self.llm_calls += 1
        
        return response
    
    def analyze_fitness_function(self) -> str:
        """Get LLM analysis of the fitness function"""
        prompt = """
        Analyze this fitness function for genetic algorithm optimization:
        
        Function: f(x, y) = (x - 3)² + (y + 1)² + 5
        Domain: x, y ∈ [-10, 10]
        Objective: Minimize the function
        
        Provide insights about:
        1. Function characteristics (convexity, global minimum, etc.)
        2. Expected difficulty for genetic algorithms
        3. Recommended GA parameters
        """
        
        return self._query_llm(prompt)
    
    def get_parameter_tuning_suggestions(self, current_performance: Dict[str, Any]) -> str:
        """
        Get LLM suggestions for parameter tuning based on current performance
        
        Args:
            current_performance: Dictionary with current GA performance metrics
            
        Returns:
            LLM suggestions for parameter improvements
        """
        context = f"""
        Current GA Performance:
        - Generation: {current_performance.get('generation', 0)}
        - Best fitness: {current_performance.get('best_fitness', 0):.6f}
        - Population diversity: {current_performance.get('diversity', 0):.6f}
        - Convergence rate: {current_performance.get('convergence_rate', 0):.6f}
        
        Current Parameters:
        - Population size: {self.ga_config.population_size}
        - Mutation rate: {self.ga_config.mutation_rate}
        - Crossover rate: {self.ga_config.crossover_rate}
        - Elitism count: {self.ga_config.elitism_count}
        """
        
        prompt = """
        Based on the current performance metrics, suggest parameter adjustments to improve the genetic algorithm.
        Focus on specific, actionable recommendations with reasoning.
        """
        
        return self._query_llm(prompt, context)
    
    def adaptive_parameter_tuning(self) -> None:
        """
        Adaptively tune parameters based on LLM suggestions and performance
        """
        if not self.llm_config.enable_parameter_tuning:
            return
        
        # Only tune after some generations to gather performance data
        if self.generation < 10 or self.generation % 25 != 0:
            return
        
        current_performance = {
            'generation': self.generation,
            'best_fitness': self.best_individual.fitness if self.best_individual else 0,
            'diversity': self._calculate_diversity(),
            'convergence_rate': self._calculate_convergence_rate()
        }
        
        suggestions = self.get_parameter_tuning_suggestions(current_performance)
        
        # Parse suggestions and apply reasonable adjustments
        # This is a simplified implementation - in practice, you'd use more sophisticated parsing
        adjustments = {}
        
        if "increase mutation" in suggestions.lower():
            new_rate = min(0.5, self.ga_config.mutation_rate * 1.2)
            adjustments['mutation_rate'] = new_rate
            self.ga_config.mutation_rate = new_rate
        
        elif "decrease mutation" in suggestions.lower():
            new_rate = max(0.01, self.ga_config.mutation_rate * 0.8)
            adjustments['mutation_rate'] = new_rate
            self.ga_config.mutation_rate = new_rate
        
        if "increase population" in suggestions.lower():
            new_size = min(200, int(self.ga_config.population_size * 1.1))
            adjustments['population_size'] = new_size
            # Note: Population size changes require reinitialization in practice
        
        if adjustments:
            self.parameter_adjustments.append({
                'generation': self.generation,
                'adjustments': adjustments,
                'reason': suggestions
            })
    
    def initialize_population(self) -> None:
        """Initialize population with LLM analysis"""
        if self.llm_config.enable_fitness_analysis:
            analysis = self.analyze_fitness_function()
            print(f"LLM Fitness Analysis:\n{analysis}\n")
        
        # Standard population initialization
        self.population = [
            Individual(bounds=self.ga_config.gene_bounds) 
            for _ in range(self.ga_config.population_size)
        ]
        
        # Evaluate initial population
        for individual in self.population:
            individual.evaluate_fitness()
            self.function_evaluations += 1
        
        self._update_best_individual()
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """Tournament selection with LLM-suggested improvements"""
        if tournament_size is None:
            tournament_size = self.ga_config.tournament_size
            
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _update_best_individual(self) -> None:
        """Update best individual and trigger adaptive tuning"""
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(current_best.genes, current_best.bounds)
            self.best_individual.fitness = current_best.fitness
        
        # Trigger adaptive parameter tuning
        self.adaptive_parameter_tuning()
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
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
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        initial_fitness = self.fitness_history[0]
        final_fitness = self.fitness_history[-1]
        improvement = final_fitness - initial_fitness
        
        return improvement / len(self.fitness_history)
    
    def evolve_generation(self) -> None:
        """Evolve one generation with LLM enhancements"""
        new_population = []
        
        # Elitism: preserve best individuals
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        for i in range(self.ga_config.elitism_count):
            elite = Individual(sorted_population[i].genes, sorted_population[i].bounds)
            elite.fitness = sorted_population[i].fitness
            new_population.append(elite)
        
        # Generate offspring
        while len(new_population) < self.ga_config.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = parent1.crossover(parent2, self.ga_config.crossover_rate)
            
            # Mutation
            child1.mutate(self.ga_config.mutation_rate)
            child2.mutate(self.ga_config.mutation_rate)
            
            # Evaluation
            child1.evaluate_fitness()
            child2.evaluate_fitness()
            self.function_evaluations += 2
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.ga_config.population_size]
        
        # Update tracking
        self._update_best_individual()
        self.fitness_history.append(self.best_individual.fitness)
        self.diversity_history.append(self._calculate_diversity())
        self.generation += 1
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the LLM-enhanced genetic algorithm
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing results and performance metrics
        """
        self.start_time = time.time()
        
        # Initialize with LLM analysis
        self.initialize_population()
        
        if verbose:
            print(f"Generation 0: Best fitness = {self.best_individual.fitness:.6f}")
            print(f"Best solution: x = {self.best_individual.genes[0]:.4f}, y = {self.best_individual.genes[1]:.4f}")
        
        # Evolution loop
        for gen in range(self.ga_config.max_generations):
            self.evolve_generation()
            
            if verbose and (gen + 1) % 20 == 0:
                print(f"Generation {gen + 1}: Best fitness = {self.best_individual.fitness:.6f}")
                print(f"Best solution: x = {self.best_individual.genes[0]:.4f}, y = {self.best_individual.genes[1]:.4f}")
                
                # Show parameter adjustments
                if self.parameter_adjustments:
                    latest_adjustment = self.parameter_adjustments[-1]
                    if latest_adjustment['generation'] >= gen - 5:  # Recent adjustment
                        print(f"Recent parameter adjustment: {latest_adjustment['adjustments']}")
        
        self.execution_time = time.time() - self.start_time
        
        # Get final LLM analysis
        final_analysis = ""
        if self.llm_config.enable_fitness_analysis:
            final_performance = {
                'generation': self.generation,
                'best_fitness': self.best_individual.fitness,
                'diversity': self._calculate_diversity(),
                'convergence_rate': self._calculate_convergence_rate()
            }
            final_analysis = self.get_parameter_tuning_suggestions(final_performance)
        
        # Calculate final objective value
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
            'solution_quality': self._assess_solution_quality(),
            'llm_calls': self.llm_calls,
            'llm_interactions': self.llm_interactions.copy(),
            'parameter_adjustments': self.parameter_adjustments.copy(),
            'final_analysis': final_analysis
        }
        
        if verbose:
            print(f"\n=== Final Results ===")
            print(f"Best solution: x = {results['best_solution'][0]:.6f}, y = {results['best_solution'][1]:.6f}")
            print(f"Objective value: {results['best_objective_value']:.6f}")
            print(f"Expected minimum: 5.0 (at x=3, y=-1)")
            print(f"Error: {abs(results['best_objective_value'] - 5.0):.6f}")
            print(f"Execution time: {results['execution_time']:.3f} seconds")
            print(f"Function evaluations: {results['function_evaluations']}")
            print(f"LLM calls: {results['llm_calls']}")
            
            if final_analysis:
                print(f"\nFinal LLM Analysis:\n{final_analysis}")
        
        return results
    
    def _assess_solution_quality(self) -> str:
        """Assess solution quality"""
        error = abs(-self.best_individual.fitness - 5.0)
        
        if error < 0.01:
            return "Excellent"
        elif error < 0.1:
            return "Good"
        elif error < 1.0:
            return "Fair"
        else:
            return "Poor"

def compare_implementations() -> None:
    """Compare traditional vs LLM-enhanced implementations"""
    print("=== Implementation Comparison ===")
    
    # Shared configuration
    ga_config = GAConfig(
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
        gene_bounds=(-10.0, 10.0),
        elitism_count=2
    )
    
    llm_config = LLMConfig(
        provider="mock",  # Use mock for consistent testing
        enable_parameter_tuning=True,
        enable_fitness_analysis=True
    )
    
    # Run traditional GA
    print("\n--- Traditional GA ---")
    from traditional_ga import TraditionalGA
    traditional_ga = TraditionalGA(ga_config)
    traditional_results = traditional_ga.run(verbose=False)
    
    # Run LLM-enhanced GA
    print("\n--- LLM-Enhanced GA ---")
    llm_ga = LLMEnhancedGA(ga_config, llm_config)
    llm_results = llm_ga.run(verbose=False)
    
    # Comparison
    print(f"\n=== Comparison Results ===")
    print(f"{'Metric':<25} {'Traditional':<15} {'LLM-Enhanced':<15}")
    print("-" * 55)
    print(f"{'Final Error':<25} {abs(traditional_results['best_objective_value'] - 5.0):<15.6f} {abs(llm_results['best_objective_value'] - 5.0):<15.6f}")
    print(f"{'Execution Time (s)':<25} {traditional_results['execution_time']:<15.3f} {llm_results['execution_time']:<15.3f}")
    print(f"{'Function Evaluations':<25} {traditional_results['function_evaluations']:<15} {llm_results['function_evaluations']:<15}")
    print(f"{'Solution Quality':<25} {traditional_results['solution_quality']:<15} {llm_results['solution_quality']:<15}")
    print(f"{'LLM Calls':<25} {'0':<15} {llm_results['llm_calls']:<15}")
    print(f"{'Parameter Adjustments':<25} {'0':<15} {len(llm_results['parameter_adjustments']):<15}")

if __name__ == "__main__":
    # Example usage
    print("=== LLM-Enhanced Genetic Algorithm Demo ===")

    # Configuration
    ga_config = GAConfig(
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
        gene_bounds=(-10.0, 10.0),
        elitism_count=2
    )

    llm_config = LLMConfig(
        provider="mock",  # Change to "openai" if you have API key
        enable_parameter_tuning=True,
        enable_fitness_analysis=True
    )

    # Run LLM-enhanced algorithm
    llm_ga = LLMEnhancedGA(ga_config, llm_config)
    results = llm_ga.run(verbose=True)

    # Visualize results
    plot_results(results, save_path="llm_enhanced_ga_results.png")

    # Show LLM interactions
    print(f"\n=== LLM Interaction Summary ===")
    for i, interaction in enumerate(results['llm_interactions']):
        print(f"\nInteraction {i+1} (Generation {interaction['generation']}):")
        print(f"Query: {interaction['prompt'][:100]}...")
        print(f"Response: {interaction['response'][:150]}...")

    # Run comparison
    compare_implementations()
