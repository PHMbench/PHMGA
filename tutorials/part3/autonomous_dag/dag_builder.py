"""
Autonomous DAG Builder for PHMGA Tutorial Part 3

This module implements a self-optimizing signal processing DAG that automatically
discovers optimal processing sequences using genetic algorithms.

Author: PHMGA Tutorial Series
License: MIT
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import networkx as nx
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OperatorType(Enum):
    """Types of signal processing operators"""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"

@dataclass
class OperatorNode:
    """Represents a signal processing operator in the DAG"""
    id: str
    name: str
    operator_type: OperatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_requirements: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    computational_cost: float = 1.0
    accuracy_contribution: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.name}_{random.randint(1000, 9999)}"

class SignalProcessingOperator(ABC):
    """Abstract base class for signal processing operators"""
    
    @abstractmethod
    def process(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get expected output shape given input shape"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate operator parameters"""
        pass

class FilterOperator(SignalProcessingOperator):
    """Digital filtering operator"""
    
    def process(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        from scipy import signal
        
        filter_type = kwargs.get('filter_type', 'lowpass')
        cutoff = kwargs.get('cutoff', 0.5)
        order = kwargs.get('order', 4)
        
        if filter_type == 'lowpass':
            b, a = signal.butter(order, cutoff, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(order, cutoff, btype='high')
        elif filter_type == 'bandpass':
            low = kwargs.get('low_cutoff', 0.1)
            high = kwargs.get('high_cutoff', 0.9)
            b, a = signal.butter(order, [low, high], btype='band')
        else:
            return input_data
        
        return signal.filtfilt(b, a, input_data, axis=-1)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        required = ['filter_type', 'cutoff', 'order']
        return all(param in parameters for param in required)

class FFTOperator(SignalProcessingOperator):
    """Fast Fourier Transform operator"""
    
    def process(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        window = kwargs.get('window', 'hann')
        nperseg = kwargs.get('nperseg', 256)
        
        if window != 'none':
            from scipy import signal
            _, _, Zxx = signal.stft(input_data, nperseg=nperseg, window=window)
            return np.abs(Zxx)
        else:
            return np.abs(np.fft.fft(input_data, axis=-1))
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # Simplified - actual shape depends on parameters
        return input_shape[:-1] + (input_shape[-1] // 2,)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return 'nperseg' in parameters

class FeatureExtractionOperator(SignalProcessingOperator):
    """Statistical feature extraction operator"""
    
    def process(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        features = []
        
        # Time domain features
        features.extend([
            np.mean(input_data, axis=-1),
            np.std(input_data, axis=-1),
            np.max(input_data, axis=-1),
            np.min(input_data, axis=-1)
        ])
        
        # Additional features based on parameters
        if kwargs.get('include_rms', True):
            features.append(np.sqrt(np.mean(input_data**2, axis=-1)))
        
        if kwargs.get('include_crest_factor', True):
            rms = np.sqrt(np.mean(input_data**2, axis=-1))
            peak = np.max(np.abs(input_data), axis=-1)
            features.append(peak / (rms + 1e-10))
        
        return np.stack(features, axis=-1)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # Returns feature vector
        return input_shape[:-1] + (6,)  # 6 features by default
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return True  # No required parameters

class DAGChromosome:
    """Represents a DAG as a chromosome for genetic algorithm"""
    
    def __init__(self, operators: List[OperatorNode] = None, connections: List[Tuple[str, str]] = None):
        """
        Initialize DAG chromosome
        
        Args:
            operators: List of operator nodes
            connections: List of (source_id, target_id) connections
        """
        self.operators = operators or []
        self.connections = connections or []
        self.fitness = 0.0
        self.performance_metrics = {}
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from operators and connections"""
        self.graph = nx.DiGraph()
        
        # Add nodes
        for op in self.operators:
            self.graph.add_node(op.id, operator=op)
        
        # Add edges
        for source, target in self.connections:
            if source in [op.id for op in self.operators] and target in [op.id for op in self.operators]:
                self.graph.add_edge(source, target)
    
    def is_valid(self) -> bool:
        """Check if DAG is valid (acyclic and connected)"""
        if not self.graph:
            return False
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            return False
        
        # Check connectivity
        if len(self.operators) > 1:
            undirected = self.graph.to_undirected()
            return nx.is_connected(undirected)
        
        return True
    
    def get_execution_order(self) -> List[str]:
        """Get topological order for execution"""
        if not self.is_valid():
            return []
        
        return list(nx.topological_sort(self.graph))
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the DAG chromosome"""
        if random.random() < mutation_rate:
            mutation_type = random.choice(['add_operator', 'remove_operator', 'modify_connection', 'modify_parameters'])
            
            if mutation_type == 'add_operator':
                self._add_random_operator()
            elif mutation_type == 'remove_operator' and len(self.operators) > 1:
                self._remove_random_operator()
            elif mutation_type == 'modify_connection':
                self._modify_random_connection()
            elif mutation_type == 'modify_parameters':
                self._modify_random_parameters()
            
            self._build_graph()
    
    def _add_random_operator(self):
        """Add a random operator to the DAG"""
        operator_types = [
            ('filter', OperatorType.PREPROCESSING),
            ('fft', OperatorType.TRANSFORMATION),
            ('features', OperatorType.FEATURE_EXTRACTION)
        ]
        
        op_name, op_type = random.choice(operator_types)
        new_op = OperatorNode(
            id=f"{op_name}_{len(self.operators)}",
            name=op_name,
            operator_type=op_type,
            parameters=self._generate_random_parameters(op_name)
        )
        
        self.operators.append(new_op)
        
        # Add random connections
        if len(self.operators) > 1:
            # Connect to random existing operator
            source = random.choice(self.operators[:-1])
            self.connections.append((source.id, new_op.id))
    
    def _remove_random_operator(self):
        """Remove a random operator from the DAG"""
        if len(self.operators) <= 1:
            return
        
        to_remove = random.choice(self.operators)
        self.operators = [op for op in self.operators if op.id != to_remove.id]
        self.connections = [(s, t) for s, t in self.connections 
                          if s != to_remove.id and t != to_remove.id]
    
    def _modify_random_connection(self):
        """Modify a random connection"""
        if len(self.operators) < 2:
            return
        
        if self.connections and random.random() < 0.5:
            # Remove existing connection
            self.connections.pop(random.randint(0, len(self.connections) - 1))
        else:
            # Add new connection
            source = random.choice(self.operators)
            target = random.choice([op for op in self.operators if op.id != source.id])
            new_connection = (source.id, target.id)
            if new_connection not in self.connections:
                self.connections.append(new_connection)
    
    def _modify_random_parameters(self):
        """Modify parameters of a random operator"""
        if not self.operators:
            return
        
        operator = random.choice(self.operators)
        if operator.name == 'filter':
            operator.parameters['cutoff'] = random.uniform(0.1, 0.9)
            operator.parameters['order'] = random.randint(2, 8)
        elif operator.name == 'fft':
            operator.parameters['nperseg'] = random.choice([128, 256, 512, 1024])
    
    def _generate_random_parameters(self, operator_name: str) -> Dict[str, Any]:
        """Generate random parameters for an operator"""
        if operator_name == 'filter':
            return {
                'filter_type': random.choice(['lowpass', 'highpass', 'bandpass']),
                'cutoff': random.uniform(0.1, 0.9),
                'order': random.randint(2, 8)
            }
        elif operator_name == 'fft':
            return {
                'window': random.choice(['hann', 'hamming', 'blackman']),
                'nperseg': random.choice([128, 256, 512])
            }
        elif operator_name == 'features':
            return {
                'include_rms': random.choice([True, False]),
                'include_crest_factor': random.choice([True, False])
            }
        else:
            return {}
    
    def crossover(self, other: 'DAGChromosome') -> Tuple['DAGChromosome', 'DAGChromosome']:
        """Crossover with another DAG chromosome"""
        # Simple crossover: exchange random operators
        child1_ops = self.operators.copy()
        child2_ops = other.operators.copy()
        
        if len(child1_ops) > 1 and len(child2_ops) > 1:
            # Exchange random operators
            idx1 = random.randint(0, len(child1_ops) - 1)
            idx2 = random.randint(0, len(child2_ops) - 1)
            
            child1_ops[idx1], child2_ops[idx2] = child2_ops[idx2], child1_ops[idx1]
        
        # Rebuild connections (simplified)
        child1_connections = self._rebuild_connections(child1_ops)
        child2_connections = self._rebuild_connections(child2_ops)
        
        child1 = DAGChromosome(child1_ops, child1_connections)
        child2 = DAGChromosome(child2_ops, child2_connections)
        
        return child1, child2
    
    def _rebuild_connections(self, operators: List[OperatorNode]) -> List[Tuple[str, str]]:
        """Rebuild connections for given operators"""
        connections = []
        
        for i in range(len(operators) - 1):
            connections.append((operators[i].id, operators[i + 1].id))
        
        return connections

class AutonomousDAGBuilder:
    """
    Autonomous DAG builder using genetic algorithms
    
    Automatically discovers optimal signal processing pipelines
    through evolutionary optimization.
    """
    
    def __init__(self, population_size: int = 50, max_generations: int = 100):
        """
        Initialize DAG builder
        
        Args:
            population_size: Size of the genetic algorithm population
            max_generations: Maximum number of generations to evolve
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.population: List[DAGChromosome] = []
        self.best_dag: Optional[DAGChromosome] = None
        self.evolution_history = []
        
        # Available operators
        self.operator_registry = {
            'filter': FilterOperator(),
            'fft': FFTOperator(),
            'features': FeatureExtractionOperator()
        }
        
        logger.info(f"Autonomous DAG Builder initialized with population size {population_size}")
    
    def initialize_population(self) -> None:
        """Initialize random population of DAGs"""
        self.population = []
        
        for _ in range(self.population_size):
            # Create random DAG
            num_operators = random.randint(2, 5)
            operators = []
            
            for i in range(num_operators):
                op_name = random.choice(list(self.operator_registry.keys()))
                op_type = {
                    'filter': OperatorType.PREPROCESSING,
                    'fft': OperatorType.TRANSFORMATION,
                    'features': OperatorType.FEATURE_EXTRACTION
                }[op_name]
                
                operator = OperatorNode(
                    id=f"{op_name}_{i}",
                    name=op_name,
                    operator_type=op_type,
                    parameters=self._generate_random_parameters(op_name)
                )
                operators.append(operator)
            
            # Create linear connections (can be optimized later)
            connections = []
            for i in range(len(operators) - 1):
                connections.append((operators[i].id, operators[i + 1].id))
            
            dag = DAGChromosome(operators, connections)
            if dag.is_valid():
                self.population.append(dag)
        
        # Fill population if needed
        while len(self.population) < self.population_size:
            self.population.append(self._create_simple_dag())
        
        logger.info(f"Initialized population with {len(self.population)} valid DAGs")
    
    def _create_simple_dag(self) -> DAGChromosome:
        """Create a simple valid DAG"""
        operators = [
            OperatorNode("filter_0", "filter", OperatorType.PREPROCESSING, 
                        {'filter_type': 'lowpass', 'cutoff': 0.5, 'order': 4}),
            OperatorNode("features_1", "features", OperatorType.FEATURE_EXTRACTION, 
                        {'include_rms': True, 'include_crest_factor': True})
        ]
        connections = [("filter_0", "features_1")]
        return DAGChromosome(operators, connections)
    
    def _generate_random_parameters(self, operator_name: str) -> Dict[str, Any]:
        """Generate random parameters for an operator"""
        if operator_name == 'filter':
            return {
                'filter_type': random.choice(['lowpass', 'highpass']),
                'cutoff': random.uniform(0.1, 0.9),
                'order': random.randint(2, 8)
            }
        elif operator_name == 'fft':
            return {
                'window': random.choice(['hann', 'hamming']),
                'nperseg': random.choice([128, 256, 512])
            }
        elif operator_name == 'features':
            return {
                'include_rms': True,
                'include_crest_factor': True
            }
        return {}
    
    def evaluate_fitness(self, dag: DAGChromosome, test_signals: List[np.ndarray]) -> float:
        """
        Evaluate fitness of a DAG on test signals
        
        Args:
            dag: DAG chromosome to evaluate
            test_signals: List of test signals
            
        Returns:
            Fitness score (higher is better)
        """
        if not dag.is_valid():
            return 0.0
        
        try:
            total_score = 0.0
            execution_order = dag.get_execution_order()
            
            for signal in test_signals:
                # Execute DAG on signal
                current_data = signal.copy()
                processing_time = 0.0
                
                start_time = time.time()
                
                for op_id in execution_order:
                    operator_node = next(op for op in dag.operators if op.id == op_id)
                    operator = self.operator_registry[operator_node.name]
                    
                    try:
                        current_data = operator.process(current_data, **operator_node.parameters)
                    except Exception as e:
                        logger.warning(f"Operator {op_id} failed: {e}")
                        return 0.0
                
                processing_time = time.time() - start_time
                
                # Calculate fitness based on output quality and efficiency
                output_quality = self._evaluate_output_quality(current_data)
                efficiency = 1.0 / (1.0 + processing_time)  # Prefer faster processing
                
                signal_score = 0.7 * output_quality + 0.3 * efficiency
                total_score += signal_score
            
            fitness = total_score / len(test_signals)
            dag.fitness = fitness
            
            return fitness
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def _evaluate_output_quality(self, output_data: np.ndarray) -> float:
        """Evaluate quality of processing output"""
        # Simplified quality metric
        # In practice, this would use domain-specific criteria
        
        if output_data.size == 0:
            return 0.0
        
        # Check for reasonable output range
        if np.any(np.isnan(output_data)) or np.any(np.isinf(output_data)):
            return 0.0
        
        # Prefer outputs with good dynamic range
        dynamic_range = np.max(output_data) - np.min(output_data)
        normalized_range = min(dynamic_range / (np.std(output_data) + 1e-10), 10.0) / 10.0
        
        # Prefer outputs with reasonable variance
        variance_score = min(np.var(output_data), 1.0)
        
        return 0.6 * normalized_range + 0.4 * variance_score
    
    def evolve(self, test_signals: List[np.ndarray]) -> DAGChromosome:
        """
        Evolve optimal DAG using genetic algorithm
        
        Args:
            test_signals: Test signals for fitness evaluation
            
        Returns:
            Best evolved DAG
        """
        logger.info("Starting DAG evolution...")
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            for dag in self.population:
                if dag.fitness == 0.0:  # Only evaluate if not already evaluated
                    dag.fitness = self.evaluate_fitness(dag, test_signals)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best DAG
            if not self.best_dag or self.population[0].fitness > self.best_dag.fitness:
                self.best_dag = self.population[0]
            
            # Log progress
            avg_fitness = np.mean([dag.fitness for dag in self.population])
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': avg_fitness
            })
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.population[0].fitness:.4f}, "
                          f"Avg fitness = {avg_fitness:.4f}")
            
            # Create next generation
            if generation < self.max_generations - 1:
                self._create_next_generation()
        
        logger.info(f"Evolution complete. Best fitness: {self.best_dag.fitness:.4f}")
        return self.best_dag
    
    def _create_next_generation(self):
        """Create next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, self.population_size // 10)
        new_population.extend(self.population[:elite_count])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < 0.8:  # Crossover probability
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1.mutate(0.1)
            child2.mutate(0.1)
            
            # Add valid children
            if child1.is_valid():
                new_population.append(child1)
            if len(new_population) < self.population_size and child2.is_valid():
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> DAGChromosome:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

if __name__ == "__main__":
    # Example usage
    print("=== Autonomous DAG Builder Demo ===")
    
    # Create test signals
    test_signals = []
    for i in range(5):
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
        test_signals.append(signal)
    
    # Create DAG builder
    builder = AutonomousDAGBuilder(population_size=20, max_generations=50)
    
    # Evolve optimal DAG
    best_dag = builder.evolve(test_signals)
    
    print(f"\nBest DAG found:")
    print(f"Fitness: {best_dag.fitness:.4f}")
    print(f"Operators: {[op.name for op in best_dag.operators]}")
    print(f"Execution order: {best_dag.get_execution_order()}")
    
    print("\nDemo complete!")
