"""
Routing Strategies for PHMGA Tutorial Part 2

This module implements various routing strategies for the Router Component,
including round-robin, weighted, priority-based, and least-connections routing.

Author: PHMGA Tutorial Series
License: MIT
"""

import random
import heapq
from typing import Optional, List
import logging

from .base_router import BaseRouter, Worker, Request, RoutingStrategy

logger = logging.getLogger(__name__)

class RoundRobinRouter(BaseRouter):
    """
    Round-robin routing strategy
    
    Distributes requests evenly across all available workers in a circular fashion.
    Simple and fair, but doesn't consider worker capacity or performance.
    """
    
    def __init__(self, name: str = "RoundRobinRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.ROUND_ROBIN
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select worker using round-robin strategy
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return None
        
        with self._lock:
            # Get worker at current index
            worker = available_workers[self._round_robin_index % len(available_workers)]
            self._round_robin_index = (self._round_robin_index + 1) % len(available_workers)
            
            logger.debug(f"Round-robin selected worker {worker.id} for request {request.id}")
            return worker

class WeightedRouter(BaseRouter):
    """
    Weighted routing strategy
    
    Distributes requests based on worker weights. Workers with higher weights
    receive proportionally more requests. Useful when workers have different
    capacities or performance characteristics.
    """
    
    def __init__(self, name: str = "WeightedRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.WEIGHTED
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select worker using weighted random selection
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return None
        
        # Calculate total weight
        total_weight = sum(worker.weight for worker in available_workers)
        if total_weight <= 0:
            # Fallback to round-robin if no valid weights
            return available_workers[self._round_robin_index % len(available_workers)]
        
        # Weighted random selection
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for worker in available_workers:
            cumulative_weight += worker.weight
            if random_value <= cumulative_weight:
                logger.debug(f"Weighted selection chose worker {worker.id} (weight={worker.weight}) for request {request.id}")
                return worker
        
        # Fallback (shouldn't reach here)
        return available_workers[-1]

class PriorityRouter(BaseRouter):
    """
    Priority-based routing strategy
    
    Routes high-priority requests to the best-performing workers first.
    Considers both request priority and worker performance metrics.
    """
    
    def __init__(self, name: str = "PriorityRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.PRIORITY
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select worker based on request priority and worker performance
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return None
        
        # Sort workers by performance score (higher is better)
        def worker_score(worker: Worker) -> float:
            # Combine success rate, response time, and load factor
            success_score = worker.success_rate
            speed_score = 1.0 / (1.0 + worker.avg_response_time)  # Faster = higher score
            load_score = 1.0 - worker.load_factor  # Less loaded = higher score
            
            # Weight the scores
            return 0.4 * success_score + 0.3 * speed_score + 0.3 * load_score
        
        # For high-priority requests, prefer best workers
        if request.priority > 5:
            best_worker = max(available_workers, key=worker_score)
            logger.debug(f"Priority routing selected best worker {best_worker.id} for high-priority request {request.id}")
            return best_worker
        else:
            # For normal priority, use weighted selection based on performance
            workers_with_scores = [(worker, worker_score(worker)) for worker in available_workers]
            total_score = sum(score for _, score in workers_with_scores)
            
            if total_score <= 0:
                return available_workers[0]
            
            random_value = random.uniform(0, total_score)
            cumulative_score = 0
            
            for worker, score in workers_with_scores:
                cumulative_score += score
                if random_value <= cumulative_score:
                    logger.debug(f"Priority routing selected worker {worker.id} (score={score:.3f}) for request {request.id}")
                    return worker
            
            return workers_with_scores[-1][0]

class LeastConnectionsRouter(BaseRouter):
    """
    Least connections routing strategy
    
    Routes requests to the worker with the fewest active connections.
    Good for balancing load when request processing times vary significantly.
    """
    
    def __init__(self, name: str = "LeastConnectionsRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.LEAST_CONNECTIONS
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select worker with least active connections
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return None
        
        # Find worker with minimum connections
        min_connections = min(worker.current_connections for worker in available_workers)
        candidates = [worker for worker in available_workers 
                     if worker.current_connections == min_connections]
        
        # If multiple workers have same connection count, use weighted selection
        if len(candidates) > 1:
            total_weight = sum(worker.weight for worker in candidates)
            if total_weight > 0:
                random_value = random.uniform(0, total_weight)
                cumulative_weight = 0
                
                for worker in candidates:
                    cumulative_weight += worker.weight
                    if random_value <= cumulative_weight:
                        logger.debug(f"Least connections selected worker {worker.id} ({worker.current_connections} connections) for request {request.id}")
                        return worker
        
        selected_worker = candidates[0]
        logger.debug(f"Least connections selected worker {selected_worker.id} ({selected_worker.current_connections} connections) for request {request.id}")
        return selected_worker

class RandomRouter(BaseRouter):
    """
    Random routing strategy
    
    Randomly selects from available workers. Simple but may not provide
    optimal load distribution. Useful for testing or when all workers
    are equivalent.
    """
    
    def __init__(self, name: str = "RandomRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.RANDOM
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Randomly select an available worker
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return None
        
        selected_worker = random.choice(available_workers)
        logger.debug(f"Random selection chose worker {selected_worker.id} for request {request.id}")
        return selected_worker

class AdaptiveRouter(BaseRouter):
    """
    Adaptive routing strategy
    
    Dynamically switches between routing strategies based on current
    system conditions and performance metrics. Provides intelligent
    load balancing that adapts to changing conditions.
    """
    
    def __init__(self, name: str = "AdaptiveRouter"):
        super().__init__(name)
        self.strategy = RoutingStrategy.WEIGHTED  # Default strategy
        self._strategies = {
            RoutingStrategy.ROUND_ROBIN: RoundRobinRouter(),
            RoutingStrategy.WEIGHTED: WeightedRouter(),
            RoutingStrategy.PRIORITY: PriorityRouter(),
            RoutingStrategy.LEAST_CONNECTIONS: LeastConnectionsRouter(),
            RoutingStrategy.RANDOM: RandomRouter()
        }
        self._performance_window = []
        self._window_size = 100
        self._adaptation_threshold = 0.1
    
    def add_worker(self, worker: Worker) -> None:
        """Add worker to all internal strategies"""
        super().add_worker(worker)
        for strategy_router in self._strategies.values():
            strategy_router.add_worker(worker)
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from all internal strategies"""
        result = super().remove_worker(worker_id)
        for strategy_router in self._strategies.values():
            strategy_router.remove_worker(worker_id)
        return result
    
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select worker using adaptive strategy selection
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker or None if no workers available
        """
        # Analyze current system state
        current_strategy = self._select_optimal_strategy()
        
        # Use the selected strategy
        strategy_router = self._strategies[current_strategy]
        strategy_router.workers = self.workers  # Sync worker state
        
        selected_worker = strategy_router.select_worker(request)
        
        if selected_worker:
            logger.debug(f"Adaptive router using {current_strategy.value} strategy, selected worker {selected_worker.id} for request {request.id}")
        
        return selected_worker
    
    def _select_optimal_strategy(self) -> RoutingStrategy:
        """
        Select the optimal routing strategy based on current conditions
        
        Returns:
            Optimal RoutingStrategy for current conditions
        """
        available_workers = self.get_available_workers()
        if not available_workers:
            return self.strategy
        
        # Analyze system conditions
        total_load = sum(worker.load_factor for worker in available_workers) / len(available_workers)
        load_variance = self._calculate_load_variance(available_workers)
        avg_response_time = sum(worker.avg_response_time for worker in available_workers) / len(available_workers)
        
        # Decision logic based on system state
        if load_variance > 0.3:  # High load imbalance
            return RoutingStrategy.LEAST_CONNECTIONS
        elif total_load > 0.8:  # High overall load
            return RoutingStrategy.PRIORITY
        elif avg_response_time > 1.0:  # Slow response times
            return RoutingStrategy.WEIGHTED
        else:  # Normal conditions
            return RoutingStrategy.ROUND_ROBIN
    
    def _calculate_load_variance(self, workers: List[Worker]) -> float:
        """Calculate variance in worker load factors"""
        if len(workers) < 2:
            return 0.0
        
        load_factors = [worker.load_factor for worker in workers]
        mean_load = sum(load_factors) / len(load_factors)
        variance = sum((load - mean_load) ** 2 for load in load_factors) / len(load_factors)
        
        return variance

def create_router(strategy: RoutingStrategy, name: str = None) -> BaseRouter:
    """
    Factory function to create router instances
    
    Args:
        strategy: Routing strategy to use
        name: Optional name for the router
        
    Returns:
        Router instance implementing the specified strategy
    """
    if name is None:
        name = f"{strategy.value.title()}Router"
    
    router_classes = {
        RoutingStrategy.ROUND_ROBIN: RoundRobinRouter,
        RoutingStrategy.WEIGHTED: WeightedRouter,
        RoutingStrategy.PRIORITY: PriorityRouter,
        RoutingStrategy.LEAST_CONNECTIONS: LeastConnectionsRouter,
        RoutingStrategy.RANDOM: RandomRouter
    }
    
    router_class = router_classes.get(strategy)
    if router_class:
        return router_class(name)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")

if __name__ == "__main__":
    # Example usage and testing
    print("=== Routing Strategies Demo ===")
    
    from .base_router import Worker, Request
    
    # Create workers with different characteristics
    workers = [
        Worker(id="fast_worker", name="Fast Worker", weight=2.0, max_concurrent=10),
        Worker(id="slow_worker", name="Slow Worker", weight=1.0, max_concurrent=5),
        Worker(id="heavy_worker", name="Heavy Worker", weight=3.0, max_concurrent=15)
    ]
    
    # Set different performance characteristics
    workers[0].avg_response_time = 0.1  # Fast
    workers[1].avg_response_time = 0.5  # Slow
    workers[2].avg_response_time = 0.2  # Medium
    
    # Test different routing strategies
    strategies = [
        RoutingStrategy.ROUND_ROBIN,
        RoutingStrategy.WEIGHTED,
        RoutingStrategy.PRIORITY,
        RoutingStrategy.LEAST_CONNECTIONS,
        RoutingStrategy.RANDOM
    ]
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} Strategy ---")
        router = create_router(strategy)
        
        # Add workers
        for worker in workers:
            router.add_worker(worker)
        
        # Create test requests
        requests = [
            Request(payload={"task": f"task_{i}"}, priority=i % 3)
            for i in range(5)
        ]
        
        # Process requests and show routing decisions
        for request in requests:
            selected_worker = router.select_worker(request)
            if selected_worker:
                print(f"  Request {request.id[:8]} -> Worker {selected_worker.name}")
            else:
                print(f"  Request {request.id[:8]} -> No available worker")
        
        router.shutdown()
    
    print("\n=== Routing Strategies Demo Complete ===")
