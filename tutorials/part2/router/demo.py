"""
Router Component Demonstration for PHMGA Tutorial Part 2

This module provides a comprehensive demonstration of the Router Component
capabilities, including different routing strategies, performance monitoring,
and real-world usage scenarios.

Author: PHMGA Tutorial Series
License: MIT
"""

import time
import random
import threading
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from base_router import Worker, Request, RequestStatus
from strategies import (
    create_router, RoutingStrategy, RoundRobinRouter, WeightedRouter,
    PriorityRouter, LeastConnectionsRouter, AdaptiveRouter
)

class RouterDemo:
    """Comprehensive demonstration of router capabilities"""
    
    def __init__(self):
        self.results = {}
    
    def create_test_workers(self) -> List[Worker]:
        """Create a diverse set of test workers"""
        workers = [
            Worker(
                id="high_performance",
                name="High Performance Worker",
                weight=3.0,
                max_concurrent=20
            ),
            Worker(
                id="standard_worker_1",
                name="Standard Worker 1",
                weight=2.0,
                max_concurrent=10
            ),
            Worker(
                id="standard_worker_2",
                name="Standard Worker 2",
                weight=2.0,
                max_concurrent=10
            ),
            Worker(
                id="low_capacity",
                name="Low Capacity Worker",
                weight=1.0,
                max_concurrent=5
            ),
            Worker(
                id="unreliable_worker",
                name="Unreliable Worker",
                weight=1.5,
                max_concurrent=8
            )
        ]
        
        # Set realistic performance characteristics
        workers[0].avg_response_time = 0.1  # High performance
        workers[1].avg_response_time = 0.3  # Standard
        workers[2].avg_response_time = 0.3  # Standard
        workers[3].avg_response_time = 0.5  # Slower
        workers[4].avg_response_time = 0.4  # Unreliable
        
        # Simulate some failed requests for the unreliable worker
        workers[4].total_requests = 100
        workers[4].failed_requests = 15  # 15% failure rate
        
        return workers
    
    def create_test_requests(self, count: int = 50) -> List[Request]:
        """Create a set of test requests with varying characteristics"""
        requests = []
        
        for i in range(count):
            priority = random.choice([0, 1, 2, 5, 8, 10])  # Mix of priorities
            timeout = random.uniform(5.0, 30.0)
            
            request = Request(
                payload={
                    "task_type": random.choice(["optimization", "analysis", "simulation"]),
                    "complexity": random.choice(["low", "medium", "high"]),
                    "data_size": random.randint(100, 10000)
                },
                priority=priority,
                timeout=timeout
            )
            requests.append(request)
        
        return requests
    
    def benchmark_routing_strategy(self, strategy: RoutingStrategy, 
                                 workers: List[Worker], 
                                 requests: List[Request]) -> Dict[str, Any]:
        """Benchmark a specific routing strategy"""
        print(f"\n--- Benchmarking {strategy.value} Strategy ---")
        
        # Create router
        router = create_router(strategy, f"Benchmark_{strategy.value}")
        
        # Add workers
        for worker in workers:
            router.add_worker(worker)
        
        # Start health checks
        router.start_health_checks()
        
        # Process requests
        start_time = time.time()
        processed_requests = []
        
        for request in requests:
            processed_request = router.process_request(request)
            processed_requests.append(processed_request)
            
            # Small delay to simulate realistic request arrival
            time.sleep(0.01)
        
        end_time = time.time()
        
        # Collect metrics
        metrics = router.get_metrics()
        
        # Calculate additional statistics
        successful_requests = [r for r in processed_requests if r.status == RequestStatus.COMPLETED]
        failed_requests = [r for r in processed_requests if r.status == RequestStatus.FAILED]
        
        processing_times = [r.processing_time for r in successful_requests if r.processing_time]
        
        results = {
            'strategy': strategy.value,
            'total_time': end_time - start_time,
            'total_requests': len(processed_requests),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(processed_requests) if processed_requests else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'median_processing_time': np.median(processing_times) if processing_times else 0,
            'p95_processing_time': np.percentile(processing_times, 95) if processing_times else 0,
            'throughput': len(processed_requests) / (end_time - start_time),
            'router_metrics': metrics,
            'worker_distribution': self._analyze_worker_distribution(processed_requests, workers)
        }
        
        # Print summary
        print(f"  Total Requests: {results['total_requests']}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Avg Processing Time: {results['avg_processing_time']:.3f}s")
        print(f"  Throughput: {results['throughput']:.1f} req/s")
        
        router.shutdown()
        return results
    
    def _analyze_worker_distribution(self, requests: List[Request], 
                                   workers: List[Worker]) -> Dict[str, int]:
        """Analyze how requests were distributed across workers"""
        distribution = {worker.id: 0 for worker in workers}
        
        for request in requests:
            if request.worker_id and request.worker_id in distribution:
                distribution[request.worker_id] += 1
        
        return distribution
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all routing strategies"""
        print("=== Comprehensive Router Benchmark ===")
        
        # Create test data
        workers = self.create_test_workers()
        requests = self.create_test_requests(100)
        
        # Test all strategies
        strategies = [
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.WEIGHTED,
            RoutingStrategy.PRIORITY,
            RoutingStrategy.LEAST_CONNECTIONS,
            RoutingStrategy.RANDOM
        ]
        
        results = {}
        
        for strategy in strategies:
            # Create fresh worker instances for each test
            test_workers = [
                Worker(
                    id=w.id,
                    name=w.name,
                    weight=w.weight,
                    max_concurrent=w.max_concurrent
                ) for w in workers
            ]
            
            # Copy performance characteristics
            for i, worker in enumerate(test_workers):
                worker.avg_response_time = workers[i].avg_response_time
                worker.total_requests = workers[i].total_requests
                worker.failed_requests = workers[i].failed_requests
            
            # Create fresh request instances
            test_requests = [
                Request(
                    payload=r.payload.copy(),
                    priority=r.priority,
                    timeout=r.timeout
                ) for r in requests
            ]
            
            results[strategy.value] = self.benchmark_routing_strategy(
                strategy, test_workers, test_requests
            )
        
        self.results = results
        return results
    
    def visualize_results(self, save_path: str = None):
        """Create comprehensive visualizations of benchmark results"""
        if not self.results:
            print("No results to visualize. Run benchmark first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        strategies = list(self.results.keys())
        
        # 1. Success Rate Comparison
        success_rates = [self.results[s]['success_rate'] for s in strategies]
        bars1 = ax1.bar(strategies, success_rates, alpha=0.8, color='skyblue')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Routing Strategy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # 2. Average Processing Time
        avg_times = [self.results[s]['avg_processing_time'] for s in strategies]
        bars2 = ax2.bar(strategies, avg_times, alpha=0.8, color='lightcoral')
        ax2.set_ylabel('Average Processing Time (s)')
        ax2.set_title('Average Processing Time by Strategy')
        
        for bar, time_val in zip(bars2, avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 3. Throughput Comparison
        throughputs = [self.results[s]['throughput'] for s in strategies]
        bars3 = ax3.bar(strategies, throughputs, alpha=0.8, color='lightgreen')
        ax3.set_ylabel('Throughput (requests/second)')
        ax3.set_title('Throughput by Routing Strategy')
        
        for bar, throughput in zip(bars3, throughputs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{throughput:.1f}', ha='center', va='bottom')
        
        # 4. Worker Distribution for Weighted Strategy
        if 'weighted' in self.results:
            worker_dist = self.results['weighted']['worker_distribution']
            worker_names = [name.replace('_', ' ').title() for name in worker_dist.keys()]
            request_counts = list(worker_dist.values())
            
            ax4.pie(request_counts, labels=worker_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Request Distribution (Weighted Strategy)')
        
        # Rotate x-axis labels for better readability
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_adaptive_routing(self):
        """Demonstrate adaptive routing capabilities"""
        print("\n=== Adaptive Routing Demonstration ===")
        
        # Create adaptive router
        adaptive_router = AdaptiveRouter("AdaptiveDemo")
        
        # Add workers
        workers = self.create_test_workers()
        for worker in workers:
            adaptive_router.add_worker(worker)
        
        # Simulate changing load conditions
        print("Simulating changing system conditions...")
        
        # Phase 1: Normal load
        print("\nPhase 1: Normal load conditions")
        requests = self.create_test_requests(20)
        for request in requests[:10]:
            worker = adaptive_router.select_worker(request)
            if worker:
                print(f"  Request -> {worker.name}")
        
        # Phase 2: Simulate high load on some workers
        print("\nPhase 2: High load conditions")
        workers[0].current_connections = 18  # Near capacity
        workers[1].current_connections = 8   # High load
        
        for request in requests[10:]:
            worker = adaptive_router.select_worker(request)
            if worker:
                print(f"  Request -> {worker.name} (load: {worker.load_factor:.2f})")
        
        adaptive_router.shutdown()
        print("Adaptive routing demonstration complete!")

def main():
    """Main demonstration function"""
    demo = RouterDemo()
    
    # Run comprehensive benchmark
    results = demo.run_comprehensive_benchmark()
    
    # Visualize results
    demo.visualize_results("router_benchmark_results.png")
    
    # Demonstrate adaptive routing
    demo.demonstrate_adaptive_routing()
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Strategy':<20} {'Success Rate':<12} {'Avg Time':<10} {'Throughput':<12}")
    print("-" * 60)
    
    for strategy, result in results.items():
        print(f"{strategy:<20} {result['success_rate']:<12.2%} "
              f"{result['avg_processing_time']:<10.3f} "
              f"{result['throughput']:<12.1f}")
    
    print("\nRouter Component demonstration complete!")
    print("Next: Explore Graph Component in Section 2.2")

if __name__ == "__main__":
    main()
