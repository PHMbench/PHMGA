"""
Base Router Component for PHMGA Tutorial Part 2

This module provides the foundational router architecture for workflow orchestration
within the PHMGA framework. It supports multiple routing strategies, async operations,
and comprehensive error handling.

Author: PHMGA Tutorial Series
License: MIT
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Enumeration of available routing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PRIORITY = "priority"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"

class RequestStatus(Enum):
    """Enumeration of request statuses"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Request:
    """Represents a request to be routed and processed"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher values = higher priority
    timeout: float = 30.0  # Timeout in seconds
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    worker_id: Optional[str] = None

@dataclass
class Worker:
    """Represents a worker that can process requests"""
    id: str
    name: str
    weight: float = 1.0
    max_concurrent: int = 10
    current_connections: int = 0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    def increment_connections(self) -> bool:
        """Increment connection count if under limit"""
        with self._lock:
            if self.current_connections < self.max_concurrent:
                self.current_connections += 1
                return True
            return False
    
    def decrement_connections(self):
        """Decrement connection count"""
        with self._lock:
            if self.current_connections > 0:
                self.current_connections -= 1
    
    def update_stats(self, success: bool, response_time: float):
        """Update worker statistics"""
        with self._lock:
            self.total_requests += 1
            if not success:
                self.failed_requests += 1
            
            # Update average response time (exponential moving average)
            alpha = 0.1
            self.avg_response_time = (alpha * response_time + 
                                    (1 - alpha) * self.avg_response_time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)"""
        return self.current_connections / self.max_concurrent

class RouterException(Exception):
    """Base exception for router-related errors"""
    pass

class NoAvailableWorkersException(RouterException):
    """Raised when no workers are available to handle a request"""
    pass

class RequestTimeoutException(RouterException):
    """Raised when a request times out"""
    pass

class WorkerHealthCheckException(RouterException):
    """Raised when worker health check fails"""
    pass

class BaseRouter(ABC):
    """
    Abstract base class for all router implementations
    
    Provides common functionality for request routing, worker management,
    and error handling. Subclasses must implement the route_request method.
    """
    
    def __init__(self, name: str = "BaseRouter"):
        """
        Initialize the router
        
        Args:
            name: Router instance name for logging and identification
        """
        self.name = name
        self.workers: Dict[str, Worker] = {}
        self.requests: Dict[str, Request] = {}
        self.strategy = RoutingStrategy.ROUND_ROBIN
        self._round_robin_index = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._health_check_interval = 30.0  # seconds
        self._health_check_task = None
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'active_connections': 0
        }
        
        logger.info(f"Router '{self.name}' initialized")
    
    def add_worker(self, worker: Worker) -> None:
        """
        Add a worker to the router
        
        Args:
            worker: Worker instance to add
        """
        with self._lock:
            self.workers[worker.id] = worker
            logger.info(f"Added worker '{worker.name}' (ID: {worker.id}) to router '{self.name}'")
    
    def remove_worker(self, worker_id: str) -> bool:
        """
        Remove a worker from the router
        
        Args:
            worker_id: ID of the worker to remove
            
        Returns:
            True if worker was removed, False if not found
        """
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers.pop(worker_id)
                logger.info(f"Removed worker '{worker.name}' (ID: {worker_id}) from router '{self.name}'")
                return True
            return False
    
    def get_healthy_workers(self) -> List[Worker]:
        """
        Get list of healthy workers
        
        Returns:
            List of healthy Worker instances
        """
        with self._lock:
            return [worker for worker in self.workers.values() if worker.is_healthy]
    
    def get_available_workers(self) -> List[Worker]:
        """
        Get list of workers that can accept new connections
        
        Returns:
            List of available Worker instances
        """
        healthy_workers = self.get_healthy_workers()
        return [worker for worker in healthy_workers 
                if worker.current_connections < worker.max_concurrent]
    
    @abstractmethod
    def select_worker(self, request: Request) -> Optional[Worker]:
        """
        Select a worker for the given request based on routing strategy
        
        Args:
            request: Request to be routed
            
        Returns:
            Selected Worker instance or None if no worker available
        """
        pass
    
    def process_request(self, request: Request) -> Request:
        """
        Process a request by routing it to an appropriate worker
        
        Args:
            request: Request to process
            
        Returns:
            Updated Request with results or error information
        """
        start_time = time.time()
        request.status = RequestStatus.PROCESSING
        
        try:
            # Select worker
            worker = self.select_worker(request)
            if not worker:
                raise NoAvailableWorkersException("No available workers for request")
            
            # Check if worker can accept connection
            if not worker.increment_connections():
                raise NoAvailableWorkersException(f"Worker {worker.id} at capacity")
            
            request.worker_id = worker.id
            
            try:
                # Simulate request processing (in real implementation, this would call the actual worker)
                result = self._simulate_worker_processing(request, worker)
                
                request.result = result
                request.status = RequestStatus.COMPLETED
                
                # Update worker stats
                processing_time = time.time() - start_time
                worker.update_stats(success=True, response_time=processing_time)
                
            finally:
                worker.decrement_connections()
        
        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            
            if request.worker_id:
                worker = self.workers.get(request.worker_id)
                if worker:
                    processing_time = time.time() - start_time
                    worker.update_stats(success=False, response_time=processing_time)
            
            logger.error(f"Request {request.id} failed: {e}")
        
        finally:
            request.processing_time = time.time() - start_time
            self._update_metrics(request)
        
        return request
    
    def _simulate_worker_processing(self, request: Request, worker: Worker) -> Dict[str, Any]:
        """
        Simulate worker processing (replace with actual worker call in production)
        
        Args:
            request: Request being processed
            worker: Worker processing the request
            
        Returns:
            Simulated processing result
        """
        # Simulate processing time based on worker performance
        processing_time = max(0.1, worker.avg_response_time + 0.1)
        time.sleep(processing_time)
        
        # Simulate occasional failures
        import random
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated worker processing failure")
        
        return {
            'worker_id': worker.id,
            'worker_name': worker.name,
            'processed_at': time.time(),
            'payload_size': len(str(request.payload)),
            'result': f"Processed by {worker.name}"
        }
    
    def _update_metrics(self, request: Request) -> None:
        """Update router metrics"""
        with self._lock:
            self._metrics['total_requests'] += 1
            
            if request.status == RequestStatus.COMPLETED:
                self._metrics['successful_requests'] += 1
            elif request.status == RequestStatus.FAILED:
                self._metrics['failed_requests'] += 1
            
            if request.processing_time:
                # Update average response time (exponential moving average)
                alpha = 0.1
                self._metrics['avg_response_time'] = (
                    alpha * request.processing_time + 
                    (1 - alpha) * self._metrics['avg_response_time']
                )
            
            # Update active connections
            self._metrics['active_connections'] = sum(
                worker.current_connections for worker in self.workers.values()
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get router performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            metrics = self._metrics.copy()
            metrics['worker_count'] = len(self.workers)
            metrics['healthy_workers'] = len(self.get_healthy_workers())
            metrics['available_workers'] = len(self.get_available_workers())
            
            if metrics['total_requests'] > 0:
                metrics['success_rate'] = (
                    metrics['successful_requests'] / metrics['total_requests']
                )
            else:
                metrics['success_rate'] = 0.0
            
            return metrics
    
    def health_check_worker(self, worker: Worker) -> bool:
        """
        Perform health check on a worker
        
        Args:
            worker: Worker to check
            
        Returns:
            True if worker is healthy, False otherwise
        """
        try:
            # Simulate health check (in production, this would ping the actual worker)
            import random
            is_healthy = random.random() > 0.1  # 90% health rate
            
            worker.is_healthy = is_healthy
            worker.last_health_check = time.time()
            
            if not is_healthy:
                logger.warning(f"Worker {worker.id} failed health check")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for worker {worker.id}: {e}")
            worker.is_healthy = False
            return False
    
    def start_health_checks(self) -> None:
        """Start periodic health checks for all workers"""
        if self._health_check_task is None:
            self._health_check_task = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self._health_check_task.start()
            logger.info(f"Started health checks for router '{self.name}'")
    
    def _health_check_loop(self) -> None:
        """Background loop for health checks"""
        while True:
            try:
                with self._lock:
                    workers_to_check = list(self.workers.values())
                
                for worker in workers_to_check:
                    self.health_check_worker(worker)
                
                time.sleep(self._health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def shutdown(self) -> None:
        """Shutdown the router and cleanup resources"""
        logger.info(f"Shutting down router '{self.name}'")
        self._executor.shutdown(wait=True)
    
    def __str__(self) -> str:
        return f"Router(name='{self.name}', workers={len(self.workers)}, strategy={self.strategy.value})"
    
    def __repr__(self) -> str:
        return self.__str__()

if __name__ == "__main__":
    # Example usage and testing
    print("=== Base Router Component Demo ===")

    # This is an abstract class, so we can't instantiate it directly
    # See strategies.py for concrete implementations

    # Create some sample workers
    worker1 = Worker(id="worker_1", name="Worker 1", weight=1.0, max_concurrent=5)
    worker2 = Worker(id="worker_2", name="Worker 2", weight=2.0, max_concurrent=10)
    worker3 = Worker(id="worker_3", name="Worker 3", weight=0.5, max_concurrent=3)

    print(f"Created workers:")
    print(f"  {worker1.name}: weight={worker1.weight}, max_concurrent={worker1.max_concurrent}")
    print(f"  {worker2.name}: weight={worker2.weight}, max_concurrent={worker2.max_concurrent}")
    print(f"  {worker3.name}: weight={worker3.weight}, max_concurrent={worker3.max_concurrent}")

    # Create a sample request
    request = Request(
        payload={"task": "optimize_function", "parameters": {"x": 1.0, "y": 2.0}},
        priority=1,
        timeout=10.0
    )

    print(f"\nCreated request: {request.id}")
    print(f"  Payload: {request.payload}")
    print(f"  Priority: {request.priority}")
    print(f"  Timeout: {request.timeout}s")

    print("\nBase router component ready for concrete implementations!")
    print("See strategies.py for RoundRobinRouter, WeightedRouter, etc.")
