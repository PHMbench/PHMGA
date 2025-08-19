"""
Enhanced Case 1: Unified PHMGA System Implementation

This module demonstrates the integration of all PHMGA components into a unified
system for bearing fault diagnosis with real-time monitoring and production-ready features.

Author: PHMGA Tutorial Series
License: MIT
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

# Import PHMGA components
from tutorials.part2.router.base_router import Worker, Request
from tutorials.part2.router.strategies import create_router, RoutingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SignalData:
    """Represents signal data for processing"""
    id: str
    data: List[float]
    sampling_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ProcessingResult:
    """Results from signal processing pipeline"""
    signal_id: str
    features: Dict[str, float]
    classification: str
    confidence: float
    processing_time: float
    pipeline_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class SignalProcessor:
    """Advanced signal processing with genetic algorithm optimization"""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.processing_history = []
        self.performance_metrics = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'accuracy': 0.0,
            'last_updated': time.time()
        }
    
    def process_signal(self, signal: SignalData) -> ProcessingResult:
        """
        Process signal using optimized pipeline
        
        Args:
            signal: Input signal data
            
        Returns:
            Processing results with features and classification
        """
        start_time = time.time()
        
        try:
            # Simulate advanced signal processing pipeline
            features = self._extract_features(signal)
            classification = self._classify_signal(features)
            confidence = self._calculate_confidence(features, classification)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                signal_id=signal.id,
                features=features,
                classification=classification,
                confidence=confidence,
                processing_time=processing_time,
                pipeline_steps=[
                    "preprocessing",
                    "feature_extraction", 
                    "classification",
                    "confidence_estimation"
                ],
                metadata={
                    'processor_id': self.processor_id,
                    'sampling_rate': signal.sampling_rate,
                    'signal_length': len(signal.data)
                }
            )
            
            self._update_metrics(result)
            return result
            
        except Exception as e:
            logger.error(f"Signal processing failed for {signal.id}: {e}")
            raise
    
    def _extract_features(self, signal: SignalData) -> Dict[str, float]:
        """Extract features from signal data"""
        import numpy as np
        
        data = np.array(signal.data)
        
        # Time domain features
        features = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'rms': float(np.sqrt(np.mean(data**2))),
            'peak': float(np.max(np.abs(data))),
            'crest_factor': float(np.max(np.abs(data)) / np.sqrt(np.mean(data**2))),
            'skewness': float(self._calculate_skewness(data)),
            'kurtosis': float(self._calculate_kurtosis(data))
        }
        
        # Frequency domain features (simplified)
        fft = np.fft.fft(data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        features.update({
            'spectral_centroid': float(np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)),
            'spectral_rolloff': float(self._calculate_spectral_rolloff(magnitude)),
            'spectral_flux': float(np.sum(np.diff(magnitude)**2))
        })
        
        return features
    
    def _classify_signal(self, features: Dict[str, float]) -> str:
        """Classify signal based on extracted features"""
        # Simplified classification logic
        # In production, this would use trained ML models
        
        if features['crest_factor'] > 5.0:
            return "outer_race_fault"
        elif features['kurtosis'] > 4.0:
            return "inner_race_fault"
        elif features['spectral_flux'] > 1000:
            return "ball_fault"
        elif features['rms'] > 0.5:
            return "cage_fault"
        else:
            return "healthy"
    
    def _calculate_confidence(self, features: Dict[str, float], classification: str) -> float:
        """Calculate classification confidence"""
        # Simplified confidence calculation
        # In production, this would use model uncertainty estimation
        
        confidence_factors = {
            'outer_race_fault': features['crest_factor'] / 10.0,
            'inner_race_fault': features['kurtosis'] / 8.0,
            'ball_fault': min(features['spectral_flux'] / 2000.0, 1.0),
            'cage_fault': features['rms'],
            'healthy': 1.0 - max(features['crest_factor'] / 10.0, features['kurtosis'] / 8.0)
        }
        
        return min(max(confidence_factors.get(classification, 0.5), 0.0), 1.0)
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _calculate_spectral_rolloff(self, magnitude, rolloff_percent=0.85):
        """Calculate spectral rolloff frequency"""
        import numpy as np
        total_energy = np.sum(magnitude)
        cumulative_energy = np.cumsum(magnitude)
        rolloff_index = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
        return rolloff_index[0] if len(rolloff_index) > 0 else len(magnitude) - 1
    
    def _update_metrics(self, result: ProcessingResult):
        """Update processor performance metrics"""
        self.performance_metrics['total_processed'] += 1
        
        # Update average processing time (exponential moving average)
        alpha = 0.1
        self.performance_metrics['avg_processing_time'] = (
            alpha * result.processing_time + 
            (1 - alpha) * self.performance_metrics['avg_processing_time']
        )
        
        self.performance_metrics['last_updated'] = time.time()
        
        # Store processing history (keep last 100 results)
        self.processing_history.append(result)
        if len(self.processing_history) > 100:
            self.processing_history.pop(0)

class UnifiedPHMGASystem:
    """
    Unified PHMGA system integrating all components for production use
    
    This system combines:
    - Router Component for workflow orchestration
    - Signal Processing with genetic algorithm optimization
    - State Management for persistence and recovery
    - Real-time monitoring and alerting
    """
    
    def __init__(self, system_id: str = None, config: Dict[str, Any] = None):
        """
        Initialize unified PHMGA system
        
        Args:
            system_id: Unique identifier for this system instance
            config: System configuration parameters
        """
        self.system_id = system_id or str(uuid.uuid4())
        self.config = config or self._default_config()
        
        # Initialize components
        self.router = create_router(
            RoutingStrategy.ADAPTIVE, 
            f"PHMGA_Router_{self.system_id}"
        )
        
        self.processors = {}
        self.results_store = {}
        self.system_metrics = {
            'start_time': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'current_load': 0.0
        }
        
        # Initialize signal processors as workers
        self._initialize_processors()
        
        # Start monitoring
        self.monitoring_active = False
        self._start_monitoring()
        
        logger.info(f"Unified PHMGA System {self.system_id} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default system configuration"""
        return {
            'num_processors': 3,
            'max_concurrent_per_processor': 5,
            'monitoring_interval': 10.0,
            'results_retention_hours': 24,
            'auto_scaling_enabled': True,
            'performance_threshold': 0.8
        }
    
    def _initialize_processors(self):
        """Initialize signal processors and register as workers"""
        for i in range(self.config['num_processors']):
            processor_id = f"processor_{i}"
            processor = SignalProcessor(processor_id)
            self.processors[processor_id] = processor
            
            # Create worker for router
            worker = Worker(
                id=processor_id,
                name=f"Signal Processor {i}",
                weight=1.0,
                max_concurrent=self.config['max_concurrent_per_processor']
            )
            
            self.router.add_worker(worker)
        
        # Start router health checks
        self.router.start_health_checks()
    
    async def process_signal_async(self, signal: SignalData) -> ProcessingResult:
        """
        Process signal asynchronously through the unified system
        
        Args:
            signal: Input signal data
            
        Returns:
            Processing results
        """
        # Create processing request
        request = Request(
            payload={
                'signal': signal,
                'operation': 'process_signal'
            },
            priority=1,
            timeout=30.0
        )
        
        # Route and process request
        processed_request = self.router.process_request(request)
        
        if processed_request.status.value == 'completed':
            result = processed_request.result
            self.results_store[signal.id] = result
            self._update_system_metrics(True, processed_request.processing_time)
            return result
        else:
            self._update_system_metrics(False, processed_request.processing_time)
            raise Exception(f"Processing failed: {processed_request.error}")
    
    def process_signal(self, signal: SignalData) -> ProcessingResult:
        """
        Process signal synchronously
        
        Args:
            signal: Input signal data
            
        Returns:
            Processing results
        """
        return asyncio.run(self.process_signal_async(signal))
    
    def process_batch(self, signals: List[SignalData]) -> List[ProcessingResult]:
        """
        Process multiple signals in batch
        
        Args:
            signals: List of signal data to process
            
        Returns:
            List of processing results
        """
        results = []
        
        for signal in signals:
            try:
                result = self.process_signal(signal)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process signal {signal.id}: {e}")
                # Create error result
                error_result = ProcessingResult(
                    signal_id=signal.id,
                    features={},
                    classification="error",
                    confidence=0.0,
                    processing_time=0.0,
                    pipeline_steps=["error"],
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        router_metrics = self.router.get_metrics()
        
        processor_status = {}
        for proc_id, processor in self.processors.items():
            processor_status[proc_id] = {
                'metrics': processor.performance_metrics,
                'recent_results': len(processor.processing_history),
                'status': 'active'
            }
        
        return {
            'system_id': self.system_id,
            'uptime': time.time() - self.system_metrics['start_time'],
            'system_metrics': self.system_metrics,
            'router_metrics': router_metrics,
            'processor_status': processor_status,
            'results_stored': len(self.results_store),
            'monitoring_active': self.monitoring_active
        }
    
    def _update_system_metrics(self, success: bool, processing_time: float):
        """Update system-level metrics"""
        self.system_metrics['total_requests'] += 1
        
        if success:
            self.system_metrics['successful_requests'] += 1
        else:
            self.system_metrics['failed_requests'] += 1
        
        # Update average response time
        alpha = 0.1
        self.system_metrics['avg_response_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.system_metrics['avg_response_time']
        )
        
        # Update current load
        active_connections = sum(
            worker.current_connections 
            for worker in self.router.workers.values()
        )
        total_capacity = sum(
            worker.max_concurrent 
            for worker in self.router.workers.values()
        )
        
        self.system_metrics['current_load'] = (
            active_connections / total_capacity if total_capacity > 0 else 0
        )
    
    def _start_monitoring(self):
        """Start system monitoring"""
        import threading
        
        def monitoring_loop():
            self.monitoring_active = True
            while self.monitoring_active:
                try:
                    self._perform_health_check()
                    self._cleanup_old_results()
                    time.sleep(self.config['monitoring_interval'])
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(5)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        logger.info("System monitoring started")
    
    def _perform_health_check(self):
        """Perform system health check"""
        status = self.get_system_status()
        
        # Check system load
        if status['system_metrics']['current_load'] > self.config['performance_threshold']:
            logger.warning(f"High system load: {status['system_metrics']['current_load']:.2%}")
        
        # Check processor health
        for proc_id, proc_status in status['processor_status'].items():
            if proc_status['metrics']['total_processed'] == 0:
                logger.warning(f"Processor {proc_id} has not processed any signals")
    
    def _cleanup_old_results(self):
        """Clean up old results based on retention policy"""
        cutoff_time = time.time() - (self.config['results_retention_hours'] * 3600)
        
        to_remove = []
        for signal_id, result in self.results_store.items():
            if hasattr(result, 'metadata') and 'timestamp' in result.metadata:
                if result.metadata['timestamp'] < cutoff_time:
                    to_remove.append(signal_id)
        
        for signal_id in to_remove:
            del self.results_store[signal_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old results")
    
    def save_state(self, filepath: str):
        """Save system state to file"""
        state = {
            'system_id': self.system_id,
            'config': self.config,
            'system_metrics': self.system_metrics,
            'results_store': self.results_store,
            'timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"System state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.system_id = state['system_id']
        self.config = state['config']
        self.system_metrics = state['system_metrics']
        self.results_store = state['results_store']
        
        logger.info(f"System state loaded from {filepath}")
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info(f"Shutting down PHMGA System {self.system_id}")
        
        self.monitoring_active = False
        self.router.shutdown()
        
        # Save final state
        self.save_state(f"phmga_system_{self.system_id}_final_state.pkl")
        
        logger.info("System shutdown complete")

if __name__ == "__main__":
    # Example usage and demonstration
    print("=== Unified PHMGA System Demo ===")
    
    # Create system
    system = UnifiedPHMGASystem("demo_system")
    
    # Create sample signals
    import numpy as np
    
    signals = []
    for i in range(5):
        # Generate synthetic bearing signal
        t = np.linspace(0, 1, 1000)
        signal_data = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)
        
        # Add fault characteristics for some signals
        if i % 2 == 0:
            signal_data += 0.5 * np.sin(2 * np.pi * 200 * t)  # Add fault frequency
        
        signal = SignalData(
            id=f"signal_{i}",
            data=signal_data.tolist(),
            sampling_rate=1000.0,
            metadata={'source': 'synthetic', 'fault_type': 'outer_race' if i % 2 == 0 else 'healthy'}
        )
        signals.append(signal)
    
    # Process signals
    print("\nProcessing signals...")
    results = system.process_batch(signals)
    
    # Display results
    print("\nProcessing Results:")
    for result in results:
        print(f"Signal {result.signal_id}: {result.classification} (confidence: {result.confidence:.2f})")
    
    # Show system status
    print("\nSystem Status:")
    status = system.get_system_status()
    print(f"Total Requests: {status['system_metrics']['total_requests']}")
    print(f"Success Rate: {status['system_metrics']['successful_requests'] / status['system_metrics']['total_requests']:.2%}")
    print(f"Average Response Time: {status['system_metrics']['avg_response_time']:.3f}s")
    print(f"Current Load: {status['system_metrics']['current_load']:.2%}")
    
    # Shutdown
    system.shutdown()
    print("\nDemo complete!")
