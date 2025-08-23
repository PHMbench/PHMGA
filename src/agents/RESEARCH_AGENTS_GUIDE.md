# PHM Research Agents - Comprehensive Guide

This guide provides complete documentation for the enhanced PHM Research Agent System, featuring improved architecture, performance monitoring, and service-oriented design.

## Quick Navigation Index

- [Agent Overview](#agent-overview)
- [Agent Responsibility Matrix](#agent-responsibility-matrix)
- [Architecture Diagram](#architecture-diagram)
- [Usage Examples](#usage-examples)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Guidelines](#performance-guidelines)
- [Extension Guide](#extension-guide)

## Agent Overview

The PHM Research Agent System consists of four specialized agents working together to provide comprehensive bearing fault diagnosis and research insights:

### Core Research Agents

#### 1. Data Analyst Agent (`data_analyst_agent.py`)
**Primary Responsibilities:**
- Signal quality assessment and anomaly detection
- Statistical characterization and feature extraction
- Feature space exploration using PCA and clustering
- Data preprocessing recommendations
- Uncertainty quantification for measurement quality

**Key Features:**
- Parallel feature extraction with caching
- Multiple anomaly detection methods
- Configurable analysis depth (quick mode vs comprehensive)
- Performance monitoring and resource tracking

#### 2. Algorithm Researcher Agent (`algorithm_researcher_agent.py`)
**Primary Responsibilities:**
- Signal processing method comparison and optimization
- Machine learning algorithm evaluation
- Hyperparameter optimization and cross-validation
- Performance benchmarking and separability analysis

**Key Features:**
- Automated hyperparameter tuning
- Cross-validation with multiple metrics
- Algorithm performance comparison
- Feature importance analysis

#### 3. Domain Expert Agent (`domain_expert_agent.py`)
**Primary Responsibilities:**
- Physics-based validation and bearing analysis
- Failure mode identification and characterization
- Frequency analysis and bearing fault detection
- Domain knowledge integration and validation

**Key Features:**
- Bearing frequency calculation and analysis
- Physics-based signal validation
- Failure mode pattern recognition
- Domain-specific hypothesis generation

#### 4. Integration Agent (`integration_agent.py`)
**Primary Responsibilities:**
- Multi-agent result synthesis and conflict resolution
- Consensus building and confidence assessment
- Cross-agent consistency validation
- Final research insights generation

**Key Features:**
- Intelligent conflict resolution
- Confidence-weighted consensus building
- Cross-agent validation
- Comprehensive result synthesis

### Supporting Infrastructure

#### Agent Factory (`agent_factory.py`)
- Centralized agent creation and configuration management
- Dependency injection and service registration
- Predefined agent presets for common use cases
- Fluent builder pattern for agent configuration

#### Service Layer (`services.py`)
- Decoupled signal processing and ML services
- Parallel processing and caching capabilities
- Statistical analysis and feature extraction services
- Performance monitoring and resource management

#### Enhanced Base Classes (`research_base.py`)
- Performance monitoring decorators
- Circuit breaker pattern for reliability
- Input validation and error handling
- Standardized communication interfaces

## Agent Responsibility Matrix

| Agent | Input | Primary Output | Secondary Output | Performance Focus |
|-------|-------|----------------|------------------|-------------------|
| **Data Analyst** | Raw signals, sampling frequency | Signal quality metrics, statistical features | Anomaly detection, preprocessing recommendations | Memory efficiency, parallel processing |
| **Algorithm Researcher** | Feature sets, signal data | Algorithm performance comparison | Hyperparameter optimization results | Computational efficiency, cross-validation |
| **Domain Expert** | Signal characteristics, frequency data | Physics validation, bearing analysis | Failure mode identification | Domain accuracy, knowledge integration |
| **Integration** | All agent results | Synthesized findings, consensus | Conflict resolution, confidence assessment | Result quality, consistency validation |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHM Research Agent System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    Data     │    │ Algorithm   │    │   Domain    │         │
│  │  Analyst    │    │ Researcher  │    │   Expert    │         │
│  │   Agent     │    │   Agent     │    │   Agent     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             │                                  │
│                    ┌─────────────┐                             │
│                    │Integration  │                             │
│                    │   Agent     │                             │
│                    └─────────────┘                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Service Layer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  Feature    │ │Statistical  │ │   ML Model  │ │  Cache    │ │
│  │Extraction   │ │ Analysis    │ │  Service    │ │ Service   │ │
│  │  Service    │ │  Service    │ │             │ │           │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │   Agent     │ │Performance  │ │   Input     │ │  Config   │ │
│  │  Factory    │ │ Monitoring  │ │ Validation  │ │ Manager   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Agent Usage

```python
from src.agents.agent_factory import AgentFactory, AgentType
from src.agents.data_analyst_agent import DataAnalystAgent

# Method 1: Direct instantiation
agent = DataAnalystAgent(
    config={
        "quick_mode": False,
        "enable_advanced_features": True,
        "use_parallel": True
    }
)

# Method 2: Using Agent Factory
factory = AgentFactory()
agent = factory.create_agent_builder(AgentType.DATA_ANALYST) \
    .with_name("my_data_analyst") \
    .with_config(enable_advanced_features=True) \
    .with_validator("state", "state") \
    .build()

# Method 3: Using Presets
agents = factory.create_agent_from_preset("QuickDiagnosis")
```

### Agent Configuration Examples

```python
# Quick diagnosis configuration
quick_config = {
    "quick_mode": True,
    "max_features": 5,
    "enable_advanced_features": False,
    "use_parallel": False
}

# Comprehensive research configuration
comprehensive_config = {
    "quick_mode": False,
    "enable_advanced_features": True,
    "enable_pca_analysis": True,
    "enable_clustering": True,
    "use_parallel": True,
    "use_cache": True
}

# Predictive maintenance configuration
predictive_config = {
    "focus_on_degradation": True,
    "enable_trend_analysis": True,
    "health_features": ["rms", "kurtosis", "crest_factor"],
    "enable_rul_estimation": True
}
```

### Performance Monitoring Example

```python
# Analyze with performance monitoring
analysis_result = agent.analyze(state)

print(f"Analysis Results:")
print(f"  Confidence: {analysis_result.confidence:.3f}")
print(f"  Execution Time: {analysis_result.execution_time:.2f}s")
print(f"  Memory Usage: {analysis_result.memory_usage:.1f}MB")
print(f"  Success: {analysis_result.is_successful()}")

# Get agent performance metrics
metrics = agent.get_performance_metrics()
if not metrics.get("no_executions"):
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Avg Execution Time: {metrics['avg_execution_time']:.2f}s")
```

### Service Layer Usage

```python
from src.agents.services import service_registry

# Get services
feature_service = service_registry.get("feature_extraction")
stats_service = service_registry.get("statistical_analysis")

# Extract features with caching and parallel processing
features = feature_service.extract_features(
    signals, 
    feature_names=["mean", "std", "rms", "kurtosis"],
    use_cache=True,
    use_parallel=True
)

# Perform statistical analysis
stats = stats_service.analyze(signals, methods=["basic_stats", "outlier_detection"])
```

## Configuration Reference

### Data Analyst Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quick_mode` | bool | False | Enable quick analysis mode with reduced features |
| `enable_advanced_features` | bool | True | Enable advanced analysis features |
| `enable_pca_analysis` | bool | True | Enable PCA-based feature space analysis |
| `enable_clustering` | bool | True | Enable clustering analysis |
| `max_features` | int | None | Maximum number of features to extract |
| `use_parallel` | bool | True | Enable parallel feature extraction |
| `use_cache` | bool | True | Enable result caching |

### Algorithm Researcher Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_hyperparameter_optimization` | bool | True | Enable hyperparameter tuning |
| `enable_cross_validation` | bool | True | Enable cross-validation |
| `max_algorithms` | int | 10 | Maximum number of algorithms to test |
| `cv_folds` | int | 5 | Number of cross-validation folds |
| `random_state` | int | 42 | Random state for reproducibility |

### Domain Expert Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_physics_validation` | bool | True | Enable physics-based validation |
| `enable_bearing_analysis` | bool | True | Enable bearing-specific analysis |
| `enable_frequency_analysis` | bool | True | Enable frequency domain analysis |
| `bearing_parameters` | dict | {} | Custom bearing parameters |

### Integration Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_conflict_resolution` | bool | True | Enable conflict resolution |
| `enable_consensus_building` | bool | True | Enable consensus building |
| `confidence_threshold` | float | 0.7 | Minimum confidence for consensus |

### Performance Limits Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_execution_time` | int | 300 | Maximum execution time in seconds |
| `max_memory_mb` | int | 1024 | Maximum memory usage in MB |
| `max_failures` | int | 3 | Maximum failures before circuit breaker |

## Troubleshooting

### Common Issues and Solutions

#### Low Research Confidence
**Symptoms:** Research confidence < 0.5, poor analysis results
**Causes:** 
- Poor signal quality (low SNR)
- Insufficient data
- Agent configuration issues
- Service failures

**Solutions:**
1. Check signal SNR and quality metrics
2. Increase `max_research_iterations` in workflow config
3. Review agent-specific error logs
4. Validate input data format and content
5. Check service registry for failed services

#### Memory Usage Issues
**Symptoms:** High memory consumption, out-of-memory errors
**Causes:**
- Large signal arrays processed without chunking
- Memory leaks in long-running workflows
- Inefficient feature extraction

**Solutions:**
1. Enable chunked processing for large signals
2. Reduce `max_features` in agent configuration
3. Enable caching to avoid recomputation
4. Monitor memory usage with performance decorators
5. Use `quick_mode` for memory-constrained environments

#### Performance Problems
**Symptoms:** Slow execution, timeouts
**Causes:**
- Sequential processing instead of parallel
- Disabled caching
- Complex analysis configurations

**Solutions:**
1. Enable `use_parallel=True` in agent configs
2. Enable `use_cache=True` for repeated operations
3. Use `quick_mode=True` for faster analysis
4. Reduce analysis complexity in configuration
5. Check service performance metrics

#### Agent Communication Errors
**Symptoms:** Integration failures, inconsistent results
**Causes:**
- Agent result format mismatches
- Missing dependencies
- Service unavailability

**Solutions:**
1. Validate agent result formats
2. Check service registry for missing services
3. Review agent dependency configurations
4. Ensure proper agent initialization order
5. Check network connectivity for distributed setups

### Error Message Reference

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "Circuit breaker open" | Too many consecutive failures | Wait for reset timeout or fix underlying issue |
| "Service not found" | Missing service registration | Register service in service registry |
| "Invalid agent configuration" | Configuration validation failed | Check configuration parameters and types |
| "Insufficient data for analysis" | Empty or invalid signal data | Validate input signal format and content |
| "Memory limit exceeded" | High memory usage | Enable chunked processing or reduce data size |

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("src.agents").setLevel(logging.DEBUG)

# Enable performance monitoring
from src.agents.research_base import monitor_performance

# All agent methods will now log detailed performance information
```

## Performance Guidelines

### Memory Usage Expectations

| Agent Type | Typical Memory Usage | Large Dataset Usage |
|------------|---------------------|-------------------|
| Data Analyst | 50-200 MB | 500-1000 MB |
| Algorithm Researcher | 100-300 MB | 1-2 GB |
| Domain Expert | 20-100 MB | 200-500 MB |
| Integration | 10-50 MB | 100-200 MB |

### Processing Time Estimates

| Operation | Quick Mode | Standard Mode | Comprehensive Mode |
|-----------|------------|---------------|-------------------|
| Signal Analysis | 1-5 seconds | 5-15 seconds | 15-60 seconds |
| Feature Extraction | 0.5-2 seconds | 2-8 seconds | 8-30 seconds |
| ML Algorithm Testing | 2-10 seconds | 10-30 seconds | 30-120 seconds |
| Integration & Synthesis | 0.5-2 seconds | 2-5 seconds | 5-15 seconds |

### Optimization Tips

1. **Use Parallel Processing:** Enable `use_parallel=True` for multi-core systems
2. **Enable Caching:** Set `use_cache=True` to avoid recomputation
3. **Quick Mode:** Use `quick_mode=True` for rapid prototyping
4. **Feature Selection:** Limit `max_features` for faster processing
5. **Memory Monitoring:** Use performance decorators to track resource usage
6. **Service Optimization:** Configure services for your specific use case

### Scalability Considerations

- **Horizontal Scaling:** Agents can be distributed across multiple processes
- **Vertical Scaling:** Increase memory limits and CPU cores for better performance
- **Caching Strategy:** Use persistent caching for production environments
- **Load Balancing:** Distribute agent workload based on computational requirements

This guide provides comprehensive information for using and extending the PHM Research Agent System. For additional support, refer to the source code documentation and test examples.
