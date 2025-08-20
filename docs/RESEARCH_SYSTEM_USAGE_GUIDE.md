# PHM Research Agent System - Usage Guide

## Overview

The PHM Research Agent System transforms traditional bearing fault diagnosis into an advanced research-driven platform that autonomously investigates signal processing techniques, generates hypotheses, and provides comprehensive insights for predictive maintenance.

## System Architecture

### Core Components

1. **Research Agents**
   - **Data Analyst Agent**: Signal quality assessment and exploratory analysis
   - **Algorithm Researcher Agent**: Comparative analysis of signal processing methods
   - **Domain Expert Agent**: Physics-informed validation and bearing fault expertise
   - **Integration Agent**: Multi-agent coordination and conflict resolution

2. **Enhanced Case Studies**
   - **Case 1 Enhanced**: Traditional diagnosis with research agent oversight
   - **Case 2 Predictive**: Advanced predictive maintenance with RUL estimation

3. **Research Workflow**
   - Dynamic LangGraph-based orchestration
   - Parallel agent execution
   - Hypothesis generation and validation
   - Automated research reporting

## Quick Start

### Installation and Setup

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export LANGCHAIN_TRACING_V2=false
export PHM_SAVE_DIR="/path/to/save/directory"
```

3. Validate system installation:
```bash
python test_research_system.py
```

### Running Case Studies

#### Enhanced Case 1: Research-Augmented Diagnosis

```bash
# Run enhanced case with research agents
python -m src.cases.case1_enhanced --config config/case1.yaml --mode enhanced

# Compare traditional vs research-enhanced approaches
python -m src.cases.case1_enhanced --config config/case1.yaml --mode compare

# Run research agents only
python -m src.cases.case1_enhanced --config config/case1.yaml --mode research
```

#### Case 2: Predictive Maintenance

```bash
# Run complete predictive maintenance analysis
python -m src.cases.case2_predictive --config config/case2.yaml
```

## Configuration

### Case 1 Configuration (config/case1.yaml)

```yaml
name: "case1_enhanced"
user_instruction: >
  Analyze bearing signals with research agent oversight.
  Provide comprehensive insights beyond traditional classification.

# Data selection
ref_ids: [47050, 47052, 47044, 47046, 47047]
test_ids: [47051, 47045, 47048, 47054, 47057]

# Research configuration
research:
  max_iterations: 3
  quality_threshold: 0.8
  enable_parallel_agents: true
```

### Case 2 Configuration (config/case2.yaml)

```yaml
name: "case2_predictive"
user_instruction: >
  Perform comprehensive predictive maintenance analysis.
  Estimate remaining useful life and generate maintenance schedules.

# Predictive maintenance parameters
predictive_maintenance:
  health_features: ["rms", "kurtosis", "crest_factor"]
  rul_estimation:
    failure_threshold: 0.3
    degradation_models: ["linear", "exponential", "power_law"]
  
  maintenance_strategies:
    immediate: {rul_threshold: 7, health_threshold: 0.3}
    urgent: {rul_threshold: 30, health_threshold: 0.5}
    planned: {rul_threshold: 90, health_threshold: 0.7}
```

## Advanced Usage

### Custom Research Workflows

```python
from src.research_workflow import build_research_graph
from src.states.research_states import ResearchPHMState

# Build custom research graph
graph = build_research_graph()

# Create research state
state = ResearchPHMState(
    case_name="custom_research",
    user_instruction="Custom research objective",
    # ... other parameters
)

# Run research workflow
for event in graph.stream(state):
    print(f"Node executed: {list(event.keys())}")
```

### Individual Agent Usage

```python
from src.agents.data_analyst_agent import DataAnalystAgent

# Create and configure agent
agent = DataAnalystAgent()

# Run analysis
results = agent.analyze(research_state)
hypotheses = agent.generate_hypotheses(research_state, results)

print(f"Analysis confidence: {results['confidence']:.3f}")
print(f"Generated {len(hypotheses)} hypotheses")
```

### Predictive Maintenance Components

```python
from src.cases.case2_predictive import (
    RULEstimator, HealthIndexCalculator, MaintenanceScheduler
)

# RUL Estimation
rul_estimator = RULEstimator()
health_history = [1.0, 0.9, 0.8, 0.7, 0.6]  # Historical health data
time_points = [0, 30, 60, 90, 120]  # Days
rul_result = rul_estimator.estimate_rul(health_history, time_points)

print(f"Estimated RUL: {rul_result['rul_estimate']:.1f} days")

# Health Index Calculation
health_calc = HealthIndexCalculator()
current_features = {"rms": 1.2, "kurtosis": 4.0}
baseline_features = {"rms": 1.0, "kurtosis": 3.0}
health_result = health_calc.calculate_health_index(current_features, baseline_features)

print(f"Health Index: {health_result['health_index']:.3f}")

# Maintenance Scheduling
scheduler = MaintenanceScheduler()
schedule = scheduler.generate_schedule(
    rul_days=45, 
    health_index=0.6
)

print(f"Maintenance Strategy: {schedule['strategy']}")
print(f"Next Inspection: {schedule['schedule']['next_inspection']}")
```

## Output and Results

### Research Reports

The system generates comprehensive research reports including:

- **Executive Summary**: Key findings and confidence levels
- **Methodology**: Research approach and agent coordination
- **Findings**: Detailed analysis results from each agent
- **Hypotheses**: Generated and validated research hypotheses
- **Recommendations**: Actionable insights for maintenance and operations
- **Conclusions**: Overall assessment and confidence evaluation

### Predictive Maintenance Outputs

Case 2 provides:

- **Health Index**: Comprehensive bearing condition assessment (0-1 scale)
- **RUL Estimation**: Remaining useful life in days with confidence intervals
- **Maintenance Schedule**: Optimal timing for inspections and replacements
- **Risk Assessment**: Operational risk levels and mitigation strategies

### Data Exports

Results can be exported in multiple formats:

```python
# Save research state
from src.utils import save_state
save_state(research_state, "results/research_state.pkl")

# Export maintenance schedule
schedule_data = research_state.predictive_maintenance_schedule
with open("results/maintenance_schedule.json", "w") as f:
    json.dump(schedule_data, f, indent=2, default=str)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes src directory
   - Verify environment variables are set

2. **Low Research Confidence**
   - Check signal quality and data completeness
   - Increase max_research_iterations in configuration
   - Review agent-specific error logs

3. **Agent Coordination Issues**
   - Enable parallel execution in configuration
   - Check for conflicting agent findings in integration results
   - Review research audit trail for coordination problems

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug logging
result = run_enhanced_case("config/case1.yaml", enable_research=True)
```

Check research audit trail:

```python
for entry in research_state.research_audit_trail:
    print(f"{entry.timestamp}: {entry.agent} - {entry.action} (confidence: {entry.confidence:.3f})")
```

### Performance Optimization

1. **Parallel Execution**: Enable parallel agent execution for faster processing
2. **Iteration Limits**: Set appropriate max_research_iterations based on requirements
3. **Quality Thresholds**: Adjust research_quality_threshold to balance speed vs. thoroughness
4. **Agent Selection**: Disable specific agents if not needed for your use case

## Best Practices

### Configuration Management

- Use version control for configuration files
- Document configuration changes and their impact
- Test configurations with validation data before production use

### Research Quality

- Set appropriate quality thresholds based on application criticality
- Review generated hypotheses for domain relevance
- Validate research findings with independent data when possible

### Maintenance Integration

- Integrate predictive maintenance schedules with existing CMMS systems
- Establish clear escalation procedures for different risk levels
- Regularly update baseline health characteristics as equipment ages

### Monitoring and Validation

- Track research confidence trends over time
- Validate RUL predictions against actual equipment performance
- Monitor maintenance schedule effectiveness and adjust parameters as needed

## Support and Development

### Testing

Run comprehensive tests before deployment:

```bash
python test_research_system.py
```

### Contributing

When extending the system:

1. Follow existing agent architecture patterns
2. Implement proper error handling and logging
3. Add comprehensive tests for new functionality
4. Update documentation and configuration examples

### Performance Monitoring

Monitor system performance:

```python
# Check research progress
progress = research_state.calculate_research_progress()
print(f"Research progress: {progress:.1%}")

# Review agent performance
for entry in research_state.research_audit_trail:
    if entry.confidence < 0.5:
        print(f"Low confidence: {entry.agent} - {entry.action}")
```

This guide provides a comprehensive overview of the PHM Research Agent System. For specific implementation details, refer to the source code and additional documentation in the docs/ directory.
