# PHM Research Agent Architecture Design

## Executive Summary

This document outlines the transformation of the existing bearing fault diagnosis system (Case 1) into an advanced PHM scientist agent system (Case 2) by implementing research-oriented multi-agent capabilities based on LangGraph patterns.

## Current System Analysis

### Existing Architecture (Case 1)
- **Static Workflow**: Plan → Execute → Reflect → Report
- **Core Agents**: 
  - Plan Agent: Goal decomposition and processing plan generation
  - Execute Agent: Signal processing DAG execution
  - Reflect Agent: Quality assessment and revision decisions
  - Report Agent: Final analysis documentation
- **Capabilities**: Basic bearing fault classification using predefined signal processing pipelines
- **Limitations**: Reactive diagnosis only, limited research autonomy, no hypothesis generation

### Reference LangGraph Patterns
- **Multi-agent coordination**: Parallel execution with Send() for concurrent research tasks
- **Conditional routing**: Dynamic workflow adaptation based on intermediate findings
- **State management**: Centralized state with specialized sub-states for different agent types
- **Reflection loops**: Iterative refinement with configurable termination conditions

## Research Agent Architecture

### Core Research Agents

#### 1. Data Analyst Agent (`data_analyst_agent.py`)
**Purpose**: Exploratory data analysis and data quality assessment

**Responsibilities**:
- Automated signal quality assessment and anomaly detection
- Statistical characterization of signal properties (stationarity, noise levels, frequency content)
- Feature space exploration using dimensionality reduction techniques
- Data preprocessing recommendations based on signal characteristics
- Uncertainty quantification for measurement quality

**State Interface**:
```python
class DataAnalysisState(TypedDict):
    signal_quality_metrics: Dict[str, float]
    statistical_summary: Dict[str, Any]
    preprocessing_recommendations: List[str]
    feature_space_analysis: Dict[str, np.ndarray]
    uncertainty_estimates: Dict[str, float]
```

#### 2. Algorithm Researcher Agent (`algorithm_researcher_agent.py`)
**Purpose**: Investigation and comparison of signal processing and ML techniques

**Responsibilities**:
- Automated hyperparameter optimization for signal processing pipelines
- Comparative analysis of different feature extraction methods
- Algorithm performance benchmarking and statistical validation
- Novel signal processing technique discovery through literature integration
- Adaptive algorithm selection based on signal characteristics

**State Interface**:
```python
class AlgorithmResearchState(TypedDict):
    algorithm_comparisons: Dict[str, Dict[str, float]]
    hyperparameter_results: Dict[str, Any]
    performance_benchmarks: Dict[str, float]
    recommended_algorithms: List[str]
    research_insights: List[str]
```

#### 3. Domain Expert Agent (`domain_expert_agent.py`)
**Purpose**: PHM domain knowledge integration and validation

**Responsibilities**:
- Physics-informed analysis validation using bearing fault mechanics
- Domain-specific feature interpretation and relevance assessment
- Failure mode identification and progression analysis
- Research direction suggestions based on PHM best practices
- Knowledge base integration for bearing fault diagnosis

**State Interface**:
```python
class DomainExpertState(TypedDict):
    physics_validation: Dict[str, bool]
    failure_mode_analysis: Dict[str, Any]
    domain_insights: List[str]
    research_directions: List[str]
    knowledge_base_matches: List[Dict[str, Any]]
```

#### 4. Integration Agent (`integration_agent.py`)
**Purpose**: Coordination between agents and synthesis of research findings

**Responsibilities**:
- Multi-agent workflow orchestration and task scheduling
- Research finding synthesis and conflict resolution
- Hypothesis generation based on integrated agent outputs
- Research quality assessment and validation
- Final research report compilation

**State Interface**:
```python
class IntegrationState(TypedDict):
    agent_coordination: Dict[str, str]
    synthesized_findings: Dict[str, Any]
    generated_hypotheses: List[str]
    research_quality_score: float
    integration_conflicts: List[str]
```

### Enhanced PHM State

```python
class ResearchPHMState(PHMState):
    """Extended PHM state for research-oriented workflows."""
    
    # Research workflow state
    research_phase: str = Field(default="initialization", description="Current research phase")
    research_objectives: List[str] = Field(default_factory=list, description="Research goals")
    research_hypotheses: List[str] = Field(default_factory=list, description="Generated hypotheses")
    
    # Agent-specific states
    data_analysis_state: DataAnalysisState = Field(default_factory=dict)
    algorithm_research_state: AlgorithmResearchState = Field(default_factory=dict)
    domain_expert_state: DomainExpertState = Field(default_factory=dict)
    integration_state: IntegrationState = Field(default_factory=dict)
    
    # Research quality metrics
    research_confidence: float = Field(default=0.0, description="Overall research confidence")
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    research_audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Advanced analysis results
    remaining_useful_life: Optional[float] = Field(default=None, description="RUL estimation")
    health_index: Optional[float] = Field(default=None, description="Overall health score")
    predictive_maintenance_schedule: Optional[Dict[str, Any]] = Field(default=None)
```

## Research Workflow Design

### Dynamic Research Graph Structure

```python
def build_research_graph() -> StateGraph:
    """Build the research-oriented LangGraph workflow."""
    
    builder = StateGraph(ResearchPHMState)
    
    # Core research nodes
    builder.add_node("initialize_research", initialize_research_objectives)
    builder.add_node("data_analysis", data_analyst_agent)
    builder.add_node("algorithm_research", algorithm_researcher_agent)
    builder.add_node("domain_expert", domain_expert_agent)
    builder.add_node("integration", integration_agent)
    builder.add_node("hypothesis_generation", generate_research_hypotheses)
    builder.add_node("validation", validate_research_findings)
    builder.add_node("research_report", generate_research_report)
    
    # Workflow edges
    builder.add_edge(START, "initialize_research")
    builder.add_conditional_edges(
        "initialize_research",
        route_to_parallel_research,
        ["data_analysis", "algorithm_research", "domain_expert"]
    )
    builder.add_edge(["data_analysis", "algorithm_research", "domain_expert"], "integration")
    builder.add_edge("integration", "hypothesis_generation")
    builder.add_conditional_edges(
        "hypothesis_generation",
        evaluate_research_progress,
        ["validation", "data_analysis", "research_report"]
    )
    builder.add_edge("validation", "research_report")
    builder.add_edge("research_report", END)
    
    return builder.compile()
```

### Conditional Routing Logic

```python
def route_to_parallel_research(state: ResearchPHMState) -> List[Send]:
    """Route to parallel research agents based on research objectives."""
    
    routes = []
    
    # Always include data analysis
    routes.append(Send("data_analysis", {"research_focus": "signal_characterization"}))
    
    # Conditional algorithm research based on signal complexity
    if state.data_analysis_state.get("signal_complexity", 0) > 0.5:
        routes.append(Send("algorithm_research", {"research_focus": "advanced_methods"}))
    else:
        routes.append(Send("algorithm_research", {"research_focus": "standard_methods"}))
    
    # Domain expert analysis for all cases
    routes.append(Send("domain_expert", {"research_focus": "bearing_mechanics"}))
    
    return routes

def evaluate_research_progress(state: ResearchPHMState) -> str:
    """Determine next research action based on current findings."""
    
    confidence = state.research_confidence
    iteration_count = state.iteration_count
    max_iterations = state.max_research_iterations
    
    if confidence > 0.8 or iteration_count >= max_iterations:
        return "research_report"
    elif state.integration_state.get("needs_validation", False):
        return "validation"
    else:
        return "data_analysis"  # Continue research loop
```

## Implementation Strategy

### Phase 1: Core Agent Development
1. Implement base research agent classes with common interfaces
2. Develop Data Analyst Agent with signal quality assessment
3. Create Algorithm Researcher Agent with comparative analysis
4. Build Domain Expert Agent with PHM knowledge integration
5. Implement Integration Agent for workflow coordination

### Phase 2: Research Workflow Integration
1. Extend PHMState with research-specific fields
2. Implement dynamic research graph with conditional routing
3. Add parallel agent execution capabilities
4. Create research quality assessment metrics
5. Develop hypothesis generation and validation logic

### Phase 3: Advanced Research Capabilities
1. Integrate automated literature review capabilities
2. Add uncertainty quantification throughout the pipeline
3. Implement adaptive experiment design
4. Create research audit trail and reproducibility features
5. Add predictive maintenance scheduling capabilities

## Success Metrics

### Research Quality Indicators
- **Hypothesis Generation Rate**: Number of testable hypotheses per analysis
- **Validation Accuracy**: Percentage of hypotheses validated through testing
- **Research Confidence**: Quantitative measure of finding reliability
- **Discovery Rate**: Novel insights beyond traditional diagnostic approaches

### Performance Improvements
- **Diagnostic Accuracy**: Improvement over baseline Case 1 performance
- **Analysis Depth**: Comprehensive coverage of signal processing techniques
- **Automation Level**: Reduction in manual intervention requirements
- **Research Reproducibility**: Consistency of findings across repeated analyses

This architecture provides a foundation for transforming reactive fault diagnosis into proactive research-driven PHM analysis while maintaining compatibility with existing system components.

## Usage Examples

### Running Enhanced Case 1

```python
from src.cases.case1_enhanced import run_enhanced_case

# Run with research agents enabled
result = run_enhanced_case("config/case1.yaml", enable_research=True)

# Run traditional workflow only
result = run_enhanced_case("config/case1.yaml", enable_research=False)

# Run research agents only (skip traditional DAG building)
result = run_enhanced_case("config/case1.yaml", enable_research=True, research_only=True)
```

### Running Case 2 Predictive Maintenance

```python
from src.cases.case2_predictive import run_predictive_maintenance_case

# Run complete predictive maintenance analysis
result = run_predictive_maintenance_case("config/case2.yaml")

if result:
    print(f"Health Index: {result.health_index:.3f}")
    print(f"RUL: {result.remaining_useful_life:.1f} days")
    print(f"Maintenance Strategy: {result.predictive_maintenance_schedule['strategy']}")
```

### Using Individual Research Agents

```python
from src.agents.data_analyst_agent import DataAnalystAgent
from src.agents.algorithm_researcher_agent import AlgorithmResearcherAgent
from src.agents.domain_expert_agent import DomainExpertAgent
from src.agents.integration_agent import IntegrationAgent

# Create agents
data_agent = DataAnalystAgent()
algo_agent = AlgorithmResearcherAgent()
domain_agent = DomainExpertAgent()
integration_agent = IntegrationAgent()

# Run analysis
data_results = data_agent.analyze(research_state)
algo_results = algo_agent.analyze(research_state)
domain_results = domain_agent.analyze(research_state)
integration_results = integration_agent.analyze(research_state)
```

## Testing and Validation

Run the comprehensive test suite to validate the system:

```bash
python test_research_system.py
```

This will test all components including:
- Research state management
- Individual agent functionality
- Research workflow orchestration
- Case study implementations
- System integration
