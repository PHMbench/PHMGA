# PHMGA Agent System Documentation

This document provides comprehensive guidance for working with the PHMGA multi-agent system.

## Core Agent System

The PHMGA system features a sophisticated multi-agent architecture with 7+ specialized agents orchestrated through LangGraph workflows.

### 1. Plan Agent (`plan_agent.py`)

**Purpose:** Goal decomposition and high-level planning using LLM-powered analysis.

**API Reference:**
```python
def plan_agent(state: PHMState) -> dict:
    """
    Generate detailed processing plan using structured output.
    
    Parameters
    ----------
    state : PHMState
        Current system state with user_instruction, reflection_history, dag_state
        
    Returns
    -------
    dict
        Dictionary with 'detailed_plan' key containing list of Step objects
    """
```

**Data Structures:**
```python
class Step(BaseModel):
    parent: str = Field(..., description="Parent node ID")
    op_name: str = Field(..., description="Operator name from OP_REGISTRY")  
    params: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    plan: List[Step] = Field(..., description="Sequence of processing steps")
```

**Key Features:**
- LLM-based structured output generation using PLANNER_PROMPT
- Automatic operator discovery from OP_REGISTRY
- Sampling frequency injection for operators requiring fs parameter
- Error handling with graceful degradation

**Usage Example:**
```python
from src.agents.plan_agent import plan_agent
from src.states.phm_states import PHMState

# Agent automatically called in LangGraph workflow
result = plan_agent(state)
detailed_plan = result["detailed_plan"]
```

### 2. Execute Agent (`execute_agent.py`)

**Purpose:** Dynamic signal processing pipeline construction through operator application.

**API Reference:**
```python
def execute_agent(state: PHMState) -> Dict[str, Any]:
    """
    Execute signal processing operations based on detailed plan.
    
    Returns updated DAG state with new nodes and executed steps count.
    """
```

**Key Features:**
- Think-act loops with iterative operator application
- Support for both single-variable and multi-variable operators
- Automatic parameter resolution using LLM for missing parameters
- Result persistence to disk with case-specific organization
- Immutable DAG topology updates

**Operation Flow:**
1. Parse detailed plan from plan_agent
2. Iterate through processing steps
3. Apply operators to DAG nodes
4. Update DAG topology immutably
5. Persist results to save directory

### 3. Reflect Agent (`reflect_agent.py`)

**Purpose:** Quality assessment and revision decisions.

**API Reference:**
```python
def reflect_agent_node(state: PHMState, *, stage: str) -> Dict[str, Any]:
    """
    Quality check DAG and determine if revision needed.
    
    Returns needs_revision flag and updated reflection_history.
    """
```

**Decision Options:**
- `"finish"`: Analysis complete, proceed to reporting  
- `"need_patch"`: Minor fixes needed, continue execution
- `"need_replan"`: Major issues, restart planning
- `"halt"`: Critical errors, stop execution

**Reflection Criteria:**
- DAG depth and complexity adequacy
- Processing coverage of signal data
- Error analysis and recovery strategies
- User instruction fulfillment assessment

### 4. Report Agent (`report_agent.py`)

**Purpose:** Comprehensive result analysis and documentation generation.

**Report Sections Generated:**
1. **Executive Summary** - Key findings and recommendations
2. **Data Overview** - Signal characteristics and preprocessing steps  
3. **Signal Processing Pipeline** - DAG visualization and processing steps
4. **Similarity Analysis** - Comparative metrics between reference and test signals
5. **Machine Learning Results** - Model performance and classification results
6. **Conclusions** - Final diagnosis with confidence assessment
7. **Technical Details** - Parameters, intermediate results, error analysis

**Output Format:** Structured markdown report with embedded visualizations

### 5. Inquirer Agent (`inquirer_agent.py`)

**Purpose:** Similarity analysis between reference and test signals.

**API Reference:**
```python
def inquirer_agent(state: PHMState, metrics: List[str]) -> Dict[str, List[str]]:
    """
    Calculate similarity matrix between reference and test signals.
    
    Parameters
    ----------
    metrics : List[str]
        Similarity metrics: ["cosine", "euclidean", "pearson"]
    """
```

**Supported Metrics:**
- **Cosine similarity**: Angular similarity between feature vectors
- **Euclidean distance**: L2 distance in feature space
- **Pearson correlation**: Linear correlation coefficient

**Process:**
1. Extract features from processed DAG nodes
2. Compute similarity matrices for each metric
3. Store results in SimilarityNode structures
4. Generate comparative analysis summaries

### 6. Dataset Preparer Agent (`dataset_preparer_agent.py`)

**Purpose:** Feature extraction and dataset assembly for machine learning.

**API Reference:**
```python
def dataset_preparer_agent(state: PHMState) -> Dict:
    """
    Gather features and assemble datasets with labels.
    
    Returns datasets dict with X_train, X_test, y_train, y_test per node.
    """
```

**Dataset Assembly Process:**
1. Traverse DAG to identify feature-containing nodes
2. Extract numerical features from processed results
3. Align features with corresponding labels
4. Split into training/testing sets per node
5. Validate data consistency and completeness

**Output Structure:**
```python
datasets = {
    "node_id": {
        "X_train": np.ndarray,  # Reference features
        "X_test": np.ndarray,   # Test features  
        "y_train": List[str],   # Reference labels
        "y_test": List[str],    # Test labels (if available)
        "feature_names": List[str]
    }
}
```

### 7. Shallow ML Agent (`shallow_ml_agent.py`)

**Purpose:** Machine learning model training and ensemble inference.

**API Reference:**
```python
def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]], 
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train models per dataset and perform ensemble inference.
    
    Supports RandomForest and SVM with hard/soft voting ensemble.
    """
```

**Supported Algorithms:**
- **RandomForest**: Ensemble of decision trees with random feature selection
- **SVM**: Support Vector Machine with RBF kernel

**Ensemble Methods:**
- **Hard Voting**: Majority vote classification
- **Soft Voting**: Probability-weighted voting

**Training Process:**
1. Cross-validation for hyperparameter optimization
2. Model training per dataset node
3. Ensemble inference across all models
4. Performance evaluation and confidence assessment

## Research Agent System

### Advanced Multi-Agent Architecture

The system includes four specialized research agents for comprehensive analysis:

#### 1. Data Analyst Agent (`data_analyst_agent.py`)

**Capabilities:**
- Signal quality assessment and anomaly detection
- Statistical characterization and feature extraction  
- Feature space exploration using PCA and clustering
- Parallel processing with caching support
- Configurable analysis depth (quick mode vs comprehensive)

**API Reference:**
```python
from src.agents.data_analyst_agent import DataAnalystAgent

agent = DataAnalystAgent(config={
    "quick_mode": False,
    "use_parallel": True,
    "use_cache": True
})

results = agent.analyze_signal_quality(signals, labels)
```

**Analysis Types:**
- Signal-to-noise ratio estimation
- Distribution analysis and normality tests  
- Outlier detection using statistical methods
- Feature separability assessment

#### 2. Algorithm Researcher Agent (`algorithm_researcher_agent.py`)

**Capabilities:**
- Signal processing method comparison and optimization
- Machine learning algorithm evaluation
- Hyperparameter optimization with cross-validation
- Performance benchmarking and separability analysis

**Research Areas:**
- Time-frequency analysis method selection
- Feature extraction technique evaluation
- Classifier performance comparison
- Ensemble method optimization

#### 3. Domain Expert Agent (`domain_expert_agent.py`)

**Capabilities:**
- Physics-based validation and bearing analysis
- Failure mode identification and characterization
- Frequency analysis and bearing fault detection
- Domain knowledge integration

**Expertise Areas:**
- Bearing fault signatures and frequencies
- Vibration analysis and interpretation
- Maintenance strategy recommendations
- Root cause analysis

#### 4. Integration Agent (`integration_agent.py`)

**Capabilities:**
- Multi-agent result synthesis and conflict resolution
- Consensus building and confidence assessment
- Cross-agent consistency validation

**Integration Functions:**
- Result aggregation from multiple agents
- Conflict detection and resolution
- Confidence weighting and scoring
- Final recommendation synthesis

### Agent Factory Pattern (`agent_factory.py`)

**Design Pattern:** Factory pattern for flexible agent instantiation and configuration.

**Usage Examples:**
```python
from src.agents.agent_factory import AgentFactory, AgentType

# Method 1: Direct instantiation
agent = DataAnalystAgent(config={"quick_mode": False, "use_parallel": True})

# Method 2: Using Agent Factory
factory = AgentFactory()
agent = factory.create_agent_builder(AgentType.DATA_ANALYST) \
    .with_name("my_analyst") \
    .with_config(enable_advanced_features=True) \
    .build()

# Method 3: Using Presets  
agents = factory.create_agent_from_preset("QuickDiagnosis")
```

**Available Presets:**
- **QuickDiagnosis**: Fast analysis with essential agents
- **ComprehensiveAnalysis**: Full analysis with all research agents
- **ValidationOnly**: Focus on result validation and verification

**Agent Configuration Options:**
- Parallel processing enablement
- Caching strategies
- Analysis depth levels
- Performance monitoring
- Error handling policies

### Service Layer Architecture (`services.py`)

**Purpose:** Decoupled service layer for agent operations and shared functionality.

**Core Services:**
- **SignalProcessingService**: Common signal processing operations
- **FeatureExtractionService**: Standardized feature extraction
- **ValidationService**: Data and result validation
- **PersistenceService**: State and result persistence
- **VisualizationService**: Chart and graph generation

**Usage Pattern:**
```python
from src.agents.services import SignalProcessingService

service = SignalProcessingService()
processed_signals = service.apply_preprocessing_pipeline(
    signals=raw_signals,
    methods=["normalize", "filter", "denoise"]
)
```

## Agent Development Guidelines

### Custom Agent Development

**Base Class Pattern:**
```python
from src.agents.research_base import BaseResearchAgent

class CustomAgent(BaseResearchAgent):
    """Custom analysis agent following PHMGA patterns."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.agent_name = "custom_agent"
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Implement custom analysis logic."""
        
        # 1. Get configured LLM
        llm = self.get_llm()
        
        # 2. Perform analysis
        results = self._custom_analysis_logic(data)
        
        # 3. Return structured results
        return {
            "results": results,
            "confidence": self._calculate_confidence(results),
            "recommendations": self._generate_recommendations(results)
        }
```

**Integration with LangGraph:**
```python
def custom_agent_node(state: PHMState) -> Dict[str, Any]:
    """LangGraph node wrapper for custom agent."""
    
    agent = CustomAgent(config=state.get_agent_config())
    results = agent.analyze(state.get_current_data())
    
    # Return state updates
    return {
        "custom_results": results,
        "iteration_count": state.iteration_count + 1
    }
```

### Error Handling Patterns

**Robust State Updates:**
```python
def robust_agent(state: PHMState) -> Dict[str, Any]:
    """Agent with comprehensive error handling."""
    
    try:
        # Perform agent operations
        results = agent_operation(state)
        return results
        
    except LLMError as e:
        # LLM-specific error handling
        state.error_logs.append(f"LLM error in {agent_name}: {e}")
        return {"llm_error": True, "retry_needed": True}
        
    except ProcessingError as e:
        # Signal processing error handling
        state.dag_state.error_log.append(f"Processing error: {e}")
        return {"processing_error": True, "continue": True}
        
    except Exception as e:
        # General error handling
        state.error_logs.append(f"Unexpected error: {e}")
        return {"error_occurred": True, "halt_processing": True}
```

### Testing Patterns

**Mock LLM Testing:**
```python
from langchain_community.chat_models import FakeListChatModel

def test_plan_agent():
    """Test plan agent with predictable responses."""
    
    mock_responses = [
        '{"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]}'
    ]
    mock_llm = FakeListChatModel(responses=mock_responses)
    
    # Create test state
    test_state = create_test_state()
    
    # Test agent with mock LLM
    with patch('src.model.get_llm', return_value=mock_llm):
        result = plan_agent(test_state)
    
    # Validate results
    assert "detailed_plan" in result
    assert len(result["detailed_plan"]) == 1
    assert result["detailed_plan"][0]["op_name"] == "fft"
```

**Integration Testing:**
```python
def test_agent_integration():
    """Test multi-agent workflow integration."""
    
    # Initialize test state
    initial_state = initialize_test_state()
    
    # Run agent sequence
    plan_result = plan_agent(initial_state)
    updated_state = initial_state.model_copy(update=plan_result)
    
    execute_result = execute_agent(updated_state)
    final_state = updated_state.model_copy(update=execute_result)
    
    # Validate integration
    assert final_state.iteration_count > initial_state.iteration_count
    assert len(final_state.dag_state.nodes) > len(initial_state.dag_state.nodes)
```

### Performance Monitoring

**Agent Performance Tracking:**
```python
from src.agents.research_base import performance_monitor

class MonitoredAgent(BaseResearchAgent):
    
    @performance_monitor
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analysis with automatic performance monitoring."""
        # Analysis logic here
        return results
```

**Metrics Collected:**
- Execution time per agent
- Memory usage patterns
- LLM API call frequency
- Error rates and recovery success
- DAG construction efficiency

This comprehensive agent system provides the foundation for flexible, scalable, and maintainable signal processing workflows in the PHMGA framework.