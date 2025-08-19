# PHMGA Schemas Module

The schemas module defines structured data models for the PHMGA system, ensuring type safety and consistent data exchange between components.

## Overview

The schemas provide:
- **Type Safety**: Pydantic-based validation for all data structures
- **Structured Outputs**: Consistent formats for LLM responses
- **Data Validation**: Automatic validation of inputs and outputs
- **Documentation**: Self-documenting data structures

## Core Schemas

### 1. Insight Schema (`insight_schema.py`)

#### AnalysisInsight

```python
class AnalysisInsight(BaseModel):
    """Structured representation of analysis insights."""
    
    insight_id: str = Field(default_factory=lambda: f"insight_{uuid.uuid4().hex[:8]}")
    category: Literal["signal_quality", "pattern_detection", "anomaly", "classification"] = Field(
        ..., description="Type of insight"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    description: str = Field(..., description="Human-readable insight description")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    timestamp: datetime = Field(default_factory=datetime.now)
```

#### Usage Example

```python
from src.schemas.insight_schema import AnalysisInsight

# Create insight from analysis
insight = AnalysisInsight(
    category="anomaly",
    confidence=0.85,
    description="Detected bearing fault in test signal #3",
    evidence={
        "rms_ratio": 2.3,
        "kurtosis_value": 4.7,
        "frequency_peaks": [150, 300, 450]
    },
    recommendations=[
        "Schedule immediate maintenance",
        "Monitor vibration levels",
        "Replace bearing within 100 hours"
    ]
)

print(f"Insight: {insight.description}")
print(f"Confidence: {insight.confidence:.2%}")
```

### 2. Plan Schema (`plan_schema.py`)

#### Step

```python
class Step(BaseModel):
    """A single step in the processing plan."""
    
    parent: str = Field(..., description="The ID of the parent node")
    op_name: str = Field(..., description="The name of the operator to use")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operator parameters")
    priority: int = Field(1, description="Execution priority (1=highest)")
    dependencies: List[str] = Field(default_factory=list, description="Required predecessor steps")
```

#### AnalysisPlan

```python
class AnalysisPlan(BaseModel):
    """The complete processing plan."""
    
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan: List[Step] = Field(..., description="A list of processing steps")
    estimated_duration: Optional[float] = Field(None, description="Estimated execution time in seconds")
    complexity_score: Optional[float] = Field(None, description="Plan complexity (0-1)")
    created_at: datetime = Field(default_factory=datetime.now)
```

#### Usage Example

```python
from src.schemas.plan_schema import AnalysisPlan, Step

# Create structured plan
plan = AnalysisPlan(
    plan=[
        Step(
            parent="ch1",
            op_name="fft",
            params={"axis": -2},
            priority=1
        ),
        Step(
            parent="fft_ch1",
            op_name="mean",
            params={"axis": -2},
            priority=2,
            dependencies=["fft_ch1"]
        )
    ],
    estimated_duration=30.0,
    complexity_score=0.6
)

# Validate and use
print(f"Plan has {len(plan.plan)} steps")
for step in plan.plan:
    print(f"  {step.op_name} on {step.parent}")
```

## Research Schemas

### Search and Query Schemas

```python
class SearchQuery(BaseModel):
    """Individual search query."""
    
    query: str = Field(..., description="Search query text")
    rationale: str = Field(..., description="Reasoning for this query")
    priority: int = Field(1, description="Query priority")

class SearchQueryList(BaseModel):
    """List of search queries."""
    
    query: List[SearchQuery] = Field(..., description="List of search queries")

class Reflection(BaseModel):
    """Reflection on research completeness."""
    
    is_sufficient: bool = Field(..., description="Whether research is sufficient")
    knowledge_gap: str = Field(..., description="Identified knowledge gaps")
    follow_up_queries: List[SearchQuery] = Field(
        default_factory=list,
        description="Follow-up queries to address gaps"
    )
```

## Validation and Type Safety

### Automatic Validation

```python
from pydantic import ValidationError

try:
    # Invalid confidence score
    insight = AnalysisInsight(
        category="anomaly",
        confidence=1.5,  # Invalid: > 1.0
        description="Test insight"
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Custom Validators

```python
from pydantic import validator

class EnhancedInsight(AnalysisInsight):
    """Enhanced insight with custom validation."""
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0.1:
            raise ValueError('Confidence must be at least 0.1')
        return v
    
    @validator('evidence')
    def validate_evidence(cls, v):
        if not v:
            raise ValueError('Evidence cannot be empty')
        return v
```

## Integration Patterns

### LLM Structured Output

```python
from src.model import get_llm
from src.schemas.plan_schema import AnalysisPlan

# Use schema for structured LLM output
llm = get_llm()
structured_llm = llm.with_structured_output(AnalysisPlan)

# LLM will return validated AnalysisPlan object
plan = structured_llm.invoke("Create a signal processing plan")
assert isinstance(plan, AnalysisPlan)
```

### Agent Communication

```python
def plan_agent(state: PHMState) -> Dict[str, Any]:
    """Plan agent using structured schemas."""
    
    # Generate plan using schema
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnalysisPlan)
    
    plan = structured_llm.invoke(planning_prompt)
    
    # Convert to dict for state update
    return {"detailed_plan": [step.dict() for step in plan.plan]}
```

### Data Serialization

```python
# JSON serialization
insight_json = insight.json()
plan_json = plan.json()

# Dictionary conversion
insight_dict = insight.dict()
plan_dict = plan.dict()

# Reconstruction
loaded_insight = AnalysisInsight.parse_raw(insight_json)
loaded_plan = AnalysisPlan.parse_raw(plan_json)
```

## Advanced Features

### Schema Evolution

```python
class AnalysisInsightV2(AnalysisInsight):
    """Enhanced insight schema with additional fields."""
    
    version: str = Field("2.0", description="Schema version")
    source_agent: str = Field(..., description="Agent that generated insight")
    related_insights: List[str] = Field(default_factory=list, description="Related insight IDs")
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"
```

### Dynamic Schema Generation

```python
def create_dynamic_schema(fields: Dict[str, Any]) -> Type[BaseModel]:
    """Create schema dynamically based on requirements."""
    
    return create_model(
        'DynamicSchema',
        **{name: (field_type, Field(...)) for name, field_type in fields.items()}
    )

# Usage
custom_schema = create_dynamic_schema({
    'signal_id': str,
    'amplitude': float,
    'frequency': float
})
```

### Schema Composition

```python
class ComprehensiveAnalysis(BaseModel):
    """Composite schema combining multiple analysis aspects."""
    
    insights: List[AnalysisInsight] = Field(default_factory=list)
    plan: AnalysisPlan
    execution_status: Literal["pending", "running", "completed", "failed"] = "pending"
    results: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def high_confidence_insights(self) -> List[AnalysisInsight]:
        """Get insights with confidence > 0.8."""
        return [insight for insight in self.insights if insight.confidence > 0.8]
```

## Testing and Validation

### Schema Testing

```python
def test_schema_validation():
    """Test schema validation behavior."""
    
    # Valid insight
    valid_insight = AnalysisInsight(
        category="anomaly",
        confidence=0.9,
        description="Test anomaly detection"
    )
    assert valid_insight.confidence == 0.9
    
    # Invalid insight
    with pytest.raises(ValidationError):
        AnalysisInsight(
            category="invalid_category",  # Not in allowed values
            confidence=0.9,
            description="Test"
        )
```

### Mock Data Generation

```python
def generate_mock_insight() -> AnalysisInsight:
    """Generate mock insight for testing."""
    
    return AnalysisInsight(
        category="pattern_detection",
        confidence=0.75,
        description="Mock pattern detected in test data",
        evidence={"pattern_strength": 0.8, "frequency": 150},
        recommendations=["Further analysis recommended"]
    )

def generate_mock_plan() -> AnalysisPlan:
    """Generate mock plan for testing."""
    
    return AnalysisPlan(
        plan=[
            Step(parent="ch1", op_name="fft", params={}),
            Step(parent="fft_ch1", op_name="mean", params={"axis": -2})
        ],
        estimated_duration=25.0
    )
```

## Dependencies

### Required Packages

```python
# Core dependencies
pydantic>=2.0.0
typing-extensions
datetime  # Built-in
uuid      # Built-in

# Optional for advanced features
python-dateutil>=2.8.0  # Enhanced datetime handling
```

### Installation

```bash
# Install core dependencies
pip install pydantic typing-extensions

# Install optional dependencies
pip install python-dateutil
```

This schema system provides robust, type-safe data structures that ensure consistency and reliability throughout the PHMGA analysis pipeline.
