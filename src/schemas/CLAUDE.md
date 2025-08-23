# PHMGA Schemas Documentation

This document provides comprehensive guidance for working with the PHMGA data schemas.

## Overview

The schemas provide:
- **Type Safety**: Pydantic-based validation for all data structures
- **Structured Outputs**: Consistent formats for LLM responses
- **Data Validation**: Automatic validation of inputs and outputs
- **Documentation**: Self-documenting data structures

## Core Schemas

### AnalysisInsight Schema

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

**Usage Example:**
```python
from src.schemas.insight_schema import AnalysisInsight

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
        "Monitor vibration levels"
    ]
)
```

### Plan Schema

```python
class Step(BaseModel):
    """A single step in the processing plan."""
    
    parent: str = Field(..., description="The ID of the parent node")
    op_name: str = Field(..., description="The name of the operator to use")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operator parameters")
    priority: int = Field(1, description="Execution priority (1=highest)")
    dependencies: List[str] = Field(default_factory=list, description="Required predecessor steps")

class AnalysisPlan(BaseModel):
    """The complete processing plan."""
    
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan: List[Step] = Field(..., description="A list of processing steps")
    estimated_duration: Optional[float] = Field(None, description="Estimated execution time in seconds")
    complexity_score: Optional[float] = Field(None, description="Plan complexity (0-1)")
    created_at: datetime = Field(default_factory=datetime.now)
```

**Usage Example:**
```python
from src.schemas.plan_schema import AnalysisPlan, Step

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
            priority=2
        )
    ],
    estimated_duration=30.0,
    complexity_score=0.6
)
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
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnalysisPlan)
    
    plan = structured_llm.invoke(planning_prompt)
    
    # Convert to dict for state update
    return {"detailed_plan": [step.dict() for step in plan.plan]}
```

### Validation

```python
from pydantic import ValidationError

try:
    insight = AnalysisInsight(
        category="invalid_category",  # Not in allowed values
        confidence=1.5,  # Invalid: > 1.0
        description="Test insight"
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

This schema system provides robust, type-safe data structures that ensure consistency and reliability throughout the PHMGA analysis pipeline.