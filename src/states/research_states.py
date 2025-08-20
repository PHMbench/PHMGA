"""
Research-oriented state management for PHM scientist agent system.

This module extends the base PHMState with research-specific capabilities,
agent coordination states, and advanced analysis tracking.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from .phm_states import PHMState, InputData, DAGState


# Use regular Dict types instead of TypedDict to avoid compatibility issues
DataAnalysisState = Dict[str, Any]
AlgorithmResearchState = Dict[str, Any]
DomainExpertState = Dict[str, Any]
IntegrationState = Dict[str, Any]


class ResearchHypothesis(BaseModel):
    """Structured representation of a research hypothesis."""
    
    hypothesis_id: str = Field(..., description="Unique identifier for the hypothesis")
    statement: str = Field(..., description="Clear statement of the hypothesis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    testable: bool = Field(default=True, description="Whether hypothesis can be tested")
    test_methods: List[str] = Field(default_factory=list, description="Proposed testing methods")
    validation_status: Optional[str] = Field(default=None, description="Validation result")
    generated_by: str = Field(..., description="Agent that generated the hypothesis")


class ResearchObjective(BaseModel):
    """Structured representation of a research objective."""
    
    objective_id: str = Field(..., description="Unique identifier for the objective")
    description: str = Field(..., description="Clear description of the objective")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1=highest, 5=lowest)")
    status: str = Field(default="pending", description="Current status")
    assigned_agents: List[str] = Field(default_factory=list, description="Agents working on this objective")
    completion_criteria: List[str] = Field(default_factory=list, description="Criteria for completion")
    results: Dict[str, Any] = Field(default_factory=dict, description="Objective results")


class ResearchAuditEntry(BaseModel):
    """Entry in the research audit trail."""
    
    timestamp: str = Field(..., description="ISO timestamp of the entry")
    agent: str = Field(..., description="Agent that performed the action")
    action: str = Field(..., description="Action performed")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Action inputs")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Action outputs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the action")
    validation_status: Optional[str] = Field(default=None, description="Validation result")


class ResearchPHMState(PHMState):
    """
    Extended PHM state for research-oriented workflows.
    
    This class extends the base PHMState with research-specific capabilities,
    multi-agent coordination, and advanced analysis tracking.
    """
    
    # Research workflow state
    research_phase: str = Field(default="initialization", description="Current research phase")
    research_objectives: List[ResearchObjective] = Field(
        default_factory=list, description="Structured research objectives"
    )
    research_hypotheses: List[ResearchHypothesis] = Field(
        default_factory=list, description="Generated and tested hypotheses"
    )
    
    # Agent-specific states
    data_analysis_state: DataAnalysisState = Field(
        default_factory=dict, description="Data Analyst Agent state"
    )
    algorithm_research_state: AlgorithmResearchState = Field(
        default_factory=dict, description="Algorithm Researcher Agent state"
    )
    domain_expert_state: DomainExpertState = Field(
        default_factory=dict, description="Domain Expert Agent state"
    )
    integration_state: IntegrationState = Field(
        default_factory=dict, description="Integration Agent state"
    )
    
    # Research quality and validation
    research_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall research confidence"
    )
    validation_results: Dict[str, Any] = Field(
        default_factory=dict, description="Validation test results"
    )
    research_audit_trail: List[ResearchAuditEntry] = Field(
        default_factory=list, description="Complete research audit trail"
    )
    
    # Advanced analysis results
    remaining_useful_life: Optional[float] = Field(
        default=None, description="Estimated remaining useful life"
    )
    health_index: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Overall health score (0=failed, 1=healthy)"
    )
    predictive_maintenance_schedule: Optional[Dict[str, Any]] = Field(
        default=None, description="Recommended maintenance schedule"
    )
    
    # Research configuration
    max_research_iterations: int = Field(
        default=5, description="Maximum number of research iterations"
    )
    research_quality_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum quality threshold for completion"
    )
    parallel_agent_execution: bool = Field(
        default=True, description="Whether to execute agents in parallel"
    )
    
    # Literature and knowledge integration
    literature_references: List[Dict[str, str]] = Field(
        default_factory=list, description="Relevant literature references"
    )
    knowledge_base_entries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relevant knowledge base entries"
    )
    
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
    
    def add_research_objective(self, description: str, priority: int = 3) -> str:
        """Add a new research objective."""
        import uuid
        objective_id = f"obj_{uuid.uuid4().hex[:8]}"
        objective = ResearchObjective(
            objective_id=objective_id,
            description=description,
            priority=priority
        )
        self.research_objectives.append(objective)
        return objective_id
    
    def add_hypothesis(self, statement: str, confidence: float, generated_by: str) -> str:
        """Add a new research hypothesis."""
        import uuid
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            statement=statement,
            confidence=confidence,
            generated_by=generated_by
        )
        self.research_hypotheses.append(hypothesis)
        return hypothesis_id
    
    def add_audit_entry(self, agent: str, action: str, confidence: float, 
                       inputs: Dict[str, Any] = None, outputs: Dict[str, Any] = None) -> None:
        """Add an entry to the research audit trail."""
        from datetime import datetime
        entry = ResearchAuditEntry(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            action=action,
            confidence=confidence,
            inputs=inputs or {},
            outputs=outputs or {}
        )
        self.research_audit_trail.append(entry)
    
    def get_active_objectives(self) -> List[ResearchObjective]:
        """Get all active (non-completed) research objectives."""
        return [obj for obj in self.research_objectives if obj.status != "completed"]
    
    def get_validated_hypotheses(self) -> List[ResearchHypothesis]:
        """Get all validated hypotheses."""
        return [hyp for hyp in self.research_hypotheses if hyp.validation_status == "validated"]
    
    def calculate_research_progress(self) -> float:
        """Calculate overall research progress as a percentage."""
        if not self.research_objectives:
            return 0.0
        
        completed = len([obj for obj in self.research_objectives if obj.status == "completed"])
        return completed / len(self.research_objectives)


if __name__ == "__main__":
    # Test the research state functionality
    from datetime import datetime
    
    # Create a minimal DAG state for testing
    dag_state = DAGState(
        user_instruction="Test research state",
        channels=["ch1"],
        nodes={},
        leaves=[]
    )
    
    # Create minimal input data for testing
    ref_signal = InputData(
        node_id="ref_ch1",
        parents=[],
        shape=(1, 1024, 1),
        results={"ref": np.random.randn(1, 1024, 1)},
        meta={"fs": 1000}
    )
    
    test_signal = InputData(
        node_id="test_ch1", 
        parents=[],
        shape=(1, 1024, 1),
        results={"test": np.random.randn(1, 1024, 1)},
        meta={"fs": 1000}
    )
    
    # Create research state
    state = ResearchPHMState(
        case_name="test_research",
        user_instruction="Test research capabilities",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state
    )
    
    # Test functionality
    obj_id = state.add_research_objective("Analyze signal characteristics", priority=1)
    hyp_id = state.add_hypothesis("Signal contains bearing fault signatures", 0.8, "data_analyst")
    state.add_audit_entry("data_analyst", "signal_analysis", 0.9)
    
    print(f"Created research state with {len(state.research_objectives)} objectives")
    print(f"Research progress: {state.calculate_research_progress():.1%}")
    print("Research state test completed successfully!")
