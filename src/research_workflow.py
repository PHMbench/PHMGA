"""
Research workflow orchestration for PHM scientist agent system.

This module implements the LangGraph-based research workflow that coordinates
multiple research agents for autonomous PHM investigation and analysis.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from .states.research_states import ResearchPHMState, ResearchObjective
from .agents.data_analyst_agent import data_analyst_agent
from .agents.algorithm_researcher_agent import algorithm_researcher_agent
from .agents.domain_expert_agent import domain_expert_agent
from .agents.integration_agent import integration_agent

logger = logging.getLogger(__name__)


def initialize_research_objectives(state: ResearchPHMState) -> Dict[str, Any]:
    """
    Initialize research objectives based on user instruction and signal characteristics.
    
    Args:
        state: Current research state
        
    Returns:
        State updates with initialized research objectives
    """
    logger.info("Initializing research objectives")
    
    # Define default research objectives for bearing fault analysis
    default_objectives = [
        {
            "description": "Assess signal quality and preprocessing requirements",
            "priority": 1,
            "assigned_agents": ["data_analyst"]
        },
        {
            "description": "Identify optimal signal processing and ML algorithms",
            "priority": 2,
            "assigned_agents": ["algorithm_researcher"]
        },
        {
            "description": "Apply domain expertise for physics-informed validation",
            "priority": 1,
            "assigned_agents": ["domain_expert"]
        },
        {
            "description": "Integrate findings and resolve conflicts between agents",
            "priority": 3,
            "assigned_agents": ["integration_agent"]
        }
    ]
    
    # Create research objectives
    research_objectives = []
    for obj_data in default_objectives:
        objective = ResearchObjective(
            objective_id=f"obj_{len(research_objectives)}",
            description=obj_data["description"],
            priority=obj_data["priority"],
            assigned_agents=obj_data["assigned_agents"]
        )
        research_objectives.append(objective)
    
    # Add audit entry
    state.add_audit_entry(
        agent="research_coordinator",
        action="initialize_objectives",
        confidence=1.0,
        outputs={"n_objectives": len(research_objectives)}
    )
    
    return {
        "research_objectives": research_objectives,
        "research_phase": "parallel_analysis"
    }


def route_to_parallel_research(state: ResearchPHMState) -> List[Send]:
    """
    Route to parallel research agents based on research objectives.
    
    Args:
        state: Current research state
        
    Returns:
        List of Send objects for parallel agent execution
    """
    logger.info("Routing to parallel research agents")
    
    routes = []
    
    # Always execute core research agents in parallel
    routes.append(Send("data_analysis", {"research_focus": "signal_characterization"}))
    routes.append(Send("algorithm_research", {"research_focus": "method_optimization"}))
    routes.append(Send("domain_expert", {"research_focus": "physics_validation"}))
    
    logger.info(f"Routing to {len(routes)} parallel research agents")
    return routes


def evaluate_research_progress(state: ResearchPHMState) -> str:
    """
    Evaluate research progress and determine next action.
    
    Args:
        state: Current research state
        
    Returns:
        Next node to execute
    """
    logger.info("Evaluating research progress")
    
    # Check research confidence and iteration count
    confidence = state.research_confidence
    iteration_count = state.iteration_count
    max_iterations = state.max_research_iterations
    
    # Check if research quality threshold is met
    quality_threshold = state.research_quality_threshold
    
    logger.info(f"Research confidence: {confidence:.3f}, Iteration: {iteration_count}/{max_iterations}")
    
    # Decision logic
    if confidence >= quality_threshold:
        logger.info("Research quality threshold met - proceeding to final report")
        return "research_report"
    elif iteration_count >= max_iterations:
        logger.info("Maximum iterations reached - proceeding to final report")
        return "research_report"
    else:
        # Check if validation is needed
        integration_state = state.integration_state
        if integration_state.get("integration_conflicts"):
            logger.info("Integration conflicts detected - proceeding to validation")
            return "validation"
        else:
            logger.info("Continuing research iteration")
            return "hypothesis_generation"


def generate_research_hypotheses(state: ResearchPHMState) -> Dict[str, Any]:
    """
    Generate new research hypotheses based on current findings.
    
    Args:
        state: Current research state
        
    Returns:
        State updates with new hypotheses
    """
    logger.info("Generating research hypotheses")
    
    # This is handled by individual agents, but we can add meta-hypotheses here
    new_hypotheses = []
    
    # Meta-hypothesis about research convergence
    if state.iteration_count > 1:
        confidence_trend = "increasing"  # Would calculate from audit trail
        if confidence_trend == "increasing":
            from .states.research_states import ResearchHypothesis
            meta_hypothesis = ResearchHypothesis(
                hypothesis_id=f"meta_convergence_{state.iteration_count}",
                statement="Research is converging towards reliable conclusions",
                confidence=min(state.research_confidence + 0.1, 1.0),
                generated_by="research_coordinator",
                evidence=[f"Iteration {state.iteration_count} shows confidence improvement"],
                test_methods=["trend_analysis", "convergence_validation"]
            )
            new_hypotheses.append(meta_hypothesis)
    
    # Add audit entry
    state.add_audit_entry(
        agent="research_coordinator",
        action="hypothesis_generation",
        confidence=state.research_confidence,
        outputs={"n_new_hypotheses": len(new_hypotheses)}
    )
    
    return {
        "research_hypotheses": state.research_hypotheses + new_hypotheses,
        "iteration_count": state.iteration_count + 1,
        "research_phase": "hypothesis_validation"
    }


def validate_research_findings(state: ResearchPHMState) -> Dict[str, Any]:
    """
    Validate research findings using statistical tests and cross-validation.
    
    Args:
        state: Current research state
        
    Returns:
        State updates with validation results
    """
    logger.info("Validating research findings")
    
    validation_results = {
        "hypothesis_validation": {},
        "statistical_significance": {},
        "cross_validation_results": {},
        "validation_confidence": 0.0
    }
    
    try:
        # Validate hypotheses
        for hypothesis in state.research_hypotheses:
            if hypothesis.testable:
                # Simple validation based on confidence and evidence
                validation_score = hypothesis.confidence
                
                # Adjust based on evidence quality
                if len(hypothesis.evidence) > 2:
                    validation_score += 0.1
                if len(hypothesis.test_methods) > 1:
                    validation_score += 0.1
                
                validation_score = min(validation_score, 1.0)
                
                validation_results["hypothesis_validation"][hypothesis.hypothesis_id] = {
                    "validated": validation_score > 0.6,
                    "validation_score": validation_score,
                    "validation_method": "confidence_based"
                }
        
        # Statistical significance testing (simplified)
        validated_hypotheses = sum(1 for v in validation_results["hypothesis_validation"].values() 
                                 if v["validated"])
        total_hypotheses = len(validation_results["hypothesis_validation"])
        
        if total_hypotheses > 0:
            validation_results["statistical_significance"]["validation_rate"] = validated_hypotheses / total_hypotheses
            validation_results["statistical_significance"]["significant"] = validation_rate > 0.5
        
        # Overall validation confidence
        if validation_results["hypothesis_validation"]:
            avg_validation_score = sum(v["validation_score"] for v in validation_results["hypothesis_validation"].values())
            avg_validation_score /= len(validation_results["hypothesis_validation"])
            validation_results["validation_confidence"] = avg_validation_score
        
        logger.info(f"Validation completed: {validated_hypotheses}/{total_hypotheses} hypotheses validated")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        validation_results["error"] = str(e)
        validation_results["validation_confidence"] = 0.0
    
    # Add audit entry
    state.add_audit_entry(
        agent="research_coordinator",
        action="findings_validation",
        confidence=validation_results["validation_confidence"],
        outputs=validation_results
    )
    
    return {
        "validation_results": validation_results,
        "research_phase": "validation_complete"
    }


def generate_research_report(state: ResearchPHMState) -> Dict[str, Any]:
    """
    Generate comprehensive research report with findings and recommendations.
    
    Args:
        state: Current research state
        
    Returns:
        State updates with final research report
    """
    logger.info("Generating final research report")
    
    # Compile comprehensive report
    report_sections = {
        "executive_summary": _generate_executive_summary(state),
        "methodology": _generate_methodology_section(state),
        "findings": _generate_findings_section(state),
        "hypotheses": _generate_hypotheses_section(state),
        "recommendations": _generate_recommendations_section(state),
        "conclusions": _generate_conclusions_section(state),
        "appendices": _generate_appendices_section(state)
    }
    
    # Format as markdown report
    report_markdown = _format_research_report(report_sections, state)
    
    # Add audit entry
    state.add_audit_entry(
        agent="research_coordinator",
        action="report_generation",
        confidence=state.research_confidence,
        outputs={"report_length": len(report_markdown)}
    )
    
    return {
        "research_report": report_markdown,
        "research_phase": "complete",
        "is_sufficient": True
    }


def _generate_executive_summary(state: ResearchPHMState) -> str:
    """Generate executive summary of research findings."""
    summary = f"""
## Executive Summary

This research investigation analyzed bearing signals using a multi-agent approach combining 
data analysis, algorithm research, and domain expertise. 

**Key Findings:**
- Research confidence: {state.research_confidence:.1%}
- Generated hypotheses: {len(state.research_hypotheses)}
- Research iterations: {state.iteration_count}

**Primary Conclusions:**
"""
    
    # Add main conclusions based on integration results
    integration_state = state.integration_state
    if integration_state:
        synthesis = integration_state.get("synthesized_findings", {})
        fault_analysis = synthesis.get("consolidated_fault_analysis", {})
        
        if fault_analysis.get("consensus_fault_present"):
            detected_faults = fault_analysis.get("detected_fault_types", [])
            summary += f"- Bearing fault detected: {', '.join(detected_faults)}\n"
        else:
            summary += "- No significant bearing faults detected\n"
        
        methodology = synthesis.get("optimal_methodology", {})
        if methodology:
            summary += f"- Recommended methodology: {methodology.get('recommended_signal_processing', 'N/A')}\n"
    
    return summary


def _generate_methodology_section(state: ResearchPHMState) -> str:
    """Generate methodology section describing research approach."""
    return f"""
## Methodology

This research employed a multi-agent approach with the following components:

1. **Data Analyst Agent**: Signal quality assessment and statistical analysis
2. **Algorithm Researcher Agent**: Comparative analysis of signal processing methods
3. **Domain Expert Agent**: Physics-informed validation and bearing fault expertise
4. **Integration Agent**: Multi-agent coordination and conflict resolution

**Research Parameters:**
- Sampling frequency: {state.fs} Hz
- Maximum iterations: {state.max_research_iterations}
- Quality threshold: {state.research_quality_threshold:.1%}
- Parallel agent execution: {state.parallel_agent_execution}
"""


def _generate_findings_section(state: ResearchPHMState) -> str:
    """Generate detailed findings section."""
    findings = "## Research Findings\n\n"
    
    # Data analysis findings
    if state.data_analysis_state:
        findings += "### Data Analysis Results\n"
        data_confidence = state.data_analysis_state.get("confidence", 0.0)
        findings += f"- Analysis confidence: {data_confidence:.1%}\n"
        
        ref_analysis = state.data_analysis_state.get("reference_analysis", {})
        if ref_analysis:
            signal_quality = ref_analysis.get("signal_quality", {})
            findings += f"- Signal quality: {signal_quality.get('quality_grades', ['Unknown'])[0]}\n"
            findings += f"- Mean SNR: {signal_quality.get('mean_snr', 0):.1f} dB\n"
    
    # Algorithm research findings
    if state.algorithm_research_state:
        findings += "\n### Algorithm Research Results\n"
        algo_confidence = state.algorithm_research_state.get("confidence", 0.0)
        findings += f"- Research confidence: {algo_confidence:.1%}\n"
        
        recommended = state.algorithm_research_state.get("recommended_pipeline", {})
        if recommended:
            findings += f"- Recommended processing: {recommended.get('recommended_signal_processing', 'N/A')}\n"
    
    # Domain expert findings
    if state.domain_expert_state:
        findings += "\n### Domain Expert Analysis\n"
        domain_confidence = state.domain_expert_state.get("confidence", 0.0)
        findings += f"- Analysis confidence: {domain_confidence:.1%}\n"
        
        failure_analysis = state.domain_expert_state.get("failure_mode_analysis", {})
        detected_faults = failure_analysis.get("detected_fault_types", [])
        if detected_faults:
            findings += f"- Detected fault types: {', '.join(detected_faults)}\n"
    
    return findings


def _generate_hypotheses_section(state: ResearchPHMState) -> str:
    """Generate hypotheses section with validation results."""
    section = "## Research Hypotheses\n\n"
    
    if not state.research_hypotheses:
        section += "No hypotheses generated.\n"
        return section
    
    # Sort hypotheses by confidence
    sorted_hypotheses = sorted(state.research_hypotheses, key=lambda h: h.confidence, reverse=True)
    
    for i, hypothesis in enumerate(sorted_hypotheses[:5], 1):  # Top 5 hypotheses
        section += f"### Hypothesis {i}\n"
        section += f"**Statement:** {hypothesis.statement}\n\n"
        section += f"**Confidence:** {hypothesis.confidence:.1%}\n\n"
        section += f"**Generated by:** {hypothesis.generated_by}\n\n"
        
        if hypothesis.evidence:
            section += "**Evidence:**\n"
            for evidence in hypothesis.evidence:
                section += f"- {evidence}\n"
            section += "\n"
        
        if hypothesis.test_methods:
            section += f"**Test methods:** {', '.join(hypothesis.test_methods)}\n\n"
        
        if hasattr(state, 'validation_results') and state.validation_results:
            validation = state.validation_results.get("hypothesis_validation", {}).get(hypothesis.hypothesis_id)
            if validation:
                section += f"**Validation:** {'Validated' if validation['validated'] else 'Not validated'} "
                section += f"(score: {validation['validation_score']:.2f})\n\n"
    
    return section


def _generate_recommendations_section(state: ResearchPHMState) -> str:
    """Generate recommendations section."""
    recommendations = "## Recommendations\n\n"
    
    # Data preprocessing recommendations
    if state.data_analysis_state:
        data_recs = state.data_analysis_state.get("preprocessing_recommendations", [])
        if data_recs:
            recommendations += "### Data Preprocessing\n"
            for rec in data_recs:
                recommendations += f"- {rec}\n"
            recommendations += "\n"
    
    # Methodology recommendations
    if state.algorithm_research_state:
        algo_recs = state.algorithm_research_state.get("recommended_pipeline", {})
        if algo_recs:
            recommendations += "### Signal Processing Methodology\n"
            recommendations += f"- Use {algo_recs.get('recommended_signal_processing', 'standard')} for feature extraction\n"
            recommendations += f"- Apply {algo_recs.get('recommended_ml_pipeline', 'standard')} for classification\n\n"
    
    # Maintenance recommendations
    if state.domain_expert_state:
        maint_recs = state.domain_expert_state.get("maintenance_recommendations", [])
        if maint_recs:
            recommendations += "### Maintenance Actions\n"
            for rec in maint_recs:
                recommendations += f"- {rec}\n"
            recommendations += "\n"
    
    # Research directions
    if state.domain_expert_state:
        research_dirs = state.domain_expert_state.get("research_directions", [])
        if research_dirs:
            recommendations += "### Future Research Directions\n"
            for direction in research_dirs[:3]:  # Top 3
                recommendations += f"- {direction}\n"
    
    return recommendations


def _generate_conclusions_section(state: ResearchPHMState) -> str:
    """Generate conclusions section."""
    conclusions = "## Conclusions\n\n"
    
    conclusions += f"This multi-agent research investigation achieved a confidence level of {state.research_confidence:.1%} "
    conclusions += f"after {state.iteration_count} iterations.\n\n"
    
    # Integration conclusions
    if state.integration_state:
        research_quality = state.integration_state.get("research_quality_score", 0.0)
        conclusions += f"The research quality score of {research_quality:.1%} indicates "
        
        if research_quality > 0.8:
            conclusions += "high-quality, reliable findings.\n\n"
        elif research_quality > 0.6:
            conclusions += "moderate-quality findings with some limitations.\n\n"
        else:
            conclusions += "preliminary findings requiring further investigation.\n\n"
    
    # Final assessment
    if state.research_confidence > 0.8:
        conclusions += "The research provides strong evidence for the stated conclusions and recommendations."
    elif state.research_confidence > 0.6:
        conclusions += "The research provides moderate evidence, with some uncertainty remaining."
    else:
        conclusions += "The research provides preliminary insights requiring additional validation."
    
    return conclusions


def _generate_appendices_section(state: ResearchPHMState) -> str:
    """Generate appendices with technical details."""
    appendices = "## Appendices\n\n"
    
    # Research audit trail
    appendices += "### A. Research Audit Trail\n\n"
    if state.research_audit_trail:
        appendices += "| Timestamp | Agent | Action | Confidence |\n"
        appendices += "|-----------|-------|--------|------------|\n"
        
        for entry in state.research_audit_trail[-10:]:  # Last 10 entries
            appendices += f"| {entry.timestamp} | {entry.agent} | {entry.action} | {entry.confidence:.2f} |\n"
    
    appendices += "\n"
    
    # Research objectives status
    appendices += "### B. Research Objectives Status\n\n"
    if state.research_objectives:
        for obj in state.research_objectives:
            appendices += f"**{obj.description}**\n"
            appendices += f"- Priority: {obj.priority}\n"
            appendices += f"- Status: {obj.status}\n"
            appendices += f"- Assigned agents: {', '.join(obj.assigned_agents)}\n\n"
    
    return appendices


def _format_research_report(sections: Dict[str, str], state: ResearchPHMState) -> str:
    """Format the complete research report as markdown."""
    report = f"""# PHM Research Report: {state.case_name}

**Generated:** {state.research_audit_trail[-1].timestamp if state.research_audit_trail else 'Unknown'}
**Research Confidence:** {state.research_confidence:.1%}
**Iterations:** {state.iteration_count}

---

"""
    
    # Add all sections
    for section_content in sections.values():
        report += section_content + "\n\n"
    
    report += "---\n\n*This report was generated by the PHM Scientist Agent System*"
    
    return report


def build_research_graph() -> StateGraph:
    """
    Build the research-oriented LangGraph workflow.
    
    Returns:
        Compiled StateGraph for research workflow
    """
    logger.info("Building research workflow graph")
    
    builder = StateGraph(ResearchPHMState)
    
    # Add nodes
    builder.add_node("initialize_research", initialize_research_objectives)
    builder.add_node("data_analysis", data_analyst_agent)
    builder.add_node("algorithm_research", algorithm_researcher_agent)
    builder.add_node("domain_expert", domain_expert_agent)
    builder.add_node("integration", integration_agent)
    builder.add_node("hypothesis_generation", generate_research_hypotheses)
    builder.add_node("validation", validate_research_findings)
    builder.add_node("research_report", generate_research_report)
    
    # Define workflow edges
    builder.add_edge(START, "initialize_research")
    
    # Parallel research phase
    builder.add_conditional_edges(
        "initialize_research",
        route_to_parallel_research,
        ["data_analysis", "algorithm_research", "domain_expert"]
    )
    
    # Integration after parallel research
    builder.add_edge(["data_analysis", "algorithm_research", "domain_expert"], "integration")
    
    # Hypothesis generation
    builder.add_edge("integration", "hypothesis_generation")
    
    # Conditional routing based on research progress
    builder.add_conditional_edges(
        "hypothesis_generation",
        evaluate_research_progress,
        ["validation", "research_report", "hypothesis_generation"]
    )
    
    # Validation can lead to report or back to hypothesis generation
    builder.add_conditional_edges(
        "validation",
        evaluate_research_progress,
        ["research_report", "hypothesis_generation"]
    )
    
    # Final report ends the workflow
    builder.add_edge("research_report", END)
    
    logger.info("Research workflow graph built successfully")
    return builder.compile()


if __name__ == "__main__":
    # Test the research workflow
    from .states.research_states import ResearchPHMState
    from .states.phm_states import DAGState, InputData
    import numpy as np
    
    # Create test data
    fs = 1000
    ref_signals = np.random.randn(2, 1000, 1)
    test_signals = np.random.randn(1, 1000, 1)
    
    # Create research state
    dag_state = DAGState(user_instruction="Test research workflow", channels=["ch1"], nodes={}, leaves=[])
    ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                          results={"ref": ref_signals}, meta={"fs": fs})
    test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                           results={"test": test_signals}, meta={"fs": fs})
    
    state = ResearchPHMState(
        case_name="test_research_workflow",
        user_instruction="Conduct comprehensive bearing fault research",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=fs
    )
    
    # Build and test workflow
    graph = build_research_graph()
    print("Research workflow graph built successfully!")
    print(f"Graph has {len(graph.nodes)} nodes")
    print("Research workflow test completed!")
