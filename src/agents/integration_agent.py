"""
Integration Agent for PHM research workflows.

This agent coordinates between research agents, synthesizes findings,
resolves conflicts, and generates integrated research insights.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

from .research_base import ResearchAgentBase
from ..states.research_states import ResearchPHMState, ResearchHypothesis, ResearchObjective

logger = logging.getLogger(__name__)


class ConflictResolution:
    """Handles conflicts between different agent findings."""
    
    @staticmethod
    def resolve_hypothesis_conflicts(hypotheses: List[ResearchHypothesis]) -> List[ResearchHypothesis]:
        """Resolve conflicts between hypotheses from different agents."""
        # Group hypotheses by similar statements
        hypothesis_groups = defaultdict(list)
        
        for hyp in hypotheses:
            # Simple grouping by keywords in statement
            key_words = set(hyp.statement.lower().split())
            group_key = None
            
            # Find existing group with similar keywords
            for existing_key in hypothesis_groups.keys():
                existing_words = set(existing_key.split())
                if len(key_words.intersection(existing_words)) >= 2:
                    group_key = existing_key
                    break
            
            if group_key is None:
                group_key = hyp.statement.lower()
            
            hypothesis_groups[group_key].append(hyp)
        
        # Resolve conflicts within each group
        resolved_hypotheses = []
        for group_hypotheses in hypothesis_groups.values():
            if len(group_hypotheses) == 1:
                resolved_hypotheses.extend(group_hypotheses)
            else:
                # Merge conflicting hypotheses
                merged_hyp = ConflictResolution._merge_hypotheses(group_hypotheses)
                resolved_hypotheses.append(merged_hyp)
        
        return resolved_hypotheses
    
    @staticmethod
    def _merge_hypotheses(hypotheses: List[ResearchHypothesis]) -> ResearchHypothesis:
        """Merge multiple hypotheses into a consensus hypothesis."""
        # Use highest confidence hypothesis as base
        base_hyp = max(hypotheses, key=lambda h: h.confidence)
        
        # Combine evidence from all hypotheses
        all_evidence = []
        all_test_methods = []
        all_agents = []
        
        for hyp in hypotheses:
            all_evidence.extend(hyp.evidence)
            all_test_methods.extend(hyp.test_methods)
            all_agents.append(hyp.generated_by)
        
        # Calculate consensus confidence (weighted average)
        total_confidence = sum(h.confidence for h in hypotheses)
        consensus_confidence = total_confidence / len(hypotheses)
        
        return ResearchHypothesis(
            hypothesis_id=f"merged_{base_hyp.hypothesis_id}",
            statement=base_hyp.statement,
            confidence=consensus_confidence,
            evidence=list(set(all_evidence)),  # Remove duplicates
            test_methods=list(set(all_test_methods)),
            generated_by=f"consensus_{'+'.join(set(all_agents))}"
        )


class IntegrationAgent(ResearchAgentBase):
    """
    Integration Agent for multi-agent research coordination and synthesis.
    
    Responsibilities:
    - Multi-agent workflow orchestration and task scheduling
    - Research finding synthesis and conflict resolution
    - Hypothesis generation based on integrated agent outputs
    - Research quality assessment and validation
    - Final research report compilation
    """
    
    def __init__(self):
        super().__init__("integration_agent")
        self.conflict_resolver = ConflictResolution()
    
    def analyze(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform integration analysis of all agent findings.
        
        Args:
            state: Current research state with agent results
            
        Returns:
            Dictionary containing integrated analysis results
        """
        logger.info(f"{self.agent_name}: Starting integration analysis")
        
        analysis_results = {
            "agent_coordination": {},
            "synthesized_findings": {},
            "consensus_hypotheses": [],
            "integration_conflicts": [],
            "research_quality_score": 0.0,
            "research_completeness": {},
            "confidence": 0.0
        }
        
        try:
            # Coordinate agent results
            analysis_results["agent_coordination"] = self._coordinate_agent_results(state)
            
            # Synthesize findings across agents
            analysis_results["synthesized_findings"] = self._synthesize_findings(state)
            
            # Resolve hypothesis conflicts
            analysis_results["consensus_hypotheses"] = self._resolve_hypothesis_conflicts(state)
            
            # Identify integration conflicts
            analysis_results["integration_conflicts"] = self._identify_conflicts(state)
            
            # Assess research quality
            analysis_results["research_quality_score"] = self._assess_research_quality(state)
            
            # Evaluate research completeness
            analysis_results["research_completeness"] = self._evaluate_completeness(state)
            
            # Calculate overall confidence
            analysis_results["confidence"] = self.calculate_confidence(analysis_results)
            
            logger.info(f"{self.agent_name}: Integration completed with confidence {analysis_results['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Integration failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["confidence"] = 0.0
        
        return analysis_results
    
    def _coordinate_agent_results(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Coordinate and summarize results from all research agents."""
        coordination = {
            "agent_status": {},
            "result_availability": {},
            "cross_agent_consistency": {},
            "coordination_score": 0.0
        }
        
        # Check status of each agent
        agents = ["data_analyst", "algorithm_researcher", "domain_expert"]
        
        for agent in agents:
            agent_state_key = f"{agent}_state"
            agent_state = getattr(state, agent_state_key, {})
            
            coordination["agent_status"][agent] = {
                "completed": bool(agent_state),
                "confidence": agent_state.get("confidence", 0.0),
                "has_error": "error" in agent_state
            }
            
            coordination["result_availability"][agent] = len(agent_state.keys()) if agent_state else 0
        
        # Assess cross-agent consistency
        coordination["cross_agent_consistency"] = self._assess_cross_agent_consistency(state)
        
        # Calculate coordination score
        completed_agents = sum(1 for status in coordination["agent_status"].values() if status["completed"])
        coordination["coordination_score"] = completed_agents / len(agents)
        
        return coordination
    
    def _assess_cross_agent_consistency(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Assess consistency between different agent findings."""
        consistency = {
            "signal_quality_agreement": 0.0,
            "fault_detection_agreement": 0.0,
            "confidence_alignment": 0.0,
            "overall_consistency": 0.0
        }
        
        try:
            # Get agent confidences
            data_confidence = state.data_analysis_state.get("confidence", 0.0)
            algo_confidence = state.algorithm_research_state.get("confidence", 0.0)
            domain_confidence = state.domain_expert_state.get("confidence", 0.0)
            
            confidences = [data_confidence, algo_confidence, domain_confidence]
            valid_confidences = [c for c in confidences if c > 0]
            
            if valid_confidences:
                # Confidence alignment (low variance indicates good alignment)
                confidence_variance = np.var(valid_confidences)
                consistency["confidence_alignment"] = max(0.0, 1.0 - confidence_variance)
            
            # Signal quality agreement
            data_quality = state.data_analysis_state.get("reference_analysis", {}).get("signal_quality", {})
            if data_quality:
                mean_snr = data_quality.get("mean_snr", 0)
                quality_grade = data_quality.get("quality_grades", ["F"])[0]
                
                # Domain expert should agree on signal quality
                domain_physics = state.domain_expert_state.get("physics_validation", {})
                validation_score = domain_physics.get("validation_score", 0.5)
                
                # Agreement if both indicate good or both indicate poor quality
                data_good_quality = mean_snr > 10 and quality_grade in ["A", "B"]
                domain_good_quality = validation_score > 0.7
                
                consistency["signal_quality_agreement"] = 1.0 if data_good_quality == domain_good_quality else 0.0
            
            # Fault detection agreement
            domain_faults = state.domain_expert_state.get("failure_mode_analysis", {}).get("detected_fault_types", [])
            algo_performance = state.algorithm_research_state.get("performance_benchmarks", {})
            
            # If domain expert detects faults, algorithm performance should reflect this
            if domain_faults:
                # Check if algorithm research found good separability
                sp_comparison = state.algorithm_research_state.get("signal_processing_comparison", {})
                if sp_comparison:
                    separabilities = [results.get("separability", 0) for results in sp_comparison.values() 
                                    if isinstance(results, dict)]
                    if separabilities:
                        avg_separability = np.mean(separabilities)
                        consistency["fault_detection_agreement"] = min(avg_separability / 5.0, 1.0)
            else:
                # If no faults detected, separability should be low
                consistency["fault_detection_agreement"] = 0.5  # Neutral
            
            # Overall consistency
            consistency["overall_consistency"] = np.mean([
                consistency["signal_quality_agreement"],
                consistency["fault_detection_agreement"],
                consistency["confidence_alignment"]
            ])
            
        except Exception as e:
            logger.warning(f"Cross-agent consistency assessment failed: {e}")
        
        return consistency
    
    def _synthesize_findings(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Synthesize findings from all research agents."""
        synthesis = {
            "integrated_signal_assessment": {},
            "consolidated_fault_analysis": {},
            "optimal_methodology": {},
            "research_insights": [],
            "synthesis_confidence": 0.0
        }
        
        try:
            # Integrate signal assessment
            data_analysis = state.data_analysis_state
            domain_analysis = state.domain_expert_state
            
            ref_quality = data_analysis.get("reference_analysis", {}).get("signal_quality", {})
            physics_validation = domain_analysis.get("physics_validation", {})
            
            synthesis["integrated_signal_assessment"] = {
                "quality_grade": ref_quality.get("quality_grades", ["Unknown"])[0],
                "snr_db": ref_quality.get("mean_snr", 0),
                "physics_consistent": physics_validation.get("validation_score", 0.5) > 0.7,
                "preprocessing_needed": len(data_analysis.get("preprocessing_recommendations", [])) > 0
            }
            
            # Consolidate fault analysis
            domain_faults = domain_analysis.get("failure_mode_analysis", {}).get("detected_fault_types", [])
            domain_confidence = domain_analysis.get("failure_mode_analysis", {}).get("confidence_scores", {})
            
            algo_separability = state.algorithm_research_state.get("signal_processing_comparison", {})
            
            synthesis["consolidated_fault_analysis"] = {
                "detected_fault_types": domain_faults,
                "fault_confidence": domain_confidence,
                "algorithmic_separability": algo_separability,
                "consensus_fault_present": len(domain_faults) > 0
            }
            
            # Determine optimal methodology
            algo_recommendations = state.algorithm_research_state.get("recommended_pipeline", {})
            domain_directions = domain_analysis.get("research_directions", [])
            
            synthesis["optimal_methodology"] = {
                "recommended_signal_processing": algo_recommendations.get("recommended_signal_processing", "fft"),
                "recommended_ml_pipeline": algo_recommendations.get("recommended_ml_pipeline", "unknown"),
                "domain_guided_directions": domain_directions[:3],  # Top 3 directions
                "methodology_confidence": algo_recommendations.get("confidence", 0.0)
            }
            
            # Generate integrated research insights
            synthesis["research_insights"] = self._generate_integrated_insights(state)
            
            # Calculate synthesis confidence
            agent_confidences = [
                data_analysis.get("confidence", 0.0),
                state.algorithm_research_state.get("confidence", 0.0),
                domain_analysis.get("confidence", 0.0)
            ]
            valid_confidences = [c for c in agent_confidences if c > 0]
            synthesis["synthesis_confidence"] = np.mean(valid_confidences) if valid_confidences else 0.0
            
        except Exception as e:
            logger.warning(f"Findings synthesis failed: {e}")
            synthesis["error"] = str(e)
        
        return synthesis
    
    def _generate_integrated_insights(self, state: ResearchPHMState) -> List[str]:
        """Generate insights by integrating findings across all agents."""
        insights = []
        
        # Data quality insights
        data_quality = state.data_analysis_state.get("reference_analysis", {}).get("signal_quality", {})
        if data_quality.get("mean_snr", 0) > 15:
            insights.append("High signal quality enables reliable fault detection with standard methods")
        elif data_quality.get("mean_snr", 0) < 10:
            insights.append("Low signal quality requires advanced preprocessing and robust algorithms")
        
        # Algorithm-domain integration insights
        domain_faults = state.domain_expert_state.get("failure_mode_analysis", {}).get("detected_fault_types", [])
        algo_performance = state.algorithm_research_state.get("performance_benchmarks", {})
        
        if domain_faults and algo_performance:
            insights.append("Domain knowledge confirms algorithmic fault detection capabilities")
        elif domain_faults and not algo_performance:
            insights.append("Domain expertise suggests faults that require more sophisticated algorithms")
        
        # Physics-informed insights
        physics_validation = state.domain_expert_state.get("physics_validation", {})
        if physics_validation.get("validation_score", 1.0) < 0.7:
            insights.append("Physics validation suggests need for measurement system verification")
        
        # Methodology insights
        algo_recommendations = state.algorithm_research_state.get("recommended_pipeline", {})
        if algo_recommendations.get("confidence", 0.0) > 0.8:
            insights.append("High confidence in recommended signal processing methodology")
        
        return insights
    
    def _resolve_hypothesis_conflicts(self, state: ResearchPHMState) -> List[ResearchHypothesis]:
        """Resolve conflicts between hypotheses from different agents."""
        all_hypotheses = state.research_hypotheses
        
        if not all_hypotheses:
            return []
        
        # Use conflict resolution to merge similar hypotheses
        resolved_hypotheses = self.conflict_resolver.resolve_hypothesis_conflicts(all_hypotheses)
        
        # Sort by confidence
        resolved_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return resolved_hypotheses
    
    def _identify_conflicts(self, state: ResearchPHMState) -> List[str]:
        """Identify conflicts between agent findings."""
        conflicts = []
        
        try:
            # Check for confidence conflicts
            data_confidence = state.data_analysis_state.get("confidence", 0.0)
            algo_confidence = state.algorithm_research_state.get("confidence", 0.0)
            domain_confidence = state.domain_expert_state.get("confidence", 0.0)
            
            confidences = [data_confidence, algo_confidence, domain_confidence]
            valid_confidences = [c for c in confidences if c > 0]
            
            if len(valid_confidences) > 1:
                confidence_range = max(valid_confidences) - min(valid_confidences)
                if confidence_range > 0.5:
                    conflicts.append(f"Large confidence discrepancy between agents (range: {confidence_range:.2f})")
            
            # Check for fault detection conflicts
            domain_faults = state.domain_expert_state.get("failure_mode_analysis", {}).get("detected_fault_types", [])
            data_anomalies = state.data_analysis_state.get("reference_analysis", {}).get("anomaly_detection", {})
            
            if domain_faults and data_anomalies.get("n_outliers", 0) == 0:
                conflicts.append("Domain expert detects faults but data analysis shows no anomalies")
            
            # Check for methodology conflicts
            data_recommendations = state.data_analysis_state.get("preprocessing_recommendations", [])
            algo_recommendations = state.algorithm_research_state.get("recommended_pipeline", {})
            
            if ("noise reduction" in str(data_recommendations).lower() and 
                algo_recommendations.get("confidence", 0.0) > 0.8):
                conflicts.append("Data analysis suggests noise issues but algorithm research shows high confidence")
            
        except Exception as e:
            logger.warning(f"Conflict identification failed: {e}")
            conflicts.append(f"Error in conflict detection: {str(e)}")
        
        return conflicts
    
    def _assess_research_quality(self, state: ResearchPHMState) -> float:
        """Assess overall research quality based on multiple factors."""
        quality_factors = {
            "agent_completion": 0.0,
            "confidence_consistency": 0.0,
            "hypothesis_quality": 0.0,
            "validation_strength": 0.0
        }
        
        try:
            # Agent completion factor
            completed_agents = 0
            total_agents = 3
            
            if state.data_analysis_state:
                completed_agents += 1
            if state.algorithm_research_state:
                completed_agents += 1
            if state.domain_expert_state:
                completed_agents += 1
            
            quality_factors["agent_completion"] = completed_agents / total_agents
            
            # Confidence consistency
            confidences = [
                state.data_analysis_state.get("confidence", 0.0),
                state.algorithm_research_state.get("confidence", 0.0),
                state.domain_expert_state.get("confidence", 0.0)
            ]
            valid_confidences = [c for c in confidences if c > 0]
            
            if valid_confidences:
                confidence_variance = np.var(valid_confidences)
                quality_factors["confidence_consistency"] = max(0.0, 1.0 - confidence_variance)
            
            # Hypothesis quality
            hypotheses = state.research_hypotheses
            if hypotheses:
                avg_hypothesis_confidence = np.mean([h.confidence for h in hypotheses])
                testable_fraction = sum(1 for h in hypotheses if h.testable) / len(hypotheses)
                quality_factors["hypothesis_quality"] = (avg_hypothesis_confidence + testable_fraction) / 2
            
            # Validation strength
            physics_validation = state.domain_expert_state.get("physics_validation", {})
            validation_score = physics_validation.get("validation_score", 0.5)
            quality_factors["validation_strength"] = validation_score
            
        except Exception as e:
            logger.warning(f"Research quality assessment failed: {e}")
        
        # Weighted average of quality factors
        weights = {"agent_completion": 0.3, "confidence_consistency": 0.2, 
                  "hypothesis_quality": 0.3, "validation_strength": 0.2}
        
        overall_quality = sum(quality_factors[factor] * weights[factor] 
                            for factor in quality_factors.keys())
        
        return overall_quality
    
    def _evaluate_completeness(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Evaluate research completeness across different dimensions."""
        completeness = {
            "data_analysis_completeness": 0.0,
            "algorithm_research_completeness": 0.0,
            "domain_expertise_completeness": 0.0,
            "hypothesis_coverage": 0.0,
            "overall_completeness": 0.0
        }
        
        try:
            # Data analysis completeness
            data_analysis = state.data_analysis_state
            data_components = ["signal_quality", "statistical_summary", "feature_analysis", "frequency_analysis"]
            data_completed = sum(1 for comp in data_components 
                               if comp in data_analysis.get("reference_analysis", {}))
            completeness["data_analysis_completeness"] = data_completed / len(data_components)
            
            # Algorithm research completeness
            algo_analysis = state.algorithm_research_state
            algo_components = ["signal_processing_comparison", "ml_algorithm_comparison", "performance_benchmarks"]
            algo_completed = sum(1 for comp in algo_components if comp in algo_analysis)
            completeness["algorithm_research_completeness"] = algo_completed / len(algo_components)
            
            # Domain expertise completeness
            domain_analysis = state.domain_expert_state
            domain_components = ["physics_validation", "failure_mode_analysis", "bearing_frequency_analysis"]
            domain_completed = sum(1 for comp in domain_components if comp in domain_analysis)
            completeness["domain_expertise_completeness"] = domain_completed / len(domain_components)
            
            # Hypothesis coverage
            hypotheses = state.research_hypotheses
            if hypotheses:
                agents_with_hypotheses = set(h.generated_by for h in hypotheses)
                completeness["hypothesis_coverage"] = len(agents_with_hypotheses) / 3  # 3 main agents
            
            # Overall completeness
            completeness["overall_completeness"] = np.mean([
                completeness["data_analysis_completeness"],
                completeness["algorithm_research_completeness"],
                completeness["domain_expertise_completeness"],
                completeness["hypothesis_coverage"]
            ])
            
        except Exception as e:
            logger.warning(f"Completeness evaluation failed: {e}")
        
        return completeness
    
    def generate_hypotheses(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate integration-level hypotheses based on synthesized findings."""
        hypotheses = []
        
        # Meta-hypothesis about research quality
        research_quality = analysis_results.get("research_quality_score", 0.0)
        if research_quality > 0.8:
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_integration_quality_{len(hypotheses)}",
                statement="High research quality enables reliable conclusions about bearing condition",
                confidence=research_quality,
                generated_by=self.agent_name,
                evidence=[f"Research quality score: {research_quality:.3f}"],
                test_methods=["cross_validation", "independent_verification"]
            ))
        
        # Hypothesis about agent consensus
        synthesis = analysis_results.get("synthesized_findings", {})
        consensus_fault = synthesis.get("consolidated_fault_analysis", {}).get("consensus_fault_present", False)
        
        if consensus_fault:
            synthesis_confidence = synthesis.get("synthesis_confidence", 0.0)
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_consensus_fault_{len(hypotheses)}",
                statement="Multi-agent consensus confirms presence of bearing fault condition",
                confidence=synthesis_confidence,
                generated_by=self.agent_name,
                evidence=["Agreement across data analysis, algorithm research, and domain expertise"],
                test_methods=["independent_validation", "expert_review"]
            ))
        
        return hypotheses


def integration_agent(state: ResearchPHMState) -> Dict[str, Any]:
    """
    LangGraph node function for the Integration Agent.
    
    Args:
        state: Current research state with all agent results
        
    Returns:
        State updates from integration analysis
    """
    agent = IntegrationAgent()
    
    # Perform integration analysis
    analysis_results = agent.analyze(state)
    
    # Generate integration hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_results)
    
    # Update research confidence based on integration
    research_confidence = analysis_results.get("research_quality_score", 0.0)
    
    # Update state
    state_updates = {
        "integration_state": analysis_results,
        "research_hypotheses": state.research_hypotheses + hypotheses,
        "research_confidence": research_confidence
    }
    
    # Add audit entry
    state.add_audit_entry(
        agent="integration_agent",
        action="multi_agent_integration",
        confidence=analysis_results.get("confidence", 0.0),
        outputs={"research_quality": research_confidence, "n_hypotheses": len(hypotheses)}
    )
    
    return state_updates


if __name__ == "__main__":
    # Test the Integration Agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from states.research_states import ResearchPHMState, ResearchHypothesis
    from states.phm_states import DAGState, InputData
    import numpy as np
    
    # Create test state with mock agent results
    dag_state = DAGState(user_instruction="Test integration", channels=["ch1"], nodes={}, leaves=[])
    ref_signal = InputData(node_id="ref", parents=[], shape=(1, 1000, 1), 
                          results={"ref": np.random.randn(1, 1000, 1)}, meta={"fs": 1000})
    test_signal = InputData(node_id="test", parents=[], shape=(1, 1000, 1),
                           results={"test": np.random.randn(1, 1000, 1)}, meta={"fs": 1000})
    
    state = ResearchPHMState(
        case_name="test_integration",
        user_instruction="Test multi-agent integration",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=1000
    )
    
    # Add mock agent results
    state.data_analysis_state = {"confidence": 0.8, "reference_analysis": {"signal_quality": {"mean_snr": 15}}}
    state.algorithm_research_state = {"confidence": 0.7, "recommended_pipeline": {"confidence": 0.8}}
    state.domain_expert_state = {"confidence": 0.9, "failure_mode_analysis": {"detected_fault_types": ["ball_fault"]}}
    
    # Add mock hypotheses
    state.research_hypotheses = [
        ResearchHypothesis(
            hypothesis_id="hyp1", statement="Signal shows fault characteristics", 
            confidence=0.8, generated_by="data_analyst"
        ),
        ResearchHypothesis(
            hypothesis_id="hyp2", statement="Ball fault detected in bearing", 
            confidence=0.9, generated_by="domain_expert"
        )
    ]
    
    # Test the agent
    agent = IntegrationAgent()
    results = agent.analyze(state)
    hypotheses = agent.generate_hypotheses(state, results)
    
    print(f"Integration analysis completed with confidence: {results['confidence']:.3f}")
    print(f"Research quality score: {results['research_quality_score']:.3f}")
    print(f"Generated {len(hypotheses)} integration hypotheses")
    print("Integration Agent test completed successfully!")
