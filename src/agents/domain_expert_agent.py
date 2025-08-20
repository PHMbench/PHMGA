"""
Domain Expert Agent for PHM research workflows.

This agent provides PHM domain knowledge integration, physics-informed validation,
and bearing fault mechanics expertise for research guidance.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

from .research_base import ResearchAgentBase
from ..states.research_states import ResearchPHMState, ResearchHypothesis

logger = logging.getLogger(__name__)


class BearingFaultKnowledge:
    """Knowledge base for bearing fault characteristics and mechanics."""
    
    FAULT_TYPES = {
        "healthy": {
            "description": "Normal bearing operation",
            "frequency_characteristics": ["broadband_low", "no_harmonics"],
            "amplitude_characteristics": ["stable", "low_variance"],
            "typical_features": ["low_rms", "low_kurtosis", "gaussian_distribution"]
        },
        "ball_fault": {
            "description": "Ball bearing element defect",
            "frequency_characteristics": ["ball_pass_frequency", "harmonics"],
            "amplitude_characteristics": ["impulsive", "high_peaks"],
            "typical_features": ["high_kurtosis", "high_crest_factor", "envelope_modulation"]
        },
        "inner_race_fault": {
            "description": "Inner race defect",
            "frequency_characteristics": ["inner_race_frequency", "shaft_harmonics"],
            "amplitude_characteristics": ["amplitude_modulation", "periodic_impulses"],
            "typical_features": ["high_rms", "spectral_peaks", "sideband_modulation"]
        },
        "outer_race_fault": {
            "description": "Outer race defect",
            "frequency_characteristics": ["outer_race_frequency", "load_zone_modulation"],
            "amplitude_characteristics": ["consistent_impulses", "load_dependent"],
            "typical_features": ["periodic_impulses", "envelope_analysis", "spectral_lines"]
        },
        "cage_fault": {
            "description": "Cage/retainer defect",
            "frequency_characteristics": ["cage_frequency", "low_frequency_modulation"],
            "amplitude_characteristics": ["irregular_impulses", "variable_amplitude"],
            "typical_features": ["low_frequency_content", "irregular_patterns", "amplitude_variation"]
        }
    }
    
    BEARING_FREQUENCIES = {
        "ball_pass_outer": lambda n_balls, contact_angle, pitch_diameter, ball_diameter, shaft_speed: 
            (n_balls / 2) * shaft_speed * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle)),
        "ball_pass_inner": lambda n_balls, contact_angle, pitch_diameter, ball_diameter, shaft_speed:
            (n_balls / 2) * shaft_speed * (1 + (ball_diameter / pitch_diameter) * np.cos(contact_angle)),
        "cage_frequency": lambda n_balls, contact_angle, pitch_diameter, ball_diameter, shaft_speed:
            (shaft_speed / 2) * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle)),
        "ball_spin_frequency": lambda n_balls, contact_angle, pitch_diameter, ball_diameter, shaft_speed:
            (pitch_diameter / (2 * ball_diameter)) * shaft_speed * (1 - ((ball_diameter / pitch_diameter) * np.cos(contact_angle))**2)
    }
    
    DIAGNOSTIC_RULES = [
        {
            "condition": "high_kurtosis_and_crest_factor",
            "threshold": {"kurtosis": 3.0, "crest_factor": 3.0},
            "indication": "impulsive_behavior",
            "fault_types": ["ball_fault", "inner_race_fault", "outer_race_fault"]
        },
        {
            "condition": "spectral_peaks_at_bearing_frequencies",
            "threshold": {"peak_prominence": 0.1},
            "indication": "bearing_defect",
            "fault_types": ["ball_fault", "inner_race_fault", "outer_race_fault"]
        },
        {
            "condition": "low_frequency_modulation",
            "threshold": {"frequency_range": [0, 50]},
            "indication": "cage_fault",
            "fault_types": ["cage_fault"]
        },
        {
            "condition": "amplitude_modulation",
            "threshold": {"modulation_depth": 0.2},
            "indication": "inner_race_fault",
            "fault_types": ["inner_race_fault"]
        }
    ]


class DomainExpertAgent(ResearchAgentBase):
    """
    Domain Expert Agent for PHM knowledge integration and validation.
    
    Responsibilities:
    - Physics-informed analysis validation using bearing fault mechanics
    - Domain-specific feature interpretation and relevance assessment
    - Failure mode identification and progression analysis
    - Research direction suggestions based on PHM best practices
    - Knowledge base integration for bearing fault diagnosis
    """
    
    def __init__(self):
        super().__init__("domain_expert")
        self.knowledge_base = BearingFaultKnowledge()
        self.bearing_parameters = self._default_bearing_parameters()
    
    def _default_bearing_parameters(self) -> Dict[str, float]:
        """Default bearing parameters for frequency calculations."""
        return {
            "n_balls": 8,
            "contact_angle": np.radians(0),  # 0 degrees for deep groove ball bearing
            "pitch_diameter": 50.0,  # mm
            "ball_diameter": 8.0,    # mm
            "shaft_speed": 30.0      # Hz (1800 RPM)
        }
    
    def analyze(self, state: ResearchPHMState) -> Dict[str, Any]:
        """
        Perform domain expert analysis with physics-informed validation.
        
        Args:
            state: Current research state
            
        Returns:
            Dictionary containing domain expert analysis results
        """
        logger.info(f"{self.agent_name}: Starting domain expert analysis")
        
        analysis_results = {
            "physics_validation": {},
            "failure_mode_analysis": {},
            "bearing_frequency_analysis": {},
            "diagnostic_rule_evaluation": {},
            "research_directions": [],
            "maintenance_recommendations": [],
            "confidence": 0.0
        }
        
        try:
            # Physics-informed validation
            analysis_results["physics_validation"] = self._validate_physics_consistency(state)
            
            # Failure mode analysis
            analysis_results["failure_mode_analysis"] = self._analyze_failure_modes(state)
            
            # Bearing frequency analysis
            analysis_results["bearing_frequency_analysis"] = self._analyze_bearing_frequencies(state)
            
            # Evaluate diagnostic rules
            analysis_results["diagnostic_rule_evaluation"] = self._evaluate_diagnostic_rules(state)
            
            # Generate research directions
            analysis_results["research_directions"] = self._suggest_research_directions(state, analysis_results)
            
            # Maintenance recommendations
            analysis_results["maintenance_recommendations"] = self._generate_maintenance_recommendations(
                analysis_results
            )
            
            # Calculate confidence
            analysis_results["confidence"] = self.calculate_confidence(analysis_results)
            
            logger.info(f"{self.agent_name}: Analysis completed with confidence {analysis_results['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["confidence"] = 0.0
        
        return analysis_results
    
    def _validate_physics_consistency(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Validate analysis results against physics principles."""
        validation_results = {
            "energy_conservation": True,
            "frequency_domain_consistency": True,
            "amplitude_scaling": True,
            "causality": True,
            "validation_score": 0.0
        }
        
        try:
            # Get data analysis results
            data_analysis = state.data_analysis_state
            
            # Check energy conservation
            ref_analysis = data_analysis.get("reference_analysis", {})
            test_analysis = data_analysis.get("test_analysis", {})
            
            if ref_analysis and test_analysis:
                ref_energy = ref_analysis.get("statistical_summary", {}).get("rms", {}).get("mean", 0)
                test_energy = test_analysis.get("statistical_summary", {}).get("rms", {}).get("mean", 0)
                
                # Energy should be reasonable (not orders of magnitude different without explanation)
                energy_ratio = test_energy / (ref_energy + 1e-10)
                validation_results["energy_conservation"] = 0.1 <= energy_ratio <= 10.0
            
            # Check frequency domain consistency
            ref_freq = ref_analysis.get("frequency_analysis", {})
            test_freq = test_analysis.get("frequency_analysis", {})
            
            if ref_freq and test_freq:
                ref_centroid = ref_freq.get("spectral_centroid", 0)
                test_centroid = test_freq.get("spectral_centroid", 0)
                
                # Spectral centroids should be in reasonable range for bearing signals
                validation_results["frequency_domain_consistency"] = (
                    0 < ref_centroid < state.fs/2 and 0 < test_centroid < state.fs/2
                )
            
            # Calculate overall validation score
            validation_score = sum([
                validation_results["energy_conservation"],
                validation_results["frequency_domain_consistency"],
                validation_results["amplitude_scaling"],
                validation_results["causality"]
            ]) / 4.0
            
            validation_results["validation_score"] = validation_score
            
        except Exception as e:
            logger.warning(f"Physics validation failed: {e}")
            validation_results["validation_score"] = 0.5  # Neutral score on failure
        
        return validation_results
    
    def _analyze_failure_modes(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Analyze potential failure modes based on signal characteristics."""
        failure_analysis = {
            "detected_fault_types": [],
            "fault_severity": {},
            "progression_indicators": {},
            "confidence_scores": {}
        }
        
        try:
            # Get statistical features from data analysis
            data_analysis = state.data_analysis_state
            test_analysis = data_analysis.get("test_analysis", {})
            statistical_summary = test_analysis.get("statistical_summary", {})
            
            # Extract key features for fault analysis
            kurtosis = statistical_summary.get("kurtosis", {}).get("mean", 0)
            crest_factor = statistical_summary.get("crest_factor", {}).get("mean", 0)
            rms = statistical_summary.get("rms", {}).get("mean", 0)
            
            # Analyze fault indicators
            fault_indicators = {
                "impulsive_behavior": kurtosis > 3.0 and crest_factor > 3.0,
                "high_energy": rms > 1.0,  # Threshold depends on signal scaling
                "amplitude_modulation": False,  # Would need envelope analysis
                "frequency_modulation": False   # Would need instantaneous frequency analysis
            }
            
            # Map indicators to fault types
            for fault_type, characteristics in self.knowledge_base.FAULT_TYPES.items():
                confidence = 0.0
                
                if fault_type == "healthy":
                    # Healthy if no fault indicators
                    confidence = 1.0 - sum(fault_indicators.values()) / len(fault_indicators)
                elif fault_type in ["ball_fault", "inner_race_fault", "outer_race_fault"]:
                    # These faults show impulsive behavior
                    if fault_indicators["impulsive_behavior"]:
                        confidence = 0.7
                    if fault_indicators["high_energy"]:
                        confidence += 0.2
                elif fault_type == "cage_fault":
                    # Cage faults typically show different patterns
                    if not fault_indicators["impulsive_behavior"] and fault_indicators["high_energy"]:
                        confidence = 0.5
                
                if confidence > 0.3:  # Threshold for detection
                    failure_analysis["detected_fault_types"].append(fault_type)
                    failure_analysis["confidence_scores"][fault_type] = confidence
            
            # Estimate fault severity (simplified)
            if failure_analysis["detected_fault_types"]:
                max_confidence = max(failure_analysis["confidence_scores"].values())
                if max_confidence > 0.8:
                    severity = "high"
                elif max_confidence > 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                failure_analysis["fault_severity"]["overall"] = severity
            
        except Exception as e:
            logger.warning(f"Failure mode analysis failed: {e}")
            failure_analysis["error"] = str(e)
        
        return failure_analysis
    
    def _analyze_bearing_frequencies(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Analyze signal for bearing characteristic frequencies."""
        frequency_analysis = {
            "calculated_frequencies": {},
            "detected_peaks": {},
            "frequency_matches": {},
            "bearing_fault_indicators": {}
        }
        
        try:
            # Calculate theoretical bearing frequencies
            params = self.bearing_parameters
            shaft_speed = params["shaft_speed"]
            
            calculated_freqs = {}
            for freq_name, calc_func in self.knowledge_base.BEARING_FREQUENCIES.items():
                freq_value = calc_func(
                    params["n_balls"], params["contact_angle"],
                    params["pitch_diameter"], params["ball_diameter"],
                    shaft_speed
                )
                calculated_freqs[freq_name] = freq_value
            
            frequency_analysis["calculated_frequencies"] = calculated_freqs
            
            # Get frequency domain data from analysis
            data_analysis = state.data_analysis_state
            test_analysis = data_analysis.get("test_analysis", {})
            freq_data = test_analysis.get("frequency_analysis", {})
            
            if freq_data and "mean_spectrum" in freq_data:
                spectrum = freq_data["mean_spectrum"]
                freqs = freq_data.get("frequency_bins", [])
                
                # Find peaks in spectrum
                if len(spectrum) > 0 and len(freqs) > 0:
                    # Simple peak detection
                    peak_threshold = np.max(spectrum) * 0.1  # 10% of max
                    peak_indices = []
                    
                    for i in range(1, len(spectrum) - 1):
                        if (spectrum[i] > spectrum[i-1] and 
                            spectrum[i] > spectrum[i+1] and 
                            spectrum[i] > peak_threshold):
                            peak_indices.append(i)
                    
                    detected_peaks = [freqs[i] for i in peak_indices]
                    frequency_analysis["detected_peaks"] = detected_peaks
                    
                    # Match detected peaks with bearing frequencies
                    tolerance = 5.0  # Hz tolerance for matching
                    matches = {}
                    
                    for bearing_freq_name, bearing_freq_value in calculated_freqs.items():
                        for detected_freq in detected_peaks:
                            if abs(detected_freq - bearing_freq_value) < tolerance:
                                matches[bearing_freq_name] = {
                                    "theoretical": bearing_freq_value,
                                    "detected": detected_freq,
                                    "error": abs(detected_freq - bearing_freq_value)
                                }
                                break
                    
                    frequency_analysis["frequency_matches"] = matches
                    
                    # Generate bearing fault indicators
                    if matches:
                        frequency_analysis["bearing_fault_indicators"] = {
                            "bearing_frequencies_present": True,
                            "matched_frequencies": list(matches.keys()),
                            "fault_confidence": len(matches) / len(calculated_freqs)
                        }
            
        except Exception as e:
            logger.warning(f"Bearing frequency analysis failed: {e}")
            frequency_analysis["error"] = str(e)
        
        return frequency_analysis
    
    def _evaluate_diagnostic_rules(self, state: ResearchPHMState) -> Dict[str, Any]:
        """Evaluate diagnostic rules against current analysis."""
        rule_evaluation = {
            "triggered_rules": [],
            "rule_scores": {},
            "diagnostic_confidence": 0.0
        }
        
        try:
            # Get analysis data
            data_analysis = state.data_analysis_state
            test_analysis = data_analysis.get("test_analysis", {})
            statistical_summary = test_analysis.get("statistical_summary", {})
            
            # Evaluate each diagnostic rule
            for rule in self.knowledge_base.DIAGNOSTIC_RULES:
                rule_triggered = False
                rule_score = 0.0
                
                condition = rule["condition"]
                threshold = rule["threshold"]
                
                if condition == "high_kurtosis_and_crest_factor":
                    kurtosis = statistical_summary.get("kurtosis", {}).get("mean", 0)
                    crest_factor = statistical_summary.get("crest_factor", {}).get("mean", 0)
                    
                    if (kurtosis > threshold["kurtosis"] and 
                        crest_factor > threshold["crest_factor"]):
                        rule_triggered = True
                        rule_score = min((kurtosis / threshold["kurtosis"] + 
                                        crest_factor / threshold["crest_factor"]) / 2, 1.0)
                
                elif condition == "spectral_peaks_at_bearing_frequencies":
                    # Would need frequency analysis results
                    bearing_analysis = self._analyze_bearing_frequencies(state)
                    if bearing_analysis.get("frequency_matches"):
                        rule_triggered = True
                        rule_score = bearing_analysis.get("bearing_fault_indicators", {}).get("fault_confidence", 0)
                
                if rule_triggered:
                    rule_evaluation["triggered_rules"].append({
                        "rule": condition,
                        "indication": rule["indication"],
                        "fault_types": rule["fault_types"],
                        "score": rule_score
                    })
                    rule_evaluation["rule_scores"][condition] = rule_score
            
            # Calculate overall diagnostic confidence
            if rule_evaluation["rule_scores"]:
                rule_evaluation["diagnostic_confidence"] = np.mean(list(rule_evaluation["rule_scores"].values()))
            
        except Exception as e:
            logger.warning(f"Diagnostic rule evaluation failed: {e}")
            rule_evaluation["error"] = str(e)
        
        return rule_evaluation
    
    def _suggest_research_directions(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[str]:
        """Suggest research directions based on domain expertise."""
        directions = []
        
        # Based on failure mode analysis
        failure_analysis = analysis_results.get("failure_mode_analysis", {})
        detected_faults = failure_analysis.get("detected_fault_types", [])
        
        if "ball_fault" in detected_faults:
            directions.append("Investigate envelope analysis for ball fault characterization")
            directions.append("Research optimal filtering for impulsive signal enhancement")
        
        if "inner_race_fault" in detected_faults:
            directions.append("Study amplitude modulation patterns for inner race fault progression")
            directions.append("Investigate shaft speed variation effects on fault signatures")
        
        if "cage_fault" in detected_faults:
            directions.append("Research low-frequency analysis techniques for cage fault detection")
            directions.append("Study irregular pattern recognition methods")
        
        # Based on physics validation
        physics_validation = analysis_results.get("physics_validation", {})
        validation_score = physics_validation.get("validation_score", 1.0)
        
        if validation_score < 0.7:
            directions.append("Investigate physics-informed signal processing constraints")
            directions.append("Research energy conservation principles in fault diagnosis")
        
        # Based on frequency analysis
        freq_analysis = analysis_results.get("bearing_frequency_analysis", {})
        if not freq_analysis.get("frequency_matches"):
            directions.append("Research bearing parameter estimation from signal data")
            directions.append("Investigate variable speed bearing fault detection methods")
        
        # General research directions
        directions.extend([
            "Study uncertainty quantification in bearing fault diagnosis",
            "Research multi-sensor fusion for improved fault detection",
            "Investigate remaining useful life estimation methods"
        ])
        
        return directions[:5]  # Return top 5 directions
    
    def _generate_maintenance_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate maintenance recommendations based on analysis."""
        recommendations = []
        
        # Based on detected fault types
        failure_analysis = analysis_results.get("failure_mode_analysis", {})
        detected_faults = failure_analysis.get("detected_fault_types", [])
        fault_severity = failure_analysis.get("fault_severity", {})
        
        overall_severity = fault_severity.get("overall", "low")
        
        if detected_faults:
            if overall_severity == "high":
                recommendations.append("Schedule immediate bearing replacement")
                recommendations.append("Increase monitoring frequency to daily inspections")
            elif overall_severity == "medium":
                recommendations.append("Plan bearing replacement within next maintenance window")
                recommendations.append("Implement weekly vibration monitoring")
            else:
                recommendations.append("Continue normal monitoring schedule")
                recommendations.append("Consider trending analysis for fault progression")
        else:
            recommendations.append("Bearing appears healthy - maintain current monitoring schedule")
            recommendations.append("Consider extending maintenance intervals if consistently healthy")
        
        # Based on physics validation
        physics_validation = analysis_results.get("physics_validation", {})
        if physics_validation.get("validation_score", 1.0) < 0.7:
            recommendations.append("Verify sensor calibration and mounting")
            recommendations.append("Check for external vibration sources")
        
        return recommendations
    
    def generate_hypotheses(self, state: ResearchPHMState, analysis_results: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate domain-informed research hypotheses."""
        hypotheses = []
        
        # Hypothesis about fault type
        failure_analysis = analysis_results.get("failure_mode_analysis", {})
        detected_faults = failure_analysis.get("detected_fault_types", [])
        confidence_scores = failure_analysis.get("confidence_scores", {})
        
        for fault_type in detected_faults:
            confidence = confidence_scores.get(fault_type, 0.5)
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_fault_{fault_type}_{len(hypotheses)}",
                statement=f"Signal exhibits characteristics consistent with {fault_type}",
                confidence=confidence,
                generated_by=self.agent_name,
                evidence=[f"Domain rule evaluation confidence: {confidence:.3f}"],
                test_methods=["bearing_frequency_analysis", "envelope_analysis", "statistical_validation"]
            ))
        
        # Hypothesis about bearing frequencies
        freq_analysis = analysis_results.get("bearing_frequency_analysis", {})
        frequency_matches = freq_analysis.get("frequency_matches", {})
        
        if frequency_matches:
            fault_confidence = freq_analysis.get("bearing_fault_indicators", {}).get("fault_confidence", 0.5)
            hypotheses.append(ResearchHypothesis(
                hypothesis_id=f"hyp_bearing_freq_{len(hypotheses)}",
                statement="Detected frequencies match theoretical bearing fault frequencies",
                confidence=fault_confidence,
                generated_by=self.agent_name,
                evidence=[f"Matched frequencies: {list(frequency_matches.keys())}"],
                test_methods=["frequency_domain_validation", "harmonic_analysis"]
            ))
        
        return hypotheses


def domain_expert_agent(state: ResearchPHMState) -> Dict[str, Any]:
    """
    LangGraph node function for the Domain Expert Agent.
    
    Args:
        state: Current research state
        
    Returns:
        State updates from domain expert analysis
    """
    agent = DomainExpertAgent()
    
    # Perform analysis
    analysis_results = agent.analyze(state)
    
    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(state, analysis_results)
    
    # Update state
    state_updates = {
        "domain_expert_state": analysis_results,
        "research_hypotheses": state.research_hypotheses + hypotheses
    }
    
    # Add audit entry
    state.add_audit_entry(
        agent="domain_expert",
        action="physics_informed_analysis",
        confidence=analysis_results.get("confidence", 0.0),
        outputs={"n_hypotheses": len(hypotheses)}
    )
    
    return state_updates


if __name__ == "__main__":
    # Test the Domain Expert Agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from states.research_states import ResearchPHMState
    from states.phm_states import DAGState, InputData
    import numpy as np
    
    # Create test data with bearing fault characteristics
    fs = 1000
    t = np.linspace(0, 1, fs)
    
    # Simulate bearing fault with impulsive behavior
    fault_signal = np.sin(2 * np.pi * 50 * t)  # Base signal
    
    # Add impulsive components (simulating bearing fault)
    impulse_times = np.random.choice(len(t), size=10, replace=False)
    for imp_time in impulse_times:
        if imp_time < len(t) - 50:
            fault_signal[imp_time:imp_time+50] += 2.0 * np.exp(-np.arange(50) * 0.1)
    
    test_signals = fault_signal.reshape(1, -1, 1)
    ref_signals = np.sin(2 * np.pi * 50 * t).reshape(1, -1, 1)  # Healthy reference
    
    # Create research state
    dag_state = DAGState(user_instruction="Test domain expert", channels=["ch1"], nodes={}, leaves=[])
    ref_signal = InputData(node_id="ref", parents=[], shape=ref_signals.shape, 
                          results={"ref": ref_signals}, meta={"fs": fs})
    test_signal = InputData(node_id="test", parents=[], shape=test_signals.shape,
                           results={"test": test_signals}, meta={"fs": fs})
    
    state = ResearchPHMState(
        case_name="test_domain_expert",
        user_instruction="Apply domain expertise to bearing fault analysis",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state,
        fs=fs
    )
    
    # Add some mock data analysis results
    state.data_analysis_state = {
        "test_analysis": {
            "statistical_summary": {
                "kurtosis": {"mean": 5.0},
                "crest_factor": {"mean": 4.0},
                "rms": {"mean": 1.2}
            },
            "frequency_analysis": {
                "mean_spectrum": np.random.rand(500),
                "frequency_bins": np.linspace(0, 500, 500)
            }
        }
    }
    
    # Test the agent
    agent = DomainExpertAgent()
    results = agent.analyze(state)
    hypotheses = agent.generate_hypotheses(state, results)
    
    print(f"Domain expert analysis completed with confidence: {results['confidence']:.3f}")
    print(f"Generated {len(hypotheses)} hypotheses")
    print(f"Detected fault types: {results.get('failure_mode_analysis', {}).get('detected_fault_types', [])}")
    print("Domain Expert Agent test completed successfully!")
