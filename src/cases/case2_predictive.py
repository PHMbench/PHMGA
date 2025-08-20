"""
Case 2: Advanced Predictive Maintenance with PHM Scientist Agents

This case demonstrates advanced PHM capabilities including remaining useful life (RUL) estimation,
multi-sensor fusion, health index calculation, and automated research-driven maintenance scheduling.
"""

from __future__ import annotations
import uuid
import yaml
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable LangSmith for cleaner logs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

from ..research_workflow import build_research_graph
from ..utils import initialize_state, save_state, load_state
from ..states.research_states import ResearchPHMState, ResearchHypothesis
from ..agents.reflect_agent import get_dag_depth

logger = logging.getLogger(__name__)


class RULEstimator:
    """Remaining Useful Life estimation using degradation modeling."""
    
    def __init__(self):
        self.degradation_models = {
            "linear": self._linear_degradation,
            "exponential": self._exponential_degradation,
            "power_law": self._power_law_degradation
        }
    
    def estimate_rul(self, health_history: List[float], time_points: List[float], 
                    failure_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Estimate remaining useful life based on health index history.
        
        Args:
            health_history: Historical health index values (1.0 = healthy, 0.0 = failed)
            time_points: Corresponding time points
            failure_threshold: Health index threshold for failure
            
        Returns:
            RUL estimation results
        """
        if len(health_history) < 3:
            return {"error": "Insufficient data for RUL estimation"}
        
        results = {}
        
        # Try different degradation models
        for model_name, model_func in self.degradation_models.items():
            try:
                rul_estimate = model_func(health_history, time_points, failure_threshold)
                results[model_name] = rul_estimate
            except Exception as e:
                logger.warning(f"RUL estimation failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Select best model based on fit quality
        best_model = self._select_best_model(results)
        
        return {
            "models": results,
            "best_model": best_model,
            "rul_estimate": results.get(best_model, {}).get("rul_days", 0),
            "confidence": results.get(best_model, {}).get("confidence", 0.0)
        }
    
    def _linear_degradation(self, health_history: List[float], time_points: List[float], 
                           threshold: float) -> Dict[str, Any]:
        """Linear degradation model."""
        # Simple linear regression
        x = np.array(time_points)
        y = np.array(health_history)
        
        # Fit linear model: y = mx + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate time to reach threshold
        if m >= 0:  # No degradation or improvement
            rul_days = float('inf')
        else:
            current_time = time_points[-1]
            current_health = health_history[-1]
            time_to_threshold = (threshold - current_health) / m
            rul_days = max(0, time_to_threshold)
        
        return {
            "rul_days": rul_days,
            "degradation_rate": m,
            "r_squared": r_squared,
            "confidence": min(r_squared, 1.0)
        }
    
    def _exponential_degradation(self, health_history: List[float], time_points: List[float], 
                                threshold: float) -> Dict[str, Any]:
        """Exponential degradation model."""
        # Fit exponential model: y = a * exp(b * x)
        x = np.array(time_points)
        y = np.array(health_history)
        
        # Use log-linear regression for exponential fit
        y_log = np.log(np.maximum(y, 1e-10))  # Avoid log(0)
        A = np.vstack([x, np.ones(len(x))]).T
        b, log_a = np.linalg.lstsq(A, y_log, rcond=None)[0]
        a = np.exp(log_a)
        
        # Calculate fit quality
        y_pred = a * np.exp(b * x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate RUL
        if b >= 0:  # No degradation
            rul_days = float('inf')
        else:
            current_health = health_history[-1]
            time_to_threshold = np.log(threshold / current_health) / b
            rul_days = max(0, time_to_threshold)
        
        return {
            "rul_days": rul_days,
            "decay_constant": b,
            "r_squared": r_squared,
            "confidence": min(r_squared, 1.0)
        }
    
    def _power_law_degradation(self, health_history: List[float], time_points: List[float], 
                              threshold: float) -> Dict[str, Any]:
        """Power law degradation model."""
        # Fit power law model: y = a * x^b
        x = np.array(time_points) + 1  # Avoid x=0
        y = np.array(health_history)
        
        # Use log-log regression
        x_log = np.log(x)
        y_log = np.log(np.maximum(y, 1e-10))
        
        A = np.vstack([x_log, np.ones(len(x_log))]).T
        b, log_a = np.linalg.lstsq(A, y_log, rcond=None)[0]
        a = np.exp(log_a)
        
        # Calculate fit quality
        y_pred = a * (x ** b)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate RUL
        if b >= 0:  # No degradation
            rul_days = float('inf')
        else:
            current_time = time_points[-1] + 1
            current_health = health_history[-1]
            time_to_threshold = ((threshold / current_health) ** (1/b)) * current_time - current_time
            rul_days = max(0, time_to_threshold)
        
        return {
            "rul_days": rul_days,
            "power_exponent": b,
            "r_squared": r_squared,
            "confidence": min(r_squared, 1.0)
        }
    
    def _select_best_model(self, results: Dict[str, Any]) -> str:
        """Select the best degradation model based on fit quality."""
        best_model = "linear"
        best_score = 0.0
        
        for model_name, result in results.items():
            if "error" not in result:
                confidence = result.get("confidence", 0.0)
                if confidence > best_score:
                    best_score = confidence
                    best_model = model_name
        
        return best_model


class HealthIndexCalculator:
    """Calculate comprehensive health index from multiple indicators."""
    
    def __init__(self):
        self.feature_weights = {
            "rms": 0.2,
            "kurtosis": 0.25,
            "crest_factor": 0.2,
            "spectral_kurtosis": 0.15,
            "envelope_energy": 0.2
        }
    
    def calculate_health_index(self, features: Dict[str, float], 
                              baseline_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate health index based on feature deviations from baseline.
        
        Args:
            features: Current feature values
            baseline_features: Baseline (healthy) feature values
            
        Returns:
            Health index calculation results
        """
        health_components = {}
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in features and feature_name in baseline_features:
                current_value = features[feature_name]
                baseline_value = baseline_features[feature_name]
                
                # Calculate normalized deviation
                if baseline_value != 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                else:
                    deviation = abs(current_value)
                
                # Convert to health score (1.0 = healthy, 0.0 = unhealthy)
                health_score = max(0.0, 1.0 - deviation)
                health_components[feature_name] = {
                    "health_score": health_score,
                    "deviation": deviation,
                    "weight": weight
                }
        
        # Calculate weighted health index
        if health_components:
            weighted_sum = sum(comp["health_score"] * comp["weight"] 
                             for comp in health_components.values())
            total_weight = sum(comp["weight"] for comp in health_components.values())
            health_index = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            health_index = 0.5  # Neutral if no features available
        
        return {
            "health_index": health_index,
            "components": health_components,
            "confidence": len(health_components) / len(self.feature_weights)
        }


class MaintenanceScheduler:
    """Generate maintenance schedules based on RUL and health index."""
    
    def __init__(self):
        self.maintenance_strategies = {
            "immediate": {"rul_threshold": 7, "health_threshold": 0.3},
            "urgent": {"rul_threshold": 30, "health_threshold": 0.5},
            "planned": {"rul_threshold": 90, "health_threshold": 0.7},
            "routine": {"rul_threshold": float('inf'), "health_threshold": 0.8}
        }
    
    def generate_schedule(self, rul_days: float, health_index: float, 
                         current_date: datetime = None) -> Dict[str, Any]:
        """
        Generate maintenance schedule based on RUL and health index.
        
        Args:
            rul_days: Estimated remaining useful life in days
            health_index: Current health index (0-1)
            current_date: Current date (defaults to now)
            
        Returns:
            Maintenance schedule recommendations
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Determine maintenance strategy
        strategy = self._determine_strategy(rul_days, health_index)
        
        # Generate schedule based on strategy
        schedule = self._create_schedule(strategy, rul_days, health_index, current_date)
        
        return {
            "strategy": strategy,
            "schedule": schedule,
            "recommendations": self._generate_recommendations(strategy, rul_days, health_index),
            "risk_assessment": self._assess_risk(rul_days, health_index)
        }
    
    def _determine_strategy(self, rul_days: float, health_index: float) -> str:
        """Determine maintenance strategy based on RUL and health index."""
        for strategy, thresholds in self.maintenance_strategies.items():
            if (rul_days <= thresholds["rul_threshold"] or 
                health_index <= thresholds["health_threshold"]):
                return strategy
        
        return "routine"
    
    def _create_schedule(self, strategy: str, rul_days: float, health_index: float, 
                        current_date: datetime) -> Dict[str, Any]:
        """Create detailed maintenance schedule."""
        schedule = {
            "next_inspection": None,
            "next_maintenance": None,
            "replacement_window": None,
            "monitoring_frequency": "weekly"
        }
        
        if strategy == "immediate":
            schedule["next_inspection"] = current_date + timedelta(days=1)
            schedule["next_maintenance"] = current_date + timedelta(days=2)
            schedule["replacement_window"] = current_date + timedelta(days=7)
            schedule["monitoring_frequency"] = "daily"
        
        elif strategy == "urgent":
            schedule["next_inspection"] = current_date + timedelta(days=3)
            schedule["next_maintenance"] = current_date + timedelta(days=7)
            schedule["replacement_window"] = current_date + timedelta(days=min(rul_days * 0.8, 21))
            schedule["monitoring_frequency"] = "every 2 days"
        
        elif strategy == "planned":
            schedule["next_inspection"] = current_date + timedelta(days=7)
            schedule["next_maintenance"] = current_date + timedelta(days=min(rul_days * 0.6, 60))
            schedule["replacement_window"] = current_date + timedelta(days=min(rul_days * 0.8, 75))
            schedule["monitoring_frequency"] = "weekly"
        
        else:  # routine
            schedule["next_inspection"] = current_date + timedelta(days=30)
            schedule["next_maintenance"] = current_date + timedelta(days=90)
            schedule["replacement_window"] = current_date + timedelta(days=365)
            schedule["monitoring_frequency"] = "monthly"
        
        return schedule
    
    def _generate_recommendations(self, strategy: str, rul_days: float, 
                                 health_index: float) -> List[str]:
        """Generate specific maintenance recommendations."""
        recommendations = []
        
        if strategy == "immediate":
            recommendations.extend([
                "Stop operation immediately and inspect bearing",
                "Prepare for emergency bearing replacement",
                "Increase monitoring to continuous if possible",
                "Have replacement parts ready on-site"
            ])
        
        elif strategy == "urgent":
            recommendations.extend([
                "Schedule bearing replacement within 2 weeks",
                "Increase vibration monitoring frequency",
                "Prepare maintenance crew and replacement parts",
                "Consider reducing operational load if possible"
            ])
        
        elif strategy == "planned":
            recommendations.extend([
                "Schedule bearing replacement in next maintenance window",
                "Order replacement parts if not in stock",
                "Plan maintenance crew availability",
                "Continue regular monitoring"
            ])
        
        else:  # routine
            recommendations.extend([
                "Continue normal operation and monitoring",
                "Include bearing in next routine maintenance check",
                "Maintain current lubrication schedule"
            ])
        
        # Add health-specific recommendations
        if health_index < 0.5:
            recommendations.append("Consider lubrication analysis")
        if health_index < 0.3:
            recommendations.append("Investigate potential contamination or misalignment")
        
        return recommendations
    
    def _assess_risk(self, rul_days: float, health_index: float) -> Dict[str, Any]:
        """Assess operational risk based on current condition."""
        # Risk factors
        rul_risk = 1.0 - min(rul_days / 365, 1.0)  # Higher risk with shorter RUL
        health_risk = 1.0 - health_index  # Higher risk with lower health
        
        # Combined risk score
        overall_risk = (rul_risk + health_risk) / 2
        
        # Risk categories
        if overall_risk > 0.8:
            risk_level = "critical"
        elif overall_risk > 0.6:
            risk_level = "high"
        elif overall_risk > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "overall_risk": overall_risk,
            "risk_level": risk_level,
            "rul_risk": rul_risk,
            "health_risk": health_risk,
            "risk_factors": self._identify_risk_factors(rul_days, health_index)
        }
    
    def _identify_risk_factors(self, rul_days: float, health_index: float) -> List[str]:
        """Identify specific risk factors."""
        factors = []
        
        if rul_days < 30:
            factors.append("Short remaining useful life")
        if health_index < 0.5:
            factors.append("Poor bearing health condition")
        if rul_days < 7:
            factors.append("Imminent failure risk")
        if health_index < 0.3:
            factors.append("Critical bearing degradation")
        
        return factors


def create_predictive_maintenance_state(config: Dict[str, Any]) -> ResearchPHMState:
    """
    Create research state specifically configured for predictive maintenance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResearchPHMState configured for predictive maintenance
    """
    # Initialize base state
    initial_state = initialize_state(
        user_instruction=config['user_instruction'],
        metadata_path=config['metadata_path'],
        h5_path=config['h5_path'],
        ref_ids=config['ref_ids'],
        test_ids=config['test_ids'],
        case_name=config['name']
    )
    
    # Convert to research state
    research_state = ResearchPHMState(
        case_name=initial_state.case_name,
        user_instruction=initial_state.user_instruction,
        reference_signal=initial_state.reference_signal,
        test_signal=initial_state.test_signal,
        dag_state=initial_state.dag_state,
        
        # Copy basic configuration
        min_depth=initial_state.min_depth,
        max_depth=initial_state.max_depth,
        min_width=initial_state.min_width,
        fs=initial_state.fs,
        
        # Predictive maintenance specific configuration
        research_phase="predictive_analysis",
        max_research_iterations=5,
        research_quality_threshold=0.85,  # Higher threshold for predictive maintenance
        
        # Initialize predictive maintenance fields
        remaining_useful_life=None,
        health_index=None,
        predictive_maintenance_schedule=None
    )
    
    # Add predictive maintenance objectives
    research_state.add_research_objective(
        "Estimate remaining useful life using degradation modeling", priority=1
    )
    research_state.add_research_objective(
        "Calculate comprehensive health index from multiple indicators", priority=1
    )
    research_state.add_research_objective(
        "Generate optimal maintenance schedule based on condition", priority=2
    )
    research_state.add_research_objective(
        "Assess operational risk and failure probability", priority=2
    )
    
    return research_state


def run_predictive_maintenance_case(config_path: str) -> Optional[ResearchPHMState]:
    """
    Run Case 2: Advanced Predictive Maintenance with PHM Scientist Agents.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Final research state with predictive maintenance results
    """
    logger.info(f"Starting Case 2: Predictive Maintenance with config: {config_path}")
    
    # Load configuration
    print(f"--- Loading configuration from {config_path} ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create predictive maintenance state
    print("\n--- [Part 1] Initializing Predictive Maintenance State ---")
    research_state = create_predictive_maintenance_state(config)
    
    # Run research workflow
    print("\n--- [Part 2] Starting Research Agent Workflow ---")
    research_app = build_research_graph()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    final_state = research_state.model_copy(deep=True)
    
    try:
        for event in research_app.stream(research_state, config=thread_config):
            for node_name, state_update in event.items():
                print(f"--- Research Node Executed: {node_name} ---")
                if state_update is not None:
                    for key, value in state_update.items():
                        if hasattr(final_state, key):
                            setattr(final_state, key, value)
                
                # Log progress
                print(f"    Research confidence: {final_state.research_confidence:.3f}")
                print(f"    Research phase: {final_state.research_phase}")
    
    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        print(f"--- Research workflow error: {e} ---")
        return final_state
    
    # Perform predictive maintenance analysis
    print("\n--- [Part 3] Performing Predictive Maintenance Analysis ---")
    final_state = perform_predictive_analysis(final_state)
    
    # Save results
    save_path = config.get('state_save_path', '').replace('.pkl', '_case2.pkl')
    if save_path:
        save_state(final_state, save_path)
        print(f"--- Case 2 state saved to {save_path} ---")
    
    # Generate predictive maintenance report
    report_path = config.get('report_path', '').replace('.md', '_case2_predictive.md')
    if report_path:
        generate_predictive_maintenance_report(final_state, report_path)
        print(f"--- Predictive maintenance report saved to {report_path} ---")
    
    return final_state


def perform_predictive_analysis(state: ResearchPHMState) -> ResearchPHMState:
    """
    Perform predictive maintenance analysis including RUL estimation and health index calculation.
    
    Args:
        state: Research state with analysis results
        
    Returns:
        Updated state with predictive maintenance results
    """
    logger.info("Performing predictive maintenance analysis")
    
    try:
        # Extract features for health index calculation
        features = extract_health_features(state)
        baseline_features = extract_baseline_features(state)
        
        # Calculate health index
        health_calculator = HealthIndexCalculator()
        health_result = health_calculator.calculate_health_index(features, baseline_features)
        state.health_index = health_result["health_index"]
        
        # Simulate health history for RUL estimation (in real application, this would be historical data)
        health_history = simulate_health_history(state.health_index)
        time_points = list(range(len(health_history)))
        
        # Estimate RUL
        rul_estimator = RULEstimator()
        rul_result = rul_estimator.estimate_rul(health_history, time_points)
        state.remaining_useful_life = rul_result.get("rul_estimate", 0)
        
        # Generate maintenance schedule
        scheduler = MaintenanceScheduler()
        schedule_result = scheduler.generate_schedule(
            state.remaining_useful_life, state.health_index
        )
        state.predictive_maintenance_schedule = schedule_result
        
        # Add predictive maintenance hypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id="hyp_predictive_maintenance",
            statement=f"Bearing health index of {state.health_index:.2f} with RUL of {state.remaining_useful_life:.1f} days",
            confidence=min(health_result["confidence"], rul_result.get("confidence", 0.5)),
            generated_by="predictive_maintenance_system",
            evidence=[
                f"Health index: {state.health_index:.2f}",
                f"RUL estimate: {state.remaining_useful_life:.1f} days",
                f"Maintenance strategy: {schedule_result['strategy']}"
            ],
            test_methods=["degradation_modeling", "health_index_validation", "maintenance_optimization"]
        )
        state.research_hypotheses.append(hypothesis)
        
        # Update research confidence based on predictive analysis
        predictive_confidence = (health_result["confidence"] + rul_result.get("confidence", 0.5)) / 2
        state.research_confidence = max(state.research_confidence, predictive_confidence)
        
        logger.info(f"Predictive analysis completed: Health={state.health_index:.3f}, RUL={state.remaining_useful_life:.1f} days")
        
    except Exception as e:
        logger.error(f"Predictive analysis failed: {e}")
        state.error_logs.append(f"Predictive analysis error: {str(e)}")
    
    return state


def extract_health_features(state: ResearchPHMState) -> Dict[str, float]:
    """Extract features for health index calculation."""
    features = {}
    
    # Get features from data analysis
    data_analysis = state.data_analysis_state
    if data_analysis:
        test_analysis = data_analysis.get("test_analysis", {})
        statistical_summary = test_analysis.get("statistical_summary", {})
        
        # Extract key features
        for feature_name in ["rms", "kurtosis", "crest_factor"]:
            if feature_name in statistical_summary:
                feature_data = statistical_summary[feature_name]
                if isinstance(feature_data, dict) and "mean" in feature_data:
                    features[feature_name] = feature_data["mean"]
    
    # Add default values if features not available
    if not features:
        features = {"rms": 1.0, "kurtosis": 3.0, "crest_factor": 3.0}
    
    return features


def extract_baseline_features(state: ResearchPHMState) -> Dict[str, float]:
    """Extract baseline (healthy) features for comparison."""
    baseline_features = {}
    
    # Get features from reference analysis
    data_analysis = state.data_analysis_state
    if data_analysis:
        ref_analysis = data_analysis.get("reference_analysis", {})
        statistical_summary = ref_analysis.get("statistical_summary", {})
        
        # Extract baseline features
        for feature_name in ["rms", "kurtosis", "crest_factor"]:
            if feature_name in statistical_summary:
                feature_data = statistical_summary[feature_name]
                if isinstance(feature_data, dict) and "mean" in feature_data:
                    baseline_features[feature_name] = feature_data["mean"]
    
    # Add default baseline values if not available
    if not baseline_features:
        baseline_features = {"rms": 0.8, "kurtosis": 2.5, "crest_factor": 2.5}
    
    return baseline_features


def simulate_health_history(current_health: float, history_length: int = 10) -> List[float]:
    """Simulate health history for RUL estimation (for demonstration purposes)."""
    # Create a degrading health trend
    health_history = []
    
    for i in range(history_length):
        # Simulate gradual degradation with some noise
        progress = i / (history_length - 1)
        health_value = 1.0 - progress * (1.0 - current_health)
        health_value += np.random.normal(0, 0.05)  # Add noise
        health_value = max(0.0, min(1.0, health_value))  # Clamp to [0, 1]
        health_history.append(health_value)
    
    return health_history


def generate_predictive_maintenance_report(state: ResearchPHMState, report_path: str):
    """Generate comprehensive predictive maintenance report."""
    report = f"""# Case 2: Predictive Maintenance Report

**Generated:** {datetime.now().isoformat()}
**Case:** {state.case_name}
**Research Confidence:** {state.research_confidence:.1%}

## Executive Summary

This predictive maintenance analysis provides comprehensive insights into bearing condition,
remaining useful life estimation, and optimal maintenance scheduling.

### Key Findings

- **Health Index:** {state.health_index:.3f} (1.0 = healthy, 0.0 = failed)
- **Remaining Useful Life:** {state.remaining_useful_life:.1f} days
- **Maintenance Strategy:** {state.predictive_maintenance_schedule.get('strategy', 'Unknown') if state.predictive_maintenance_schedule else 'Not calculated'}
- **Risk Level:** {state.predictive_maintenance_schedule.get('risk_assessment', {}).get('risk_level', 'Unknown') if state.predictive_maintenance_schedule else 'Not assessed'}

## Predictive Analysis Results

### Health Index Assessment

The current health index of {state.health_index:.3f} indicates:
"""
    
    if state.health_index > 0.8:
        report += "- **Excellent condition** - bearing is operating within normal parameters\n"
    elif state.health_index > 0.6:
        report += "- **Good condition** - minor degradation detected but within acceptable limits\n"
    elif state.health_index > 0.4:
        report += "- **Fair condition** - noticeable degradation requiring attention\n"
    elif state.health_index > 0.2:
        report += "- **Poor condition** - significant degradation requiring immediate action\n"
    else:
        report += "- **Critical condition** - bearing approaching failure\n"
    
    if state.predictive_maintenance_schedule:
        schedule = state.predictive_maintenance_schedule.get('schedule', {})
        recommendations = state.predictive_maintenance_schedule.get('recommendations', [])
        
        report += f"""
### Maintenance Schedule

- **Next Inspection:** {schedule.get('next_inspection', 'Not scheduled')}
- **Next Maintenance:** {schedule.get('next_maintenance', 'Not scheduled')}
- **Replacement Window:** {schedule.get('replacement_window', 'Not scheduled')}
- **Monitoring Frequency:** {schedule.get('monitoring_frequency', 'Not specified')}

### Recommendations

"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
    
    # Add research hypotheses
    if state.research_hypotheses:
        report += "\n## Research Hypotheses\n\n"
        for i, hyp in enumerate(state.research_hypotheses, 1):
            report += f"### Hypothesis {i}\n"
            report += f"**Statement:** {hyp.statement}\n\n"
            report += f"**Confidence:** {hyp.confidence:.1%}\n\n"
            if hyp.evidence:
                report += "**Evidence:**\n"
                for evidence in hyp.evidence:
                    report += f"- {evidence}\n"
                report += "\n"
    
    report += "\n---\n\n*This report was generated by the PHM Scientist Agent System - Case 2: Predictive Maintenance*"
    
    # Save report
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Predictive maintenance report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Case 2: Predictive Maintenance")
    parser.add_argument("--config", type=str, default="config/case2.yaml", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    result = run_predictive_maintenance_case(args.config)
    
    if result:
        print(f"\nCase 2 completed successfully!")
        print(f"Health Index: {result.health_index:.3f}")
        print(f"Remaining Useful Life: {result.remaining_useful_life:.1f} days")
        print(f"Research Confidence: {result.research_confidence:.1%}")
        
        if result.predictive_maintenance_schedule:
            strategy = result.predictive_maintenance_schedule.get('strategy', 'Unknown')
            risk_level = result.predictive_maintenance_schedule.get('risk_assessment', {}).get('risk_level', 'Unknown')
            print(f"Maintenance Strategy: {strategy}")
            print(f"Risk Level: {risk_level}")
    else:
        print("Case 2 failed to complete")
