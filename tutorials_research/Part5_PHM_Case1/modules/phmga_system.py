"""
Complete PHMGA System Integration

Integrates all tutorial components (Parts 1-4) into a production-ready
Prognostics and Health Management Graph Agent system.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Import foundation components from previous parts
sys.path.append('../../Part1_Foundations/modules')
sys.path.append('../../Part2_Multi_Agent_Router/modules')  
sys.path.append('../../Part3_Gemini_Research_Agent/modules')
sys.path.append('../../Part4_DAG_Architecture/modules')

# Core foundation imports
from llm_providers import create_research_llm, LLMProvider

# Multi-agent router imports
from agent_router import AgentRouter, RouterConfig
from research_agents import ArxivAgent, SemanticScholarAgent, CrossRefAgent

# Research workflow imports
from research_graph import ResearchWorkflowGraph
from state_schemas import ResearchConfiguration, create_initial_research_state

# DAG processing imports
from phm_dag_structure import PHMSignalProcessingDAG, SignalProcessingNode
from dag_fundamentals import ResearchDAG, DAGNode, NodeType


@dataclass
class PHMGAConfig:
    """Configuration for the complete PHMGA system"""
    
    # LLM Configuration (Part 1)
    llm_provider: LLMProvider = LLMProvider.AUTO
    llm_temperature: float = 0.3
    llm_model: str = "default"
    
    # Agent Router Configuration (Part 2) 
    enable_arxiv_agent: bool = True
    enable_semantic_scholar: bool = True
    enable_crossref_agent: bool = True
    max_concurrent_agents: int = 3
    
    # Research Configuration (Part 3)
    research_enabled: bool = True
    max_research_loops: int = 2
    research_quality_threshold: float = 0.8
    
    # DAG Processing Configuration (Part 4)
    enable_parallel_processing: bool = True
    max_parallel_nodes: int = 8
    dag_execution_timeout: int = 300
    
    # PHM Specific Configuration
    signal_sampling_rate: int = 10000  # 10 kHz default
    fault_classes: List[str] = field(default_factory=lambda: ["normal", "inner_race", "outer_race", "ball"])
    confidence_threshold: float = 0.7
    alert_threshold: float = 0.8
    
    # Production Settings
    real_time_mode: bool = False
    batch_processing_size: int = 100
    logging_level: str = "INFO"
    enable_monitoring: bool = True
    
    @classmethod
    def for_production(cls) -> "PHMGAConfig":
        """Create production-optimized configuration"""
        return cls(
            llm_provider=LLMProvider.AUTO,
            enable_parallel_processing=True,
            max_parallel_nodes=12,
            real_time_mode=True,
            confidence_threshold=0.8,
            alert_threshold=0.9,
            logging_level="WARNING"
        )
    
    @classmethod
    def for_research(cls) -> "PHMGAConfig":
        """Create research-optimized configuration"""
        return cls(
            research_enabled=True,
            max_research_loops=3,
            research_quality_threshold=0.9,
            llm_temperature=0.1,  # Lower temperature for consistency
            logging_level="DEBUG"
        )
    
    @classmethod
    def for_tutorial(cls) -> "PHMGAConfig":
        """Create tutorial-friendly configuration"""
        return cls(
            enable_parallel_processing=False,  # Easier to follow
            max_research_loops=1,
            batch_processing_size=10,
            logging_level="INFO"
        )


class PHMGASystem:
    """
    Complete Prognostics and Health Management Graph Agent System.
    
    Integrates all tutorial components into a unified system for
    bearing fault diagnosis and continuous research integration.
    """
    
    def __init__(self, config: PHMGAConfig):
        self.config = config
        self.session_id = f"phmga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize system components
        self._initialize_llm_system()        # Part 1: LLM Foundation
        self._initialize_agent_router()      # Part 2: Multi-Agent Router
        self._initialize_research_system()   # Part 3: Research Integration
        self._initialize_dag_processor()     # Part 4: DAG Processing
        
        # PHM-specific components
        self.fault_detection_results = {}
        self.research_knowledge_base = {}
        self.performance_metrics = {}
        
        # System state tracking
        self.system_status = "initialized"
        self.last_update = datetime.now()
        self.processing_statistics = {
            "total_signals_processed": 0,
            "total_faults_detected": 0,
            "average_processing_time": 0.0,
            "system_uptime": 0.0
        }
        
        print(f"🏭 PHMGA System initialized with session ID: {self.session_id}")
    
    def _initialize_llm_system(self):
        """Initialize LLM foundation from Part 1"""
        try:
            self.research_llm = create_research_llm(
                provider=self.config.llm_provider,
                temperature=self.config.llm_temperature,
                model=self.config.llm_model
            )
            print("✅ LLM System initialized (Part 1 Foundation)")
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            self.research_llm = None
    
    def _initialize_agent_router(self):
        """Initialize multi-agent router from Part 2"""
        try:
            # Configure research agents
            router_config = RouterConfig(
                enable_parallel_execution=True,
                max_concurrent_agents=self.config.max_concurrent_agents,
                timeout_seconds=30
            )
            
            self.agent_router = AgentRouter(self.research_llm, router_config)
            
            # Register available agents
            if self.config.enable_arxiv_agent:
                arxiv_agent = ArxivAgent()
                self.agent_router.register_agent("arxiv", arxiv_agent)
            
            if self.config.enable_semantic_scholar:
                scholar_agent = SemanticScholarAgent()
                self.agent_router.register_agent("semantic_scholar", scholar_agent)
            
            if self.config.enable_crossref_agent:
                crossref_agent = CrossRefAgent()
                self.agent_router.register_agent("crossref", crossref_agent)
            
            print(f"✅ Multi-Agent Router initialized with {len(self.agent_router.agents)} agents (Part 2)")
            
        except Exception as e:
            print(f"⚠️ Agent Router initialization failed: {e}")
            self.agent_router = None
    
    def _initialize_research_system(self):
        """Initialize research integration from Part 3"""
        if not self.config.research_enabled or not self.research_llm:
            self.research_system = None
            return
        
        try:
            # Configure research system
            research_config = ResearchConfiguration(
                llm_provider="auto",
                max_research_loops=self.config.max_research_loops,
                coverage_threshold=self.config.research_quality_threshold
            )
            
            self.research_system = ResearchWorkflowGraph(self.research_llm, research_config)
            print("✅ Research Integration System initialized (Part 3)")
            
        except Exception as e:
            print(f"⚠️ Research system initialization failed: {e}")
            self.research_system = None
    
    def _initialize_dag_processor(self):
        """Initialize DAG processing system from Part 4"""
        try:
            self.dag_processor = PHMSignalProcessingDAG("integrated_fault_diagnosis")
            
            # Configure for production settings
            if self.config.enable_parallel_processing:
                self.dag_processor.parallel_enabled = True
                
            print(f"✅ DAG Processing System initialized with {len(self.dag_processor.nodes)} nodes (Part 4)")
            
        except Exception as e:
            print(f"⚠️ DAG processor initialization failed: {e}")
            self.dag_processor = None
    
    def diagnose_bearing_faults(self, signal_data: np.ndarray, 
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete bearing fault diagnosis using integrated PHMGA system.
        
        Args:
            signal_data: Input vibration signal data
            metadata: Additional signal metadata
            
        Returns:
            Comprehensive diagnosis results
        """
        
        diagnosis_start = time.time()
        
        # Validate inputs
        if signal_data is None or signal_data.size == 0:
            return {"error": "Invalid signal data provided"}
        
        print(f"🔍 Starting bearing fault diagnosis...")
        print(f"   📊 Signal shape: {signal_data.shape}")
        print(f"   🕒 Analysis timestamp: {datetime.now().isoformat()}")
        
        diagnosis_results = {
            "session_id": self.session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "signal_metadata": metadata or {},
            "processing_stages": {}
        }
        
        try:
            # Stage 1: Signal Processing via DAG (Part 4)
            if self.dag_processor:
                print("   🕸️ Stage 1: DAG-based signal processing...")
                stage1_start = time.time()
                
                # Prepare signal data for DAG processing
                initial_inputs = {
                    "raw_signal": signal_data,
                    "sampling_rate": metadata.get("sampling_rate", self.config.signal_sampling_rate),
                    "signal_metadata": metadata
                }
                
                dag_results = self.dag_processor.execute(initial_inputs)
                
                diagnosis_results["processing_stages"]["dag_processing"] = {
                    "execution_time": time.time() - stage1_start,
                    "nodes_executed": len([r for r in dag_results.values() if r is not None]),
                    "processing_successful": "diagnosis_output" in dag_results
                }
                
                # Extract diagnosis from DAG results
                if "diagnosis_output" in dag_results:
                    dag_diagnosis = dag_results["diagnosis_output"]
                    diagnosis_results["primary_diagnosis"] = dag_diagnosis
                    
                    print(f"      ✅ DAG processing completed in {diagnosis_results['processing_stages']['dag_processing']['execution_time']:.2f}s")
                else:
                    print(f"      ⚠️ DAG processing incomplete")
            
            # Stage 2: Research Enhancement (Part 3)
            if self.research_system and self.config.research_enabled:
                print("   🔬 Stage 2: Research-enhanced analysis...")
                stage2_start = time.time()
                
                # Generate research question based on preliminary diagnosis
                primary_diag = diagnosis_results.get("primary_diagnosis", {})
                if "diagnoses" in primary_diag and primary_diag["diagnoses"]:
                    main_fault = primary_diag["diagnoses"][0].get("fault_type", "bearing fault")
                    research_question = f"Recent advances in {main_fault} detection and diagnosis methods"
                    
                    try:
                        research_results = self.research_system.conduct_research(research_question)
                        
                        diagnosis_results["processing_stages"]["research_enhancement"] = {
                            "execution_time": time.time() - stage2_start,
                            "research_question": research_question,
                            "knowledge_updates": len(research_results.get("sources_found", [])),
                            "confidence_improvement": research_results.get("confidence_score", 0.0)
                        }
                        
                        print(f"      ✅ Research enhancement completed in {diagnosis_results['processing_stages']['research_enhancement']['execution_time']:.2f}s")
                        
                    except Exception as e:
                        print(f"      ⚠️ Research enhancement failed: {e}")
                        diagnosis_results["processing_stages"]["research_enhancement"] = {
                            "execution_time": time.time() - stage2_start,
                            "error": str(e)
                        }
            
            # Stage 3: Multi-Agent Validation (Part 2)
            if self.agent_router:
                print("   🤖 Stage 3: Multi-agent validation...")
                stage3_start = time.time()
                
                # Route validation task to appropriate agents
                validation_query = {
                    "task_type": "validation",
                    "diagnosis_results": diagnosis_results.get("primary_diagnosis", {}),
                    "signal_characteristics": {
                        "shape": signal_data.shape,
                        "sampling_rate": metadata.get("sampling_rate", self.config.signal_sampling_rate)
                    }
                }
                
                try:
                    agent_results = self.agent_router.route_task(validation_query)
                    
                    diagnosis_results["processing_stages"]["agent_validation"] = {
                        "execution_time": time.time() - stage3_start,
                        "agents_consulted": len(agent_results.get("agent_responses", {})),
                        "validation_successful": agent_results.get("routing_successful", False)
                    }
                    
                    print(f"      ✅ Agent validation completed in {diagnosis_results['processing_stages']['agent_validation']['execution_time']:.2f}s")
                    
                except Exception as e:
                    print(f"      ⚠️ Agent validation failed: {e}")
                    diagnosis_results["processing_stages"]["agent_validation"] = {
                        "execution_time": time.time() - stage3_start,
                        "error": str(e)
                    }
            
            # Stage 4: Final Assessment and Recommendations
            print("   📋 Stage 4: Final assessment...")
            stage4_start = time.time()
            
            final_assessment = self._generate_final_assessment(diagnosis_results)
            diagnosis_results["final_assessment"] = final_assessment
            
            diagnosis_results["processing_stages"]["final_assessment"] = {
                "execution_time": time.time() - stage4_start
            }
            
            # Update system statistics
            self.processing_statistics["total_signals_processed"] += 1
            if final_assessment.get("fault_detected", False):
                self.processing_statistics["total_faults_detected"] += 1
            
            total_time = time.time() - diagnosis_start
            self.processing_statistics["average_processing_time"] = (
                (self.processing_statistics["average_processing_time"] * 
                 (self.processing_statistics["total_signals_processed"] - 1) + total_time) /
                self.processing_statistics["total_signals_processed"]
            )
            
            diagnosis_results["total_processing_time"] = total_time
            
            print(f"✅ Complete diagnosis finished in {total_time:.2f}s")
            
            return diagnosis_results
            
        except Exception as e:
            print(f"❌ Diagnosis failed with error: {e}")
            diagnosis_results["error"] = str(e)
            diagnosis_results["total_processing_time"] = time.time() - diagnosis_start
            return diagnosis_results
    
    def _generate_final_assessment(self, diagnosis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final integrated assessment"""
        
        assessment = {
            "fault_detected": False,
            "fault_type": "normal",
            "confidence": 0.0,
            "severity": "low",
            "recommendations": [],
            "system_integration_score": 0.0
        }
        
        # Extract primary diagnosis
        primary_diag = diagnosis_results.get("primary_diagnosis", {})
        if "diagnoses" in primary_diag and primary_diag["diagnoses"]:
            main_diagnosis = primary_diag["diagnoses"][0]
            
            assessment["fault_type"] = main_diagnosis.get("fault_type", "normal")
            assessment["confidence"] = main_diagnosis.get("confidence", 0.0)
            assessment["severity"] = main_diagnosis.get("severity", "low")
            assessment["fault_detected"] = assessment["fault_type"] != "normal"
            
            # Generate recommendations based on fault type
            if assessment["fault_detected"]:
                if assessment["confidence"] > self.config.alert_threshold:
                    assessment["recommendations"].append("Immediate maintenance required")
                elif assessment["confidence"] > self.config.confidence_threshold:
                    assessment["recommendations"].append("Schedule maintenance inspection")
                else:
                    assessment["recommendations"].append("Continue monitoring")
        
        # Calculate system integration score
        stages_completed = len([s for s in diagnosis_results.get("processing_stages", {}).values() 
                              if "error" not in s])
        total_stages = len(diagnosis_results.get("processing_stages", {}))
        
        if total_stages > 0:
            assessment["system_integration_score"] = stages_completed / total_stages
        
        return assessment
    
    def run_case_study(self, case_name: str = "case1_bearing_faults") -> Dict[str, Any]:
        """
        Run complete case study demonstration.
        
        Args:
            case_name: Name of the case study to run
            
        Returns:
            Complete case study results
        """
        
        print(f"🏭 RUNNING PHMGA CASE STUDY: {case_name}")
        print("=" * 60)
        
        case_start = time.time()
        
        case_results = {
            "case_name": case_name,
            "system_config": self.config.__dict__,
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "signal_analyses": [],
            "system_performance": {},
            "research_insights": {},
            "validation_results": {}
        }
        
        try:
            # Generate synthetic bearing fault signals for demonstration
            print("\\n📡 Generating synthetic bearing fault signals...")
            test_signals = self._generate_test_signals()
            
            print(f"   Generated {len(test_signals)} test signals:")
            for i, (signal, metadata) in enumerate(test_signals):
                print(f"   • Signal {i+1}: {metadata['fault_type']} ({metadata['signal_length']} samples)")
            
            # Process each test signal
            print("\\n🔍 Processing test signals through PHMGA system...")
            
            for i, (signal_data, signal_metadata) in enumerate(test_signals):
                print(f"\\n   📊 Processing Signal {i+1}/{len(test_signals)}...")
                
                # Run complete diagnosis
                diagnosis = self.diagnose_bearing_faults(signal_data, signal_metadata)
                diagnosis["signal_index"] = i + 1
                diagnosis["true_fault_type"] = signal_metadata["fault_type"]
                
                case_results["signal_analyses"].append(diagnosis)
                
                # Show quick results
                final_assessment = diagnosis.get("final_assessment", {})
                detected_fault = final_assessment.get("fault_type", "unknown")
                confidence = final_assessment.get("confidence", 0.0)
                true_fault = signal_metadata["fault_type"]
                
                correct = detected_fault.lower() == true_fault.lower()
                print(f"      🎯 True: {true_fault} | Detected: {detected_fault} | Confidence: {confidence:.2f} | {'✅' if correct else '❌'}\")\n",
            
            # Calculate overall performance metrics
            print("\\n📈 Calculating system performance metrics...")
            performance_metrics = self._calculate_performance_metrics(case_results["signal_analyses"])
            case_results["system_performance"] = performance_metrics
            
            # Generate research insights summary
            if self.research_system:
                print("\\n🔬 Generating research insights summary...")
                research_insights = self._generate_research_insights(case_results["signal_analyses"])
                case_results["research_insights"] = research_insights
            
            # System validation
            print("\\n✅ Running system validation...")
            validation_results = self._validate_system_performance(case_results)
            case_results["validation_results"] = validation_results
            
            case_results["total_case_time"] = time.time() - case_start
            case_results["end_time"] = datetime.now().isoformat()
            
            # Print final summary
            print(f"\\n🎓 CASE STUDY COMPLETED")
            print(f"=" * 40)
            print(f"• Total Time: {case_results['total_case_time']:.2f} seconds")
            print(f"• Signals Processed: {len(case_results['signal_analyses'])}")
            print(f"• Overall Accuracy: {performance_metrics.get('accuracy', 0.0):.1%}")
            print(f"• Average Confidence: {performance_metrics.get('average_confidence', 0.0):.2f}")
            print(f"• System Integration Score: {performance_metrics.get('integration_score', 0.0):.2f}")
            
            return case_results
            
        except Exception as e:
            print(f"❌ Case study failed: {e}")
            case_results["error"] = str(e)
            case_results["total_case_time"] = time.time() - case_start
            return case_results
    
    def _generate_test_signals(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Generate synthetic test signals for different fault types"""
        
        np.random.seed(42)  # Reproducible results
        
        fs = self.config.signal_sampling_rate
        duration = 1.0  # 1 second signals
        t = np.linspace(0, duration, int(fs * duration))
        
        test_signals = []
        
        # Normal condition
        normal_signal = np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))
        test_signals.append((normal_signal, {
            "fault_type": "normal",
            "sampling_rate": fs,
            "signal_length": len(normal_signal),
            "description": "Healthy bearing signal"
        }))
        
        # Inner race fault
        inner_fault_signal = (np.sin(2 * np.pi * 60 * t) + 
                             0.5 * np.sin(2 * np.pi * 157 * t) + 
                             0.1 * np.random.randn(len(t)))
        test_signals.append((inner_fault_signal, {
            "fault_type": "inner_race",
            "sampling_rate": fs,
            "signal_length": len(inner_fault_signal),
            "description": "Inner race fault at 157 Hz"
        }))
        
        # Outer race fault
        outer_fault_signal = (np.sin(2 * np.pi * 60 * t) + 
                             0.4 * np.sin(2 * np.pi * 236 * t) + 
                             0.1 * np.random.randn(len(t)))
        test_signals.append((outer_fault_signal, {
            "fault_type": "outer_race",
            "sampling_rate": fs, 
            "signal_length": len(outer_fault_signal),
            "description": "Outer race fault at 236 Hz"
        }))
        
        # Ball fault
        ball_fault_signal = (np.sin(2 * np.pi * 60 * t) + 
                            0.3 * np.sin(2 * np.pi * 140 * t) + 
                            0.1 * np.random.randn(len(t)))
        test_signals.append((ball_fault_signal, {
            "fault_type": "ball",
            "sampling_rate": fs,
            "signal_length": len(ball_fault_signal), 
            "description": "Ball fault at 140 Hz"
        }))
        
        return test_signals
    
    def _calculate_performance_metrics(self, signal_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall system performance metrics"""
        
        if not signal_analyses:
            return {"error": "No signal analyses to evaluate"}
        
        # Extract predictions and ground truth
        predictions = []
        true_labels = []
        confidences = []
        processing_times = []
        integration_scores = []
        
        for analysis in signal_analyses:
            final_assessment = analysis.get("final_assessment", {})
            
            predicted_fault = final_assessment.get("fault_type", "unknown")
            true_fault = analysis.get("true_fault_type", "unknown")
            confidence = final_assessment.get("confidence", 0.0)
            processing_time = analysis.get("total_processing_time", 0.0)
            integration_score = final_assessment.get("system_integration_score", 0.0)
            
            predictions.append(predicted_fault.lower())
            true_labels.append(true_fault.lower())
            confidences.append(confidence)
            processing_times.append(processing_time)
            integration_scores.append(integration_score)
        
        # Calculate accuracy
        correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct_predictions / len(predictions) if predictions else 0.0
        
        # Calculate average metrics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        avg_integration_score = np.mean(integration_scores) if integration_scores else 0.0
        
        # Fault-specific metrics
        fault_performance = {}
        unique_faults = list(set(true_labels))
        
        for fault in unique_faults:
            fault_indices = [i for i, label in enumerate(true_labels) if label == fault]
            fault_predictions = [predictions[i] for i in fault_indices]
            
            fault_correct = sum(1 for pred in fault_predictions if pred == fault)
            fault_accuracy = fault_correct / len(fault_predictions) if fault_predictions else 0.0
            
            fault_performance[fault] = {
                "accuracy": fault_accuracy,
                "sample_count": len(fault_predictions),
                "correct_predictions": fault_correct
            }
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "integration_score": avg_integration_score,
            "total_samples": len(predictions),
            "correct_predictions": correct_predictions,
            "fault_specific_performance": fault_performance,
            "processing_time_std": np.std(processing_times) if processing_times else 0.0
        }
    
    def _generate_research_insights(self, signal_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from research integration results"""
        
        research_stages = []
        for analysis in signal_analyses:
            research_stage = analysis.get("processing_stages", {}).get("research_enhancement")
            if research_stage and "error" not in research_stage:
                research_stages.append(research_stage)
        
        if not research_stages:
            return {"message": "No research integration data available"}
        
        insights = {
            "research_queries_executed": len(research_stages),
            "average_research_time": np.mean([stage["execution_time"] for stage in research_stages]),
            "total_knowledge_updates": sum(stage.get("knowledge_updates", 0) for stage in research_stages),
            "average_confidence_improvement": np.mean([stage.get("confidence_improvement", 0) for stage in research_stages]),
            "research_topics": [stage.get("research_question", "") for stage in research_stages if stage.get("research_question")]
        }
        
        return insights
    
    def _validate_system_performance(self, case_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall system performance against requirements"""
        
        performance = case_results.get("system_performance", {})
        
        validation = {
            "accuracy_target": 0.8,  # 80% minimum accuracy
            "confidence_target": 0.7,  # 70% minimum confidence
            "processing_time_target": 5.0,  # 5 seconds maximum
            "integration_target": 0.8  # 80% integration score
        }
        
        results = {}
        
        # Accuracy validation
        accuracy = performance.get("accuracy", 0.0)
        results["accuracy_validation"] = {
            "achieved": accuracy,
            "target": validation["accuracy_target"],
            "passed": accuracy >= validation["accuracy_target"],
            "margin": accuracy - validation["accuracy_target"]
        }
        
        # Confidence validation
        avg_confidence = performance.get("average_confidence", 0.0)
        results["confidence_validation"] = {
            "achieved": avg_confidence,
            "target": validation["confidence_target"],
            "passed": avg_confidence >= validation["confidence_target"],
            "margin": avg_confidence - validation["confidence_target"]
        }
        
        # Processing time validation
        avg_time = performance.get("average_processing_time", 0.0)
        results["processing_time_validation"] = {
            "achieved": avg_time,
            "target": validation["processing_time_target"],
            "passed": avg_time <= validation["processing_time_target"],
            "margin": validation["processing_time_target"] - avg_time
        }
        
        # Integration score validation
        integration_score = performance.get("integration_score", 0.0)
        results["integration_validation"] = {
            "achieved": integration_score,
            "target": validation["integration_target"],
            "passed": integration_score >= validation["integration_target"],
            "margin": integration_score - validation["integration_target"]
        }
        
        # Overall validation
        all_passed = all(result["passed"] for result in results.values())
        results["overall_validation"] = {
            "passed": all_passed,
            "passed_checks": sum(1 for result in results.values() if result["passed"]),
            "total_checks": len(results)
        }
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        
        current_time = datetime.now()
        uptime = (current_time - self.last_update).total_seconds()
        
        return {
            "session_id": self.session_id,
            "system_status": self.system_status,
            "uptime_seconds": uptime,
            "last_update": self.last_update.isoformat(),
            "processing_statistics": self.processing_statistics.copy(),
            "component_status": {
                "llm_system": "active" if self.research_llm else "inactive",
                "agent_router": "active" if self.agent_router else "inactive",
                "research_system": "active" if self.research_system else "inactive",
                "dag_processor": "active" if self.dag_processor else "inactive"
            },
            "configuration": self.config.__dict__
        }


def demonstrate_phmga_system():
    """Demonstrate the complete PHMGA system"""
    
    print("🏭 COMPLETE PHMGA SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Create tutorial configuration
    config = PHMGAConfig.for_tutorial()
    
    print("\\n⚙️ System Configuration:")
    print(f"   • LLM Provider: {config.llm_provider}")
    print(f"   • Research Enabled: {config.research_enabled}")
    print(f"   • Parallel Processing: {config.enable_parallel_processing}")
    print(f"   • Fault Classes: {config.fault_classes}")
    
    # Initialize PHMGA system
    print("\\n🚀 Initializing PHMGA System...")
    phmga = PHMGASystem(config)
    
    # Show system status
    status = phmga.get_system_status()
    print(f"\\n📊 System Status:")
    print(f"   • Session ID: {status['session_id']}")
    print(f"   • Component Status: {status['component_status']}")
    print(f"   • System Status: {status['system_status']}")
    
    # Run case study
    print("\\n🔬 Running Case Study...")
    case_results = phmga.run_case_study("tutorial_demonstration")
    
    return case_results


if __name__ == "__main__":
    demonstrate_phmga_system()