"""
Research Pipeline DAG Implementation

Simplified research workflow DAGs focusing on core DAG patterns:
- Multi-stage research processes
- Parallel processing (fan-out/fan-in)  
- Conditional workflows
- Quality gates and validation
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time
import random

# Smart import handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from dag_fundamentals import ResearchDAG, DAGNode, NodeType, ExecutionStatus

# Try to import visualization - optional dependency
try:
    from dag_visualization import DAGVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ====================================
# NUMPY-COMPATIBLE UTILITY FUNCTIONS
# ====================================

def random_randint(low, high):
    """Generate random integer, compatible with numpy.random.randint"""
    if NUMPY_AVAILABLE:
        return np.random.randint(low, high)
    else:
        return random.randint(low, high - 1)  # Python's randint is inclusive


def random_uniform(low, high):
    """Generate random float, compatible with numpy.random.uniform"""
    if NUMPY_AVAILABLE:
        return np.random.uniform(low, high)
    else:
        return random.uniform(low, high)


class ResearchPhase(Enum):
    """Simplified research process phases"""
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    REPORTING = "reporting"


@dataclass
class ResearchConfig:
    """Simple research configuration"""
    topic: str
    keywords: List[str]
    min_sources: int = 5
    quality_threshold: float = 0.7
    
    def __post_init__(self):
        if not self.keywords:
            self.keywords = [self.topic.lower()]


class SimpleResearchDAG(ResearchDAG):
    """
    Simplified research workflow DAG for educational purposes.
    
    Demonstrates: Planning → Data Collection → Analysis → Validation → Report
    """
    
    def __init__(self, config: ResearchConfig):
        super().__init__(f"research_{config.topic.replace(' ', '_')}", 
                        f"Research workflow for {config.topic}")
        self.config = config
        self._build_simple_pipeline()
    
    def _build_simple_pipeline(self):
        """Build a straightforward 5-stage research pipeline"""
        
        # Stage 1: Research Planning
        def planning_operation(inputs):
            return {
                "research_question": f"What are the key developments in {self.config.topic}?",
                "methodology": "systematic review",
                "search_strategy": f"Keywords: {', '.join(self.config.keywords)}"
            }
        
        # Stage 2: Data Collection
        def data_collection_operation(inputs):
            time.sleep(0.4)  # Simulate collection time
            planning = inputs.get('planning', {})
            num_sources = random_randint(self.config.min_sources, self.config.min_sources + 10)
            
            return {
                "sources_found": num_sources,
                "data_quality": random_uniform(0.6, 1.0),
                "collection_method": planning.get("methodology", "unknown"),
                "sources": [f"Source {i+1} on {self.config.topic}" for i in range(min(num_sources, 5))]
            }
        
        # Stage 3: Data Analysis
        def analysis_operation(inputs):
            time.sleep(0.3)
            data = inputs.get('data_collection', {})
            sources = data.get('sources', [])
            
            return {
                "analyzed_sources": len(sources),
                "key_themes": [f"Theme {i+1}" for i in range(3)],
                "findings": [f"Finding from {source}" for source in sources[:3]],
                "confidence": random_uniform(0.7, 0.95)
            }
        
        # Stage 4: Quality Validation
        def validation_operation(inputs):
            analysis = inputs.get('analysis', {})
            data = inputs.get('data_collection', {})
            
            data_quality = data.get('data_quality', 0.5)
            confidence = analysis.get('confidence', 0.5)
            overall_quality = (data_quality + confidence) / 2
            
            return {
                "quality_score": overall_quality,
                "validation_passed": overall_quality >= self.config.quality_threshold,
                "recommendations": "Good quality analysis" if overall_quality >= 0.8 else "Consider more sources"
            }
        
        # Stage 5: Report Generation
        def reporting_operation(inputs):
            analysis = inputs.get('analysis', {})
            validation = inputs.get('validation', {})
            
            return {
                "report_status": "Complete",
                "summary": f"Analysis of {analysis.get('analyzed_sources', 0)} sources completed",
                "quality_certified": validation.get('validation_passed', False),
                "recommendations": validation.get('recommendations', 'No recommendations')
            }
        
        # Create nodes
        nodes = [
            DAGNode("planning", "Research Planning", NodeType.INPUT, planning_operation),
            DAGNode("data_collection", "Data Collection", NodeType.PROCESSING, data_collection_operation),
            DAGNode("analysis", "Data Analysis", NodeType.PROCESSING, analysis_operation),
            DAGNode("validation", "Quality Validation", NodeType.VALIDATION, validation_operation),
            DAGNode("reporting", "Report Generation", NodeType.OUTPUT, reporting_operation)
        ]
        
        # Add nodes to DAG
        for node in nodes:
            self.add_node(node)
        
        # Create linear pipeline
        edges = [
            ("planning", "data_collection"),
            ("data_collection", "analysis"),
            ("analysis", "validation"),
            ("validation", "reporting")
        ]
        
        for from_node, to_node in edges:
            self.add_edge(from_node, to_node)


class ParallelAnalysisDAG(ResearchDAG):
    """
    Demonstrates parallel processing with fan-out/fan-in pattern.
    
    Input → [Statistical Analysis | Content Analysis | Network Analysis] → Synthesis
    """
    
    def __init__(self, dataset_name: str = "research_data"):
        super().__init__(f"parallel_{dataset_name}", f"Parallel analysis of {dataset_name}")
        self.dataset_name = dataset_name
        self._build_parallel_pipeline()
    
    def _build_parallel_pipeline(self):
        """Build parallel analysis pipeline with three analysis branches"""
        
        def input_operation(inputs):
            return {
                "dataset": self.dataset_name,
                "size": random_randint(100, 500),
                "format": "structured_data"
            }
        
        def statistical_analysis(inputs):
            time.sleep(0.3)  # Different processing times
            data_info = inputs.get('input', {})
            return {
                "analysis_type": "statistical",
                "mean_values": [1.2, 3.4, 2.1],
                "correlations": {"var1_var2": 0.75, "var1_var3": 0.45},
                "significance": "p < 0.05"
            }
        
        def content_analysis(inputs):
            time.sleep(0.5)  # Longer processing time
            data_info = inputs.get('input', {})
            return {
                "analysis_type": "content",
                "themes_identified": 4,
                "sentiment_score": 0.65,
                "key_topics": ["topic1", "topic2", "topic3"]
            }
        
        def network_analysis(inputs):
            time.sleep(0.2)  # Shorter processing time
            data_info = inputs.get('input', {})
            return {
                "analysis_type": "network",
                "nodes": 42,
                "edges": 89,
                "clustering_coefficient": 0.34,
                "centrality_measures": {"node_a": 0.8, "node_b": 0.6}
            }
        
        def synthesis_operation(inputs):
            # Combine results from all three analyses
            stats = inputs.get('stats_analysis', {})
            content = inputs.get('content_analysis', {})
            network = inputs.get('network_analysis', {})
            
            return {
                "combined_insights": f"Integrated {len(stats)} + {len(content)} + {len(network)} analysis results",
                "statistical_summary": stats.get('significance', 'N/A'),
                "content_themes": content.get('themes_identified', 0),
                "network_structure": f"{network.get('nodes', 0)} nodes analyzed",
                "overall_score": random_uniform(0.7, 0.9)
            }
        
        # Create nodes
        nodes = [
            DAGNode("input", "Data Input", NodeType.INPUT, input_operation),
            DAGNode("stats_analysis", "Statistical Analysis", NodeType.PROCESSING, statistical_analysis),
            DAGNode("content_analysis", "Content Analysis", NodeType.PROCESSING, content_analysis),
            DAGNode("network_analysis", "Network Analysis", NodeType.PROCESSING, network_analysis),
            DAGNode("synthesis", "Results Synthesis", NodeType.AGGREGATION, synthesis_operation)
        ]
        
        for node in nodes:
            self.add_node(node)
        
        # Fan-out from input to three parallel analyses
        for analysis_node in ["stats_analysis", "content_analysis", "network_analysis"]:
            self.add_edge("input", analysis_node)
        
        # Fan-in from all analyses to synthesis
        for analysis_node in ["stats_analysis", "content_analysis", "network_analysis"]:
            self.add_edge(analysis_node, "synthesis")


class ConditionalWorkflowDAG(ResearchDAG):
    """
    Demonstrates conditional workflow with quality gates.
    
    Shows how DAGs can branch based on intermediate results.
    """
    
    def __init__(self):
        super().__init__("conditional_workflow", "Conditional research workflow")
        self._build_conditional_pipeline()
    
    def _build_conditional_pipeline(self):
        """Build workflow with conditional branches based on data quality"""
        
        def initial_analysis(inputs):
            quality_score = random_uniform(0.3, 0.9)
            return {
                "data_quality": quality_score,
                "sample_size": random_randint(20, 100),
                "recommendation": "detailed" if quality_score > 0.6 else "basic"
            }
        
        def quality_gate(inputs):
            """Decision node that determines workflow path"""
            analysis = inputs.get('initial_analysis', {})
            quality = analysis.get('data_quality', 0.5)
            
            return {
                "gate_passed": quality > 0.6,
                "quality_level": "high" if quality > 0.8 else "medium" if quality > 0.6 else "low",
                "next_step": "detailed_analysis" if quality > 0.6 else "basic_analysis"
            }
        
        def detailed_analysis(inputs):
            time.sleep(0.4)
            return {
                "analysis_depth": "comprehensive",
                "methods_used": ["advanced_stats", "ml_models", "validation"],
                "confidence": 0.85,
                "detailed_findings": ["finding1", "finding2", "finding3"]
            }
        
        def basic_analysis(inputs):
            time.sleep(0.2)
            return {
                "analysis_depth": "basic",
                "methods_used": ["descriptive_stats"],
                "confidence": 0.65,
                "basic_findings": ["finding1", "finding2"]
            }
        
        def final_report(inputs):
            """Combines results from either analysis path"""
            gate_result = inputs.get('quality_gate', {})
            
            # Check which analysis was performed
            if 'detailed_analysis' in inputs:
                analysis_result = inputs['detailed_analysis']
                analysis_type = "detailed"
            elif 'basic_analysis' in inputs:
                analysis_result = inputs['basic_analysis']
                analysis_type = "basic"
            else:
                analysis_result = {}
                analysis_type = "unknown"
            
            return {
                "report_type": analysis_type,
                "analysis_path_taken": gate_result.get('next_step', 'unknown'),
                "final_confidence": analysis_result.get('confidence', 0.5),
                "report_summary": f"Completed {analysis_type} analysis based on data quality"
            }
        
        # Create nodes
        nodes = [
            DAGNode("initial_analysis", "Initial Analysis", NodeType.PROCESSING, initial_analysis),
            DAGNode("quality_gate", "Quality Gate", NodeType.DECISION, quality_gate),
            DAGNode("detailed_analysis", "Detailed Analysis", NodeType.PROCESSING, detailed_analysis),
            DAGNode("basic_analysis", "Basic Analysis", NodeType.PROCESSING, basic_analysis),
            DAGNode("final_report", "Final Report", NodeType.OUTPUT, final_report)
        ]
        
        for node in nodes:
            self.add_node(node)
        
        # Create conditional flow
        # Note: In a real system, the execution logic would handle conditional routing
        # Here we show the structure - both paths exist but only one would execute
        self.add_edge("initial_analysis", "quality_gate")
        self.add_edge("quality_gate", "detailed_analysis")
        self.add_edge("quality_gate", "basic_analysis")
        self.add_edge("detailed_analysis", "final_report")
        self.add_edge("basic_analysis", "final_report")
    

# Convenience functions for creating research DAGs

def create_simple_research_workflow(topic: str = "machine learning") -> SimpleResearchDAG:
    """Create a simple 5-stage research workflow"""
    config = ResearchConfig(
        topic=topic,
        keywords=[topic.lower(), "research", "analysis"],
        min_sources=8,
        quality_threshold=0.75
    )
    return SimpleResearchDAG(config)


def create_parallel_analysis_workflow(dataset: str = "survey_data") -> ParallelAnalysisDAG:
    """Create a parallel analysis workflow with fan-out/fan-in pattern"""
    return ParallelAnalysisDAG(dataset)


def create_conditional_workflow() -> ConditionalWorkflowDAG:
    """Create a conditional workflow with quality gates"""
    return ConditionalWorkflowDAG()


def demonstrate_research_patterns():
    """Demonstrate the three main research DAG patterns"""
    
    print("📚 RESEARCH DAG PATTERNS DEMONSTRATION")
    print("=" * 50)
    
    # Pattern 1: Simple Linear Pipeline
    print("\n🔄 Pattern 1: Simple Linear Research Pipeline")
    simple_dag = create_simple_research_workflow("artificial intelligence")
    print(f"   • Nodes: {len(simple_dag.nodes)} (linear sequence)")
    print(f"   • Phases: Planning → Data Collection → Analysis → Validation → Report")
    
    results = simple_dag.execute()
    stats = simple_dag.get_statistics()
    print(f"   • Execution: {stats['success_rate']:.1%} success rate")
    
    # Pattern 2: Parallel Processing  
    print("\n⚡ Pattern 2: Parallel Analysis (Fan-out/Fan-in)")
    parallel_dag = create_parallel_analysis_workflow("research_corpus")
    print(f"   • Nodes: {len(parallel_dag.nodes)} (parallel branches)")
    print(f"   • Structure: Input → [Statistical|Content|Network] → Synthesis")
    
    parallel_results = parallel_dag.execute()
    parallel_stats = parallel_dag.get_statistics()
    print(f"   • Execution: {parallel_stats['success_rate']:.1%} success rate")
    
    # Show parallelization benefit
    simple_time = sum(node.execution_time for node in simple_dag.nodes.values())
    parallel_time = max([node.execution_time for node in parallel_dag.nodes.values() 
                        if node.node_type != NodeType.INPUT])
    
    print(f"\n⚡ Parallelization Benefit:")
    print(f"   • Linear pipeline: {simple_time:.2f}s total")
    print(f"   • Parallel pipeline: {parallel_time:.2f}s (max branch)")
    if simple_time > 0 and parallel_time > 0:
        speedup = simple_time / parallel_time
        print(f"   • Theoretical speedup: {speedup:.1f}x")
    
    # Pattern 3: Conditional Workflow
    print("\n🌳 Pattern 3: Conditional Workflow with Quality Gates")
    conditional_dag = create_conditional_workflow()
    print(f"   • Nodes: {len(conditional_dag.nodes)} (conditional branches)")
    print(f"   • Logic: Quality Gate determines analysis depth")
    
    conditional_results = conditional_dag.execute()
    
    # Show which path was taken
    if 'final_report' in conditional_results:
        report = conditional_results['final_report']
        path_taken = report.get('analysis_path_taken', 'unknown')
        confidence = report.get('final_confidence', 0)
        print(f"   • Path taken: {path_taken}")
        print(f"   • Final confidence: {confidence:.2f}")
    
    print(f"\n📊 All three patterns ready for visualization!")
    print("Use DAGVisualizer to plot these workflows and see the structural differences.")


if __name__ == "__main__":
    demonstrate_research_patterns()
