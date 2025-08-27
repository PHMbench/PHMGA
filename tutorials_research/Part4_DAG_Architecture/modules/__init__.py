"""
Part 4: DAG Architecture Modules

Supporting modules for understanding and implementing
Directed Acyclic Graphs in research workflows.
Dependency-free implementation that works in any Python environment.
"""

from .dag_fundamentals import ResearchDAG, DAGNode, NodeType, ExecutionStatus
from .research_pipeline_dag import (
    SimpleResearchDAG, ParallelAnalysisDAG, ConditionalWorkflowDAG,
    create_simple_research_workflow, create_parallel_analysis_workflow,
    create_conditional_workflow, demonstrate_research_patterns
)
from .phm_dag_structure import (
    SimplePHMDAG, ParallelPHMDAG, PHMConfig,
    create_simple_phm_workflow, create_parallel_phm_workflow,
    demonstrate_phm_dag_patterns
)

# Try to import visualization tools if available
try:
    from .dag_visualization import (
        DAGVisualizer, LayoutType, quick_plot,
        compare_execution_modes, execution_timeline
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

__all__ = [
    # Core DAG components
    'ResearchDAG', 'DAGNode', 'NodeType', 'ExecutionStatus',
    
    # Simplified research DAG patterns
    'SimpleResearchDAG', 'ParallelAnalysisDAG', 'ConditionalWorkflowDAG',
    'create_simple_research_workflow', 'create_parallel_analysis_workflow',
    'create_conditional_workflow', 'demonstrate_research_patterns',
    
    # PHM DAG patterns
    'SimplePHMDAG', 'ParallelPHMDAG', 'PHMConfig',
    'create_simple_phm_workflow', 'create_parallel_phm_workflow',
    'demonstrate_phm_dag_patterns',
    
    # Availability flags
    'VISUALIZATION_AVAILABLE'
]

# Add visualization tools if available
if VISUALIZATION_AVAILABLE:
    __all__.extend([
        'DAGVisualizer', 'LayoutType', 'quick_plot',
        'compare_execution_modes', 'execution_timeline'
    ])