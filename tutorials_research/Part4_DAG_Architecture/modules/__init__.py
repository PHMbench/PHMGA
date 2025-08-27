"""
Part 4: DAG Architecture Modules

Supporting modules for understanding and implementing
Directed Acyclic Graphs in research workflows.
"""

from .dag_fundamentals import ResearchDAG, DAGNode, NodeType
from .research_pipeline_dag import LiteratureReviewDAG, ResearchPipeline
from .phm_dag_structure import PHMSignalProcessingDAG, SignalProcessingNode

__all__ = [
    'ResearchDAG', 'DAGNode', 'NodeType',
    'LiteratureReviewDAG', 'ResearchPipeline', 
    'PHMSignalProcessingDAG', 'SignalProcessingNode'
]