"""
DAG Fundamentals for Research Workflows

Core concepts and implementations for Directed Acyclic Graphs
in academic research and agent-based workflows.
"""

from enum import Enum
from typing import List, Dict, Any, Set, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import uuid
from collections import defaultdict, deque


class NodeType(Enum):
    """Types of nodes in research workflow DAGs"""
    INPUT = "input"              # Data input nodes
    PROCESSING = "processing"    # Data processing nodes  
    DECISION = "decision"        # Decision/branching nodes
    AGGREGATION = "aggregation"  # Data combination nodes
    OUTPUT = "output"           # Result output nodes
    VALIDATION = "validation"    # Quality check nodes


class ExecutionStatus(Enum):
    """Execution status for DAG nodes"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """
    Individual node in a research workflow DAG.
    
    Represents a single operation, decision point, or data transformation
    in the research process.
    """
    
    node_id: str
    name: str
    node_type: NodeType
    operation: Callable = None
    dependencies: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    
    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None
    result: Any = None
    
    # Research-specific metadata
    research_phase: str = "analysis"
    required_resources: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate unique ID if not provided"""
        if not self.node_id:
            self.node_id = f"{self.node_type.value}_{uuid.uuid4().hex[:8]}"
    
    def add_dependency(self, node_id: str):
        """Add a dependency to this node"""
        self.dependencies.add(node_id)
    
    def add_output_connection(self, node_id: str):
        """Add an output connection from this node"""
        self.outputs.add(node_id)
    
    def is_ready_to_execute(self, completed_nodes: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return self.dependencies.issubset(completed_nodes)
    
    def execute(self, inputs: Dict[str, Any]) -> Any:
        """Execute the node operation"""
        if not self.operation:
            return inputs  # Pass-through node
        
        start_time = time.time()
        self.status = ExecutionStatus.RUNNING
        
        try:
            # Execute the operation
            result = self.operation(inputs)
            self.result = result
            self.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            self.error_message = str(e)
            self.status = ExecutionStatus.FAILED
            raise
        
        finally:
            self.execution_time = time.time() - start_time
        
        return self.result


class ResearchDAG:
    """
    Directed Acyclic Graph for research workflows.
    
    Manages complex research processes with dependencies,
    parallel execution, and conditional branching.
    """
    
    def __init__(self, dag_id: str = None, description: str = ""):
        self.dag_id = dag_id or f"research_dag_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.nodes: Dict[str, DAGNode] = {}
        self.execution_order: List[str] = []
        self.metadata = {
            "created_at": time.time(),
            "total_nodes": 0,
            "execution_count": 0
        }
    
    def add_node(self, node: DAGNode) -> str:
        """Add a node to the DAG"""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        
        self.nodes[node.node_id] = node
        self.metadata["total_nodes"] = len(self.nodes)
        return node.node_id
    
    def add_edge(self, from_node_id: str, to_node_id: str):
        """Add a directed edge between two nodes"""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found")
        
        # Add dependency relationship
        self.nodes[to_node_id].add_dependency(from_node_id)
        self.nodes[from_node_id].add_output_connection(to_node_id)
        
        # Validate no cycles are created
        if self._has_cycle():
            # Undo the edge addition
            self.nodes[to_node_id].dependencies.remove(from_node_id)
            self.nodes[from_node_id].outputs.remove(to_node_id)
            raise ValueError(f"Adding edge {from_node_id} -> {to_node_id} would create a cycle")
    
    def _has_cycle(self) -> bool:
        """Check if the DAG contains cycles using DFS"""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            colors[node_id] = GRAY
            
            for output_id in self.nodes[node_id].outputs:
                if colors[output_id] == GRAY:  # Back edge found
                    return True
                if colors[output_id] == WHITE and dfs(output_id):
                    return True
            
            colors[node_id] = BLACK
            return False
        
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sorting to determine execution order.
        
        Returns:
            List of node IDs in topological order
        """
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = len(self.nodes[node_id].dependencies)
        
        # Initialize queue with nodes that have no dependencies
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        execution_order = []
        
        while queue:
            current_node = queue.popleft()
            execution_order.append(current_node)
            
            # Reduce in-degree for all output nodes
            for output_node in self.nodes[current_node].outputs:
                in_degree[output_node] -= 1
                if in_degree[output_node] == 0:
                    queue.append(output_node)
        
        # Check if all nodes were processed (no cycles)
        if len(execution_order) != len(self.nodes):
            raise ValueError("DAG contains cycles - cannot perform topological sort")
        
        self.execution_order = execution_order
        return execution_order
    
    def execute(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the entire DAG workflow.
        
        Args:
            initial_inputs: Initial data for input nodes
            
        Returns:
            Dictionary of all node results
        """
        if not self.nodes:
            return {}
        
        # Determine execution order
        execution_order = self.topological_sort()
        
        # Track execution state
        node_results = initial_inputs or {}
        completed_nodes = set()
        
        print(f"🚀 Executing DAG: {self.dag_id}")
        print(f"📋 Execution order: {len(execution_order)} nodes")
        
        start_time = time.time()
        
        try:
            for node_id in execution_order:
                node = self.nodes[node_id]
                
                print(f"   🔄 Executing {node.name} ({node.node_type.value})...")
                
                # Prepare inputs from dependencies
                node_inputs = {}
                for dep_id in node.dependencies:
                    if dep_id in node_results:
                        node_inputs[dep_id] = node_results[dep_id]
                
                # Execute the node
                try:
                    result = node.execute(node_inputs)
                    node_results[node_id] = result
                    completed_nodes.add(node_id)
                    
                    print(f"      ✅ Completed in {node.execution_time:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    # Handle failure based on node criticality
                    if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                        raise  # Critical nodes must succeed
                    else:
                        # Mark as failed but continue execution
                        node.status = ExecutionStatus.FAILED
                        continue
            
            execution_time = time.time() - start_time
            self.metadata["execution_count"] += 1
            
            print(f"✅ DAG execution completed in {execution_time:.2f}s")
            print(f"📊 Results: {len(node_results)} nodes processed")
            
            return node_results
            
        except Exception as e:
            print(f"❌ DAG execution failed: {e}")
            raise
    
    def get_ready_nodes(self, completed_nodes: Set[str]) -> List[DAGNode]:
        """Get nodes that are ready for execution"""
        ready_nodes = []
        for node_id, node in self.nodes.items():
            if (node.status == ExecutionStatus.PENDING and 
                node.is_ready_to_execute(completed_nodes)):
                ready_nodes.append(node)
        return ready_nodes
    
    def visualize_structure(self) -> str:
        """Generate a text representation of the DAG structure"""
        lines = [f"📊 DAG Structure: {self.dag_id}"]
        lines.append("=" * 50)
        
        # Sort nodes by type for better organization
        nodes_by_type = defaultdict(list)
        for node in self.nodes.values():
            nodes_by_type[node.node_type].append(node)
        
        for node_type, type_nodes in nodes_by_type.items():
            lines.append(f"\\n📋 {node_type.value.upper()} Nodes:")
            for node in type_nodes:
                lines.append(f"   • {node.name} ({node.node_id})")
                if node.dependencies:
                    deps = [self.nodes[dep_id].name for dep_id in node.dependencies]
                    lines.append(f"     ⬅ Dependencies: {', '.join(deps)}")
                if node.outputs:
                    outs = [self.nodes[out_id].name for out_id in node.outputs]
                    lines.append(f"     ➡ Outputs to: {', '.join(outs)}")
        
        lines.append(f"\\n📈 Execution Order:")
        if self.execution_order:
            for i, node_id in enumerate(self.execution_order, 1):
                node_name = self.nodes[node_id].name
                lines.append(f"   {i}. {node_name}")
        else:
            lines.append("   (Not yet calculated)")
        
        return "\\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DAG execution statistics"""
        stats = {
            "total_nodes": len(self.nodes),
            "nodes_by_type": {},
            "total_dependencies": 0,
            "execution_times": {},
            "success_rate": 0.0
        }
        
        # Count nodes by type
        for node in self.nodes.values():
            node_type = node.node_type.value
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
            stats["total_dependencies"] += len(node.dependencies)
            
            if node.execution_time > 0:
                stats["execution_times"][node.node_id] = node.execution_time
        
        # Calculate success rate
        completed_nodes = sum(1 for node in self.nodes.values() 
                            if node.status == ExecutionStatus.COMPLETED)
        if self.nodes:
            stats["success_rate"] = completed_nodes / len(self.nodes)
        
        return stats


def create_simple_research_dag() -> ResearchDAG:
    """
    Create a simple research DAG for demonstration.
    
    This example represents a basic literature search workflow:
    Input → Search → Filter → Analysis → Output
    """
    
    dag = ResearchDAG("simple_research", "Basic literature search workflow")
    
    # Define operations for each node
    def input_operation(inputs):
        return {"research_topic": "quantum error correction", "search_terms": ["quantum", "error correction", "fault tolerance"]}
    
    def search_operation(inputs):
        time.sleep(0.5)  # Simulate search time
        return {"found_papers": [f"Paper {i+1} on {inputs['input'].get('research_topic', 'unknown topic')}" for i in range(5)]}
    
    def filter_operation(inputs):
        papers = inputs['search'].get('found_papers', [])
        return {"relevant_papers": papers[:3]}  # Filter to top 3
    
    def analysis_operation(inputs):
        papers = inputs['filter'].get('relevant_papers', [])
        return {"key_findings": [f"Finding from {paper}" for paper in papers]}
    
    def output_operation(inputs):
        findings = inputs['analysis'].get('key_findings', [])
        return {"research_summary": f"Analysis of {len(findings)} key findings completed"}
    
    # Create nodes
    input_node = DAGNode("input", "Research Topic Input", NodeType.INPUT, input_operation)
    search_node = DAGNode("search", "Literature Search", NodeType.PROCESSING, search_operation)
    filter_node = DAGNode("filter", "Relevance Filtering", NodeType.PROCESSING, filter_operation) 
    analysis_node = DAGNode("analysis", "Content Analysis", NodeType.PROCESSING, analysis_operation)
    output_node = DAGNode("output", "Research Summary", NodeType.OUTPUT, output_operation)
    
    # Add nodes to DAG
    dag.add_node(input_node)
    dag.add_node(search_node)
    dag.add_node(filter_node)
    dag.add_node(analysis_node)
    dag.add_node(output_node)
    
    # Add dependencies (edges)
    dag.add_edge("input", "search")
    dag.add_edge("search", "filter")
    dag.add_edge("filter", "analysis")
    dag.add_edge("analysis", "output")
    
    return dag


def demonstrate_dag_fundamentals():
    """Demonstrate core DAG concepts and operations"""
    
    print("🔬 DAG FUNDAMENTALS DEMONSTRATION")
    print("=" * 40)
    
    print("\\n🎯 Core DAG Concepts:")
    concepts = [
        "• Directed: Edges have direction (dependencies flow one way)",
        "• Acyclic: No circular dependencies allowed",
        "• Task Dependencies: Nodes must wait for dependencies",
        "• Parallel Execution: Independent nodes can run simultaneously",
        "• Topological Ordering: Determines safe execution sequence"
    ]
    
    for concept in concepts:
        print(f"   {concept}")
    
    print("\\n🏗️ Creating Simple Research DAG...")
    dag = create_simple_research_dag()
    
    # Visualize structure
    print(dag.visualize_structure())
    
    print("\\n🚀 Executing DAG workflow...")
    try:
        results = dag.execute()
        
        print("\\n📊 Execution Results:")
        for node_id, result in results.items():
            node_name = dag.nodes[node_id].name
            print(f"   • {node_name}: {result}")
        
        # Show statistics
        stats = dag.get_statistics()
        print(f"\\n📈 DAG Statistics:")
        print(f"   • Total nodes: {stats['total_nodes']}")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Total dependencies: {stats['total_dependencies']}")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")


if __name__ == "__main__":
    demonstrate_dag_fundamentals()