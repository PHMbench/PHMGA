"""
DAG Fundamentals for Research Workflows

Core concepts and implementations for Directed Acyclic Graphs
in academic research and agent-based workflows.
Dependency-free implementation that works in any Python environment.
"""

# Smart import handling - use external libraries if available, otherwise use built-in implementations
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from enum import Enum
from typing import List, Dict, Any, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import uuid
import random
from collections import defaultdict, deque


# ====================================
# BUILT-IN GRAPH IMPLEMENTATION
# ====================================

class SimpleDAGraph:
    """
    Simple directed graph implementation using built-in Python data structures.
    Provides NetworkX-compatible interface for basic DAG operations.
    """
    
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.adjacency = defaultdict(set)  # node_id -> set of successors
        self.predecessors_dict = defaultdict(set)  # node_id -> set of predecessors
    
    def add_node(self, node_id):
        """Add a node to the graph"""
        self.nodes.add(node_id)
    
    def add_edge(self, from_node, to_node):
        """Add an edge between two nodes"""
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges.append((from_node, to_node))
        self.adjacency[from_node].add(to_node)
        self.predecessors_dict[to_node].add(from_node)
    
    def remove_edge(self, from_node, to_node):
        """Remove an edge between two nodes"""
        if (from_node, to_node) in self.edges:
            self.edges.remove((from_node, to_node))
            self.adjacency[from_node].discard(to_node)
            self.predecessors_dict[to_node].discard(from_node)
    
    def in_degree(self, node_id):
        """Get the in-degree of a node"""
        return len(self.predecessors_dict[node_id])
    
    def out_degree(self, node_id):
        """Get the out-degree of a node"""
        return len(self.adjacency[node_id])
    
    def successors(self, node_id):
        """Get successors of a node"""
        return list(self.adjacency[node_id])
    
    def predecessors(self, node_id):
        """Get predecessors of a node (NetworkX compatibility)"""
        return list(self.predecessors_dict[node_id])
    
    def predecessors_list(self, node_id):
        """Get predecessors of a node (alternative name)"""
        return list(self.predecessors_dict[node_id])


def is_directed_acyclic_graph(graph):
    """Check if graph is a DAG using DFS cycle detection"""
    if not NETWORKX_AVAILABLE:
        # Built-in cycle detection using DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in graph.nodes}
        
        def has_cycle(node):
            if colors[node] == GRAY:  # Back edge found
                return True
            if colors[node] == BLACK:  # Already processed
                return False
            
            colors[node] = GRAY
            for neighbor in graph.adjacency[node]:
                if has_cycle(neighbor):
                    return True
            colors[node] = BLACK
            return False
        
        return not any(has_cycle(node) for node in graph.nodes if colors[node] == WHITE)
    else:
        return nx.is_directed_acyclic_graph(graph)


def topological_sort(graph):
    """Topological sorting using Kahn's algorithm"""
    if not NETWORKX_AVAILABLE:
        # Built-in topological sort using Kahn's algorithm
        in_degree = {node: graph.in_degree(node) for node in graph.nodes}
        queue = deque([node for node in graph.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(graph.nodes):
            raise ValueError("Graph has cycles - cannot perform topological sort")
        
        return result
    else:
        return list(nx.topological_sort(graph))


def graph_density(graph):
    """Calculate graph density"""
    if not NETWORKX_AVAILABLE:
        n = len(graph.nodes)
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)  # For directed graph
        actual_edges = len(graph.edges)
        return actual_edges / max_edges
    else:
        return nx.density(graph)


class NetworkXError(Exception):
    """Exception compatible with NetworkX errors"""
    pass


# ====================================
# NODE AND DAG CLASSES
# ====================================

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
    
    Uses NetworkX when available, otherwise falls back to built-in graph implementation.
    Works in any Python environment.
    """
    
    def __init__(self, dag_id: str = None, description: str = ""):
        self.dag_id = dag_id or f"research_dag_{uuid.uuid4().hex[:8]}"
        self.description = description
        
        # Use NetworkX if available, otherwise use built-in implementation
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = SimpleDAGraph()
        
        self.nodes: Dict[str, DAGNode] = {}  # Node data storage
        self.execution_order: List[str] = []
        self.metadata = {
            "created_at": time.time(),
            "total_nodes": 0,
            "execution_count": 0,
            "networkx_available": NETWORKX_AVAILABLE,
            "matplotlib_available": MATPLOTLIB_AVAILABLE
        }
    
    def add_node(self, node: DAGNode) -> str:
        """Add a node to the DAG"""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        
        self.nodes[node.node_id] = node
        
        # Add node to graph with NetworkX attributes if available
        if NETWORKX_AVAILABLE:
            self.graph.add_node(node.node_id, 
                               name=node.name,
                               node_type=node.node_type.value,
                               status=node.status.value)
        else:
            self.graph.add_node(node.node_id)
        self.metadata["total_nodes"] = len(self.nodes)
        return node.node_id
    
    def add_edge(self, from_node_id: str, to_node_id: str):
        """Add a directed edge between two nodes"""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found")
        
        # Add edge to NetworkX graph
        self.graph.add_edge(from_node_id, to_node_id)
        
        # Validate no cycles are created
        if not is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_node_id, to_node_id)
            raise ValueError(f"Adding edge {from_node_id} -> {to_node_id} would create a cycle")
        
        # Update node relationships
        self.nodes[to_node_id].add_dependency(from_node_id)
        self.nodes[from_node_id].add_output_connection(to_node_id)
    
    def topological_sort(self) -> List[str]:
        """Get topological ordering"""
        try:
            self.execution_order = topological_sort(self.graph)
            return self.execution_order
        except (NetworkXError, ValueError) as e:
            raise ValueError("DAG contains cycles - cannot perform topological sort")
    
    def execute(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the entire DAG workflow using NetworkX ordering"""
        if not self.nodes:
            return {}
        
        execution_order = self.topological_sort()
        node_results = initial_inputs or {}
        
        print(f"🚀 Executing DAG: {self.dag_id}")
        print(f"📋 Execution order: {len(execution_order)} nodes")
        
        start_time = time.time()
        
        try:
            for node_id in execution_order:
                node = self.nodes[node_id]
                print(f"   🔄 Executing {node.name} ({node.node_type.value})...")
                
                # Prepare inputs from predecessors
                node_inputs = {}
                for pred_id in self.graph.predecessors(node_id):
                    if pred_id in node_results:
                        node_inputs[pred_id] = node_results[pred_id]
                
                try:
                    result = node.execute(node_inputs)
                    node_results[node_id] = result
                    print(f"      ✅ Completed in {node.execution_time:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                        raise
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
    
    def visualize_structure(self) -> str:
        """Generate a text representation of the DAG structure"""
        lines = [f"📊 DAG Structure: {self.dag_id}", "=" * 50]
        
        # Group by node types using NetworkX node attributes
        node_types = {}
        for node_id in self.graph.nodes:
            node_type = self.nodes[node_id].node_type
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node_id)
        
        for node_type, node_ids in node_types.items():
            lines.append(f"\n📋 {node_type.value.upper()} Nodes:")
            for node_id in node_ids:
                node = self.nodes[node_id]
                lines.append(f"   • {node.name} ({node_id})")
                
                # Show dependencies using NetworkX
                predecessors = list(self.graph.predecessors(node_id))
                if predecessors:
                    pred_names = [self.nodes[pred].name for pred in predecessors]
                    lines.append(f"     ⬅ Dependencies: {', '.join(pred_names)}")
                
                successors = list(self.graph.successors(node_id))
                if successors:
                    succ_names = [self.nodes[succ].name for succ in successors]
                    lines.append(f"     ➡ Outputs to: {', '.join(succ_names)}")
        
        lines.append(f"\n📈 Execution Order:")
        if self.execution_order:
            for i, node_id in enumerate(self.execution_order, 1):
                lines.append(f"   {i}. {self.nodes[node_id].name}")
        else:
            lines.append("   (Not yet calculated)")
        
        return "\n".join(lines)
    
    def plot_dag(self, figsize: Tuple[int, int] = (10, 8), save_path: str = None):
        """Plot the DAG - uses matplotlib if available, otherwise prints text visualization"""
        if MATPLOTLIB_AVAILABLE and NETWORKX_AVAILABLE:
            return self._plot_dag_matplotlib(figsize, save_path)
        else:
            return self._plot_dag_text()
    
    def _plot_dag_matplotlib(self, figsize, save_path):
        """Plot using matplotlib and NetworkX (full visualization)"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for different node types
        node_colors = {
            NodeType.INPUT: '#e8f5e8',
            NodeType.PROCESSING: '#fff2cc', 
            NodeType.DECISION: '#ffe6cc',
            NodeType.AGGREGATION: '#f0e6ff',
            NodeType.OUTPUT: '#ffe6e6',
            NodeType.VALIDATION: '#e6f3ff'
        }
        
        # Get node colors based on type
        colors = [node_colors.get(self.nodes[node_id].node_type, '#f0f0f0') 
                 for node_id in self.graph.nodes]
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=colors,
                              node_size=2000,
                              alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6, ax=ax)
        
        # Add labels
        labels = {node_id: self.nodes[node_id].name 
                 for node_id in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, 
                               font_size=10, font_weight='bold', ax=ax)
        
        # Add title and legend
        ax.set_title(f"DAG Structure: {self.dag_id}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=node_type.value.title()) 
                          for node_type, color in node_colors.items() 
                          if any(self.nodes[n].node_type == node_type for n in self.graph.nodes)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_dag_text(self):
        """Text-based DAG visualization when matplotlib is not available"""
        print(f"📊 DAG Visualization: {self.dag_id}")
        print("=" * 50)
        print("Note: Install matplotlib + networkx for enhanced visual plots")
        print("=" * 50)
        
        # Show the structure using existing visualize_structure method
        return self.visualize_structure()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DAG execution statistics using NetworkX"""
        stats = {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "nodes_by_type": {},
            "execution_times": {},
            "success_rate": 0.0,
            "graph_metrics": {
                "is_dag": is_directed_acyclic_graph(self.graph),
                "number_of_edges": len(self.graph.edges),
                "density": graph_density(self.graph)
            }
        }
        
        # Count nodes by type and execution stats
        for node in self.nodes.values():
            node_type = node.node_type.value
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
            
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
    
    Paper Review Pipeline: Input → Search → Filter → Analysis → Output
    """
    dag = ResearchDAG("paper_review", "Simple paper review workflow")
    
    # Simplified operations
    def input_operation(inputs):
        return {"topic": "machine learning", "keywords": ["ML", "neural networks"]}
    
    def search_operation(inputs):
        time.sleep(0.3)  # Simulate search
        topic = inputs.get('input', {}).get('topic', 'unknown')
        return {"papers": [f"Paper {i+1}: {topic} research" for i in range(4)]}
    
    def filter_operation(inputs):
        papers = inputs.get('search', {}).get('papers', [])
        return {"relevant_papers": papers[:2]}  # Keep top 2
    
    def analysis_operation(inputs):
        papers = inputs.get('filter', {}).get('relevant_papers', [])
        return {"insights": [f"Key insight from {paper}" for paper in papers]}
    
    def output_operation(inputs):
        insights = inputs.get('analysis', {}).get('insights', [])
        return {"summary": f"Review complete: {len(insights)} insights found"}
    
    # Create and connect nodes
    nodes = [
        DAGNode("input", "Topic Input", NodeType.INPUT, input_operation),
        DAGNode("search", "Paper Search", NodeType.PROCESSING, search_operation),
        DAGNode("filter", "Relevance Filter", NodeType.PROCESSING, filter_operation),
        DAGNode("analysis", "Content Analysis", NodeType.PROCESSING, analysis_operation),
        DAGNode("output", "Summary Report", NodeType.OUTPUT, output_operation)
    ]
    
    for node in nodes:
        dag.add_node(node)
    
    # Linear pipeline edges
    edges = [("input", "search"), ("search", "filter"), ("filter", "analysis"), ("analysis", "output")]
    for from_id, to_id in edges:
        dag.add_edge(from_id, to_id)
    
    return dag


def create_parallel_research_dag() -> ResearchDAG:
    """Create a DAG with parallel processing branches"""
    dag = ResearchDAG("parallel_analysis", "Parallel analysis workflow")
    
    def input_op(inputs):
        return {"dataset": "research_papers.csv", "size": 100}
    
    def stats_analysis(inputs):
        time.sleep(0.2)
        return {"stats": "descriptive statistics computed"}
    
    def trend_analysis(inputs):
        time.sleep(0.3)
        return {"trends": "trend analysis completed"}
    
    def synthesis_op(inputs):
        stats = inputs.get('stats_branch', {})
        trends = inputs.get('trend_branch', {})
        return {"final_report": f"Combined analysis: {len(stats)} + {len(trends)} components"}
    
    # Create nodes
    nodes = [
        DAGNode("input", "Data Input", NodeType.INPUT, input_op),
        DAGNode("stats_branch", "Statistical Analysis", NodeType.PROCESSING, stats_analysis),
        DAGNode("trend_branch", "Trend Analysis", NodeType.PROCESSING, trend_analysis),
        DAGNode("synthesis", "Results Synthesis", NodeType.AGGREGATION, synthesis_op)
    ]
    
    for node in nodes:
        dag.add_node(node)
    
    # Fan-out and fan-in edges
    dag.add_edge("input", "stats_branch")
    dag.add_edge("input", "trend_branch")
    dag.add_edge("stats_branch", "synthesis")
    dag.add_edge("trend_branch", "synthesis")
    
    return dag


def demonstrate_dag_fundamentals():
    """Demonstrate core DAG concepts with NetworkX and visualization"""
    
    print("🔬 DAG FUNDAMENTALS WITH NETWORKX")
    print("=" * 40)
    
    print("\n🎯 Core DAG Concepts:")
    concepts = [
        "• Directed: Edges have direction (dependencies flow one way)",
        "• Acyclic: No circular dependencies (NetworkX validation)",
        "• Task Dependencies: Topological ordering via NetworkX",
        "• Parallel Execution: Independent branches run simultaneously",
        "• Visual Plotting: NetworkX + matplotlib integration"
    ]
    
    for concept in concepts:
        print(f"   {concept}")
    
    # Demonstrate simple linear DAG
    print("\n🏗️ Example 1: Simple Linear Pipeline")
    simple_dag = create_simple_research_dag()
    print(simple_dag.visualize_structure())
    
    print("\n🚀 Executing simple pipeline...")
    results = simple_dag.execute()
    
    # Show statistics
    stats = simple_dag.get_statistics()
    print(f"\n📊 NetworkX Statistics:")
    print(f"   • Total nodes: {stats['total_nodes']}")
    print(f"   • Total edges: {stats['total_edges']}")
    print(f"   • Is DAG: {stats['graph_metrics']['is_dag']}")
    print(f"   • Graph density: {stats['graph_metrics']['density']:.2f}")
    print(f"   • Success rate: {stats['success_rate']:.1%}")
    
    # Demonstrate parallel DAG
    print("\n🌳 Example 2: Parallel Processing DAG")
    parallel_dag = create_parallel_research_dag()
    print(parallel_dag.visualize_structure())
    
    print("\n🚀 Executing parallel pipeline...")
    parallel_results = parallel_dag.execute()
    
    # Show comparison
    print(f"\n⚡ Execution Comparison:")
    simple_time = sum(node.execution_time for node in simple_dag.nodes.values())
    parallel_time = max([node.execution_time for node in parallel_dag.nodes.values()
                        if node.node_type != NodeType.INPUT])  # Excluding input
    
    print(f"   • Linear pipeline: {simple_time:.2f}s (sequential)")
    print(f"   • Parallel pipeline: {parallel_time:.2f}s (max branch time)")
    
    if simple_time > 0:
        speedup = simple_time / parallel_time if parallel_time > 0 else 1
        print(f"   • Theoretical speedup: {speedup:.1f}x")
    
    print("\n📊 Both DAGs ready for visualization with .plot_dag() method")


if __name__ == "__main__":
    demonstrate_dag_fundamentals()